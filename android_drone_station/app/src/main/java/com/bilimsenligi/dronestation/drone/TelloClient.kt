package com.bilimsenligi.dronestation.drone

import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.net.SocketException
import java.net.SocketTimeoutException
import java.nio.charset.StandardCharsets
import java.util.concurrent.atomic.AtomicReference
import java.net.InetSocketAddress

class TelloClient(
    private val telloIp: String = "192.168.10.1",
    private val commandPort: Int = 8889,
    private val statePort: Int = 8890,
) {

    data class TelloState(
        val raw: String,
        val batteryPercent: Int? = null,
        val flightTimeSec: Int? = null,
        val heightCm: Int? = null,
        val receivedAtMs: Long = System.currentTimeMillis(),
    )

    interface Listener {
        fun onLog(message: String)
        fun onState(state: TelloState)
    }

    @Volatile
    private var commandSocket: DatagramSocket? = null

    // Dedicated RC socket so high-rate `rc x y z w` packets never contend with the
    // command socket's `sendLock` (held for up to 2s during battery?/takeoff/streamon/etc.)
    // and never get delayed by drainPendingResponses or keep-alive receives.
    @Volatile
    private var rcSocket: DatagramSocket? = null

    @Volatile
    private var stateSocket: DatagramSocket? = null

    @Volatile
    private var stateThread: Thread? = null

    @Volatile
    private var connected: Boolean = false

    @Volatile
    private var listener: Listener? = null

    private val telloAddress: InetAddress = InetAddress.getByName(telloIp)
    private val telloSocketAddress: InetSocketAddress = InetSocketAddress(telloAddress, commandPort)
    private val sendLock = Any()
    private val lastStateRef = AtomicReference<TelloState?>(null)
    // Pre-allocated buffer for the RC command path: avoids per-tick ByteArray/DatagramPacket
    // garbage at 20 Hz and keeps the path lock-free aside from the StringBuilder.
    private val rcBuilder = StringBuilder(32)
    private val rcBuffer = ByteArray(48)

    fun setListener(listener: Listener?) {
        this.listener = listener
    }

    fun isConnected(): Boolean = connected

    fun latestState(): TelloState? = lastStateRef.get()

    fun connect(): Boolean {
        if (connected) return true
        return try {
            commandSocket = DatagramSocket().apply {
                soTimeout = 2000
            }
            // RC socket is connected-mode UDP: skips per-send route lookup and lets the
            // OS reject any unrelated inbound traffic. We never read from it.
            rcSocket = DatagramSocket().apply {
                connect(telloSocketAddress)
            }
            connected = false

            var commandResponse: String? = null
            repeat(3) {
                commandResponse = sendCommand("command", timeoutMs = 1500)
                if (commandResponse != null) return@repeat
                Thread.sleep(180)
            }

            val commandOk = commandResponse?.equals("ok", ignoreCase = true) == true
            val commandAlready = commandResponse?.contains("error", ignoreCase = true) == true ||
                commandResponse?.contains("unactive", ignoreCase = true) == true

            // Some Tello firmwares can respond non-ok even while command channel is usable.
            // We verify by battery query before deciding failure.
            val commandUsable = if (commandOk || commandAlready) {
                true
            } else {
                sendCommand("battery?", timeoutMs = 1500)?.toIntOrNull() != null
            }

            if (!commandUsable) {
                log("[TELLO] command mode girisi basarisiz (resp=$commandResponse)")
                disconnect()
                return false
            }

            connected = true
            startStateListener()

            log("[TELLO] command mode aktif (resp=${commandResponse ?: "battery-fallback"})")
            val streamResp = sendCommand("streamon")
            if (streamResp == null) {
                log("[TELLO] streamon cevap vermedi (fatal degil)")
            }
            true
        } catch (e: Exception) {
            log("[TELLO] baglanti hatasi: ${e.message}")
            disconnect()
            false
        }
    }

    fun disconnect() {
        connected = false

        try {
            stateThread?.interrupt()
        } catch (_: Exception) {
        }
        stateThread = null

        try {
            stateSocket?.close()
        } catch (_: Exception) {
        }
        stateSocket = null

        try {
            commandSocket?.close()
        } catch (_: Exception) {
        }
        commandSocket = null

        try {
            rcSocket?.close()
        } catch (_: Exception) {
        }
        rcSocket = null

        log("[TELLO] baglanti kapatildi")
    }

    fun sendCommand(command: String, timeoutMs: Int = 2000): String? {
        val socket = commandSocket ?: return null
        synchronized(sendLock) {
            return try {
                drainPendingResponses(socket)
                socket.soTimeout = timeoutMs
                val payload = command.toByteArray(StandardCharsets.UTF_8)
                val packet = DatagramPacket(payload, payload.size, telloAddress, commandPort)
                socket.send(packet)

                val buffer = ByteArray(1024)
                val recv = DatagramPacket(buffer, buffer.size)
                socket.receive(recv)
                String(recv.data, 0, recv.length, StandardCharsets.UTF_8).trim().also {
                    log("[TELLO] <$command> -> $it")
                }
            } catch (_: SocketTimeoutException) {
                // Timeout can happen intermittently on Tello; caller decides retry/fallback.
                null
            } catch (_: SocketException) {
                null
            } catch (e: Exception) {
                log("[TELLO] komut hatasi <$command>: ${e.message}")
                null
            }
        }
    }

    fun sendCommandNoWait(command: String): Boolean {
        val socket = commandSocket ?: return false
        if (!connected) return false
        synchronized(sendLock) {
            return try {
                drainPendingResponses(socket)
                val payload = command.toByteArray(StandardCharsets.UTF_8)
                val packet = DatagramPacket(payload, payload.size, telloAddress, commandPort)
                socket.send(packet)
                // Avoid stale responses polluting future command reads.
                val oldTimeout = socket.soTimeout
                try {
                    socket.soTimeout = 120
                    val buffer = ByteArray(1024)
                    val recv = DatagramPacket(buffer, buffer.size)
                    socket.receive(recv)
                } catch (_: Exception) {
                } finally {
                    try {
                        socket.soTimeout = oldTimeout
                    } catch (_: Exception) {
                    }
                }
                true
            } catch (_: Exception) {
                false
            }
        }
    }

    fun sendRcControl(lr: Int, fb: Int, ud: Int, yaw: Int) {
        val socket = rcSocket ?: return
        if (!connected) return

        val clampedLr = lr.coerceIn(-100, 100)
        val clampedFb = fb.coerceIn(-100, 100)
        val clampedUd = ud.coerceIn(-100, 100)
        val clampedYaw = yaw.coerceIn(-100, 100)

        // Lock-free path: dedicated socket, dedicated buffer. Single-threaded caller
        // (rcScheduler) so no synchronization needed on the buffer either.
        val sb = rcBuilder
        sb.setLength(0)
        sb.append("rc ").append(clampedLr).append(' ')
            .append(clampedFb).append(' ')
            .append(clampedUd).append(' ')
            .append(clampedYaw)
        val length = sb.length
        for (i in 0 until length) {
            rcBuffer[i] = sb[i].code.toByte()
        }
        try {
            // Connected-mode UDP: address/port already bound, no DatagramPacket dest needed.
            socket.send(DatagramPacket(rcBuffer, length))
        } catch (_: Exception) {
        }
    }

    fun queryBattery(): Int? {
        val response = sendCommand("battery?") ?: return null
        return response.toIntOrNull()
    }

    fun takeoff(): Boolean = sendCommand("takeoff")?.equals("ok", ignoreCase = true) == true

    fun land(): Boolean = sendCommand("land")?.equals("ok", ignoreCase = true) == true

    fun streamOn(): Boolean = sendCommand("streamon")?.equals("ok", ignoreCase = true) == true

    fun streamOff(): Boolean = sendCommand("streamoff")?.equals("ok", ignoreCase = true) == true

    fun emergency(): Boolean = sendCommand("emergency")?.equals("ok", ignoreCase = true) == true

    fun flipForward(): Boolean = sendCommand("flip f")?.equals("ok", ignoreCase = true) == true

    fun flipBack(): Boolean = sendCommand("flip b")?.equals("ok", ignoreCase = true) == true

    fun flipLeft(): Boolean = sendCommand("flip l")?.equals("ok", ignoreCase = true) == true

    fun flipRight(): Boolean = sendCommand("flip r")?.equals("ok", ignoreCase = true) == true

    fun restartStateListener() {
        if (!connected) return
        try {
            stateThread?.interrupt()
        } catch (_: Exception) {
        }
        stateThread = null

        try {
            stateSocket?.close()
        } catch (_: Exception) {
        }
        stateSocket = null

        startStateListener()
    }

    private fun startStateListener() {
        if (stateThread?.isAlive == true) return

        val socket = DatagramSocket(null).apply {
            reuseAddress = true
            bind(InetSocketAddress(statePort))
            soTimeout = 1000
        }
        stateSocket = socket

        stateThread = Thread {
            val buf = ByteArray(2048)
            while (connected && !Thread.currentThread().isInterrupted) {
                try {
                    val packet = DatagramPacket(buf, buf.size)
                    socket.receive(packet)
                    val raw = String(packet.data, 0, packet.length, StandardCharsets.UTF_8).trim()
                    if (raw.isBlank()) continue

                    val parsed = parseState(raw)
                    lastStateRef.set(parsed)
                    listener?.onState(parsed)
                } catch (_: SocketException) {
                    break
                } catch (_: Exception) {
                }
            }
        }.apply {
            isDaemon = true
            name = "TelloStateListener"
            start()
        }
    }

    private fun parseState(raw: String): TelloState {
        val kv = mutableMapOf<String, String>()
        val parts = raw.split(";")
        for (part in parts) {
            val idx = part.indexOf(':')
            if (idx <= 0 || idx >= part.length - 1) continue
            val key = part.substring(0, idx).trim()
            val value = part.substring(idx + 1).trim()
            kv[key] = value
        }

        return TelloState(
            raw = raw,
            batteryPercent = kv["bat"]?.toIntOrNull(),
            flightTimeSec = kv["time"]?.toIntOrNull(),
            heightCm = kv["h"]?.toIntOrNull(),
            receivedAtMs = System.currentTimeMillis(),
        )
    }

    private fun log(message: String) {
        listener?.onLog(message)
    }

    private fun drainPendingResponses(socket: DatagramSocket) {
        val oldTimeout = socket.soTimeout
        try {
            socket.soTimeout = 1
            val buffer = ByteArray(1024)
            val packet = DatagramPacket(buffer, buffer.size)
            repeat(4) {
                try {
                    socket.receive(packet)
                } catch (_: Exception) {
                    return@repeat
                }
            }
        } catch (_: Exception) {
        } finally {
            try {
                socket.soTimeout = oldTimeout
            } catch (_: Exception) {
            }
        }
    }
}
