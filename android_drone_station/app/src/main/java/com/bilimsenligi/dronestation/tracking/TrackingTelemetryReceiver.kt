package com.bilimsenligi.dronestation.tracking

import org.json.JSONObject
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetSocketAddress
import java.net.SocketException
import java.nio.charset.StandardCharsets
import java.util.concurrent.atomic.AtomicBoolean

class TrackingTelemetryReceiver(
    private val listenPort: Int = 5005,
) {

    data class TrackingSample(
        val tx: Float,
        val ty: Float,
        val size: Float,
        val confidence: Float,
        val targetId: Int,
        val timestampMs: Long,
    )

    interface Listener {
        fun onTrackingSample(sample: TrackingSample)
        fun onTrackingLog(message: String)
    }

    @Volatile
    private var socket: DatagramSocket? = null

    @Volatile
    private var worker: Thread? = null

    @Volatile
    private var listener: Listener? = null

    private val running = AtomicBoolean(false)

    fun setListener(listener: Listener?) {
        this.listener = listener
    }

    fun start() {
        if (running.get()) return

        try {
            val s = DatagramSocket(null).apply {
                reuseAddress = true
                bind(InetSocketAddress(listenPort))
                soTimeout = 1000
            }
            socket = s
            running.set(true)

            worker = Thread {
                val buf = ByteArray(2048)
                while (running.get() && !Thread.currentThread().isInterrupted) {
                    try {
                        val packet = DatagramPacket(buf, buf.size)
                        s.receive(packet)
                        val text = String(packet.data, 0, packet.length, StandardCharsets.UTF_8)
                        val sample = parseSample(text)
                        if (sample != null) {
                            listener?.onTrackingSample(sample)
                        }
                    } catch (_: SocketException) {
                        break
                    } catch (_: Exception) {
                    }
                }
            }.apply {
                isDaemon = true
                name = "TrackingTelemetryReceiver"
                start()
            }

            listener?.onTrackingLog("[TRACK] telemetry dinleniyor :$listenPort")
        } catch (e: Exception) {
            listener?.onTrackingLog("[TRACK] telemetry baslatma hatasi: ${e.message}")
            stop()
        }
    }

    fun stop() {
        running.set(false)
        try {
            worker?.interrupt()
        } catch (_: Exception) {
        }
        worker = null

        try {
            socket?.close()
        } catch (_: Exception) {
        }
        socket = null
    }

    private fun parseSample(jsonText: String): TrackingSample? {
        return try {
            val o = JSONObject(jsonText)
            val tx = o.optDouble("tx", 0.0).toFloat()      // -1..1
            val ty = o.optDouble("ty", 0.0).toFloat()      // -1..1
            val size = o.optDouble("size", 0.0).toFloat()  // 0..1
            val conf = o.optDouble("conf", 0.0).toFloat()  // 0..1
            val id = o.optInt("id", -1)
            val ts = o.optLong("ts", System.currentTimeMillis())

            TrackingSample(
                tx = tx.coerceIn(-1f, 1f),
                ty = ty.coerceIn(-1f, 1f),
                size = size.coerceIn(0f, 1f),
                confidence = conf.coerceIn(0f, 1f),
                targetId = id,
                timestampMs = ts,
            )
        } catch (_: Exception) {
            null
        }
    }
}
