package com.bilimsenligi.dronestation.drone

import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.media.MediaCodec
import android.media.MediaFormat
import android.media.MediaMuxer
import android.os.SystemClock
import android.view.Surface
import android.view.TextureView
import java.io.ByteArrayOutputStream
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetSocketAddress
import java.net.SocketException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class TelloVideoManager(
    private val listenPort: Int = 11111,
    private val width: Int = 960,
    private val height: Int = 720,
) {

    interface Listener {
        fun onVideoLog(message: String)
        fun onSnapshotSaved(file: File)
        fun onRecordStarted(file: File)
        fun onRecordStopped(file: File)
    }

    @Volatile
    private var listener: Listener? = null

    @Volatile
    private var textureView: TextureView? = null

    @Volatile
    private var outputSurface: Surface? = null

    @Volatile
    private var decoder: MediaCodec? = null

    @Volatile
    private var socket: DatagramSocket? = null

    @Volatile
    private var receiverThread: Thread? = null

    private val running = AtomicBoolean(false)
    private val recording = AtomicBoolean(false)
    private val recordLock = Any()
    private val lifecycleLock = Any()
    private val frameAssembler = H264AccessUnitAssembler()
    private val remuxExecutor = Executors.newSingleThreadExecutor()
    @Volatile
    private var pendingStart = false

    @Volatile
    private var recordingFile: File? = null

    @Volatile
    private var recordingStream: BufferedOutputStream? = null

    fun setListener(listener: Listener?) {
        this.listener = listener
    }

    fun attachTextureView(view: TextureView) {
        textureView = view
        if (view.isAvailable) {
            prepareSurface(view.surfaceTexture)
        }
        view.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(surfaceTexture: SurfaceTexture, w: Int, h: Int) {
                prepareSurface(surfaceTexture)
                log("[VIDEO] surface hazir")
                retryPendingStart()
            }

            override fun onSurfaceTextureSizeChanged(surfaceTexture: SurfaceTexture, w: Int, h: Int) {
            }

            override fun onSurfaceTextureDestroyed(surfaceTexture: SurfaceTexture): Boolean {
                outputSurface?.release()
                outputSurface = null
                return true
            }

            override fun onSurfaceTextureUpdated(surfaceTexture: SurfaceTexture) {
            }
        }
    }

    fun start(): Boolean {
        synchronized(lifecycleLock) {
            if (running.get()) return true
            pendingStart = true

            val surface = outputSurface
            if (surface == null) {
                log("[VIDEO] surface hazir degil, goruntu beklemeye alindi")
                return false
            }

            return try {
                val codec = MediaCodec.createDecoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
                val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height)
                format.setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 1024 * 1024)
                codec.configure(format, surface, null, 0)
                codec.start()
                decoder = codec

                val udpSocket = DatagramSocket(null).apply {
                    reuseAddress = true
                    bind(InetSocketAddress(listenPort))
                    soTimeout = 1000
                    receiveBufferSize = 2 * 1024 * 1024
                }
                socket = udpSocket

                frameAssembler.reset()
                running.set(true)
                pendingStart = false
                startReceiverLoop()
                log("[VIDEO] stream receiver basladi")
                true
            } catch (e: Exception) {
                log("[VIDEO] baslatma hatasi: ${e.message}")
                stopInternal()
                false
            }
        }
    }

    fun stop() {
        synchronized(lifecycleLock) {
            stopInternal()
        }
    }

    fun isRunning(): Boolean = running.get()

    fun isRecording(): Boolean = recording.get()

    fun startRecording(outputDir: File?): File? {
        if (recording.get()) return recordingFile
        if (outputDir == null) {
            log("[VIDEO] recording klasoru yok")
            return null
        }

        if (!outputDir.exists()) {
            outputDir.mkdirs()
        }

        return try {
            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val outFile = File(outputDir, "tello_raw_$stamp.h264")
            val outStream = BufferedOutputStream(FileOutputStream(outFile))

            synchronized(recordLock) {
                recordingFile = outFile
                recordingStream = outStream
                recording.set(true)
            }

            listener?.onRecordStarted(outFile)
            outFile
        } catch (e: Exception) {
            log("[VIDEO] recording baslatma hatasi: ${e.message}")
            null
        }
    }

    fun stopRecording(): File? {
        if (!recording.get()) return recordingFile

        val outFile: File?
        synchronized(recordLock) {
            recording.set(false)
            outFile = recordingFile
            try {
                recordingStream?.flush()
            } catch (_: Exception) {
            }
            try {
                recordingStream?.close()
            } catch (_: Exception) {
            }
            recordingStream = null
            recordingFile = null
        }

        if (outFile != null) {
            listener?.onRecordStopped(outFile)
            convertRawRecordingToMp4Async(outFile)
        }
        return outFile
    }

    fun saveSnapshot(outputDir: File?): File? {
        val view = textureView ?: return null
        if (outputDir == null) return null

        if (!outputDir.exists()) {
            outputDir.mkdirs()
        }

        val bitmap = view.bitmap ?: return null

        return try {
            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val file = File(outputDir, "tello_snapshot_$stamp.jpg")
            FileOutputStream(file).use { fos ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 92, fos)
                fos.flush()
            }
            listener?.onSnapshotSaved(file)
            file
        } catch (e: Exception) {
            log("[VIDEO] snapshot kayit hatasi: ${e.message}")
            null
        } finally {
            bitmap.recycle()
        }
    }

    private fun startReceiverLoop() {
        val sock = socket ?: return
        val codec = decoder ?: return

        receiverThread = Thread {
            val packetBuffer = ByteArray(2048)
            val packet = DatagramPacket(packetBuffer, packetBuffer.size)

            while (running.get() && !Thread.currentThread().isInterrupted) {
                try {
                    sock.receive(packet)
                    val size = packet.length
                    if (size <= 0) continue

                    val accessUnits = frameAssembler.append(packet.data, size)
                    if (accessUnits.isEmpty()) continue

                    for (accessUnit in accessUnits) {
                        synchronized(recordLock) {
                            if (recording.get()) {
                                try {
                                    recordingStream?.write(accessUnit)
                                } catch (_: Exception) {
                                }
                            }
                        }

                        queueDecoderFrame(codec, accessUnit)
                        drainDecoder(codec)
                    }
                } catch (_: SocketException) {
                    break
                } catch (_: Exception) {
                }
            }
        }.apply {
            isDaemon = true
            name = "TelloVideoReceiver"
            priority = Thread.MAX_PRIORITY
            start()
        }
    }

    private fun queueDecoderFrame(codec: MediaCodec, frame: ByteArray) {
        if (frame.isEmpty()) return

        try {
            val inputIndex = codec.dequeueInputBuffer(0)
            if (inputIndex >= 0) {
                val inputBuffer = codec.getInputBuffer(inputIndex)
                if (inputBuffer != null) {
                    inputBuffer.clear()
                    if (inputBuffer.capacity() < frame.size) {
                        codec.queueInputBuffer(inputIndex, 0, 0, 0L, 0)
                        return
                    }
                    inputBuffer.put(frame)
                    val ptsUs = SystemClock.elapsedRealtimeNanos() / 1000L
                    codec.queueInputBuffer(inputIndex, 0, frame.size, ptsUs, 0)
                }
            }
        } catch (_: Exception) {
        }
    }

    private fun drainDecoder(codec: MediaCodec) {
        try {
            val info = MediaCodec.BufferInfo()
            var outputIndex = codec.dequeueOutputBuffer(info, 0)
            while (outputIndex != MediaCodec.INFO_TRY_AGAIN_LATER) {
                if (outputIndex >= 0) {
                    codec.releaseOutputBuffer(outputIndex, true)
                }
                outputIndex = codec.dequeueOutputBuffer(info, 0)
            }
        } catch (_: Exception) {
        }
    }

    private fun prepareSurface(surfaceTexture: SurfaceTexture?) {
        if (surfaceTexture == null) return
        try {
            surfaceTexture.setDefaultBufferSize(width, height)
        } catch (_: Exception) {
        }
        try {
            outputSurface?.release()
        } catch (_: Exception) {
        }
        outputSurface = Surface(surfaceTexture)
    }

    private fun retryPendingStart() {
        if (!pendingStart || running.get()) return
        Thread {
            start()
        }.apply {
            isDaemon = true
            name = "TelloVideoSurfaceRetry"
            start()
        }
    }

    private fun stopInternal() {
        pendingStart = false
        running.set(false)

        try {
            receiverThread?.interrupt()
        } catch (_: Exception) {
        }
        receiverThread = null

        try {
            socket?.close()
        } catch (_: Exception) {
        }
        socket = null

        frameAssembler.reset()
        stopRecording()

        try {
            decoder?.stop()
        } catch (_: Exception) {
        }
        try {
            decoder?.release()
        } catch (_: Exception) {
        }
        decoder = null

        log("[VIDEO] stream receiver durdu")
    }

    private fun convertRawRecordingToMp4Async(rawFile: File) {
        remuxExecutor.execute {
            try {
                val mp4File = File(rawFile.parentFile, rawFile.nameWithoutExtension + ".mp4")
                val ok = remuxAnnexBToMp4(rawFile, mp4File)
                if (ok) {
                    log("[VIDEO] MP4 hazir: ${mp4File.name}")
                } else {
                    log("[VIDEO] MP4 donusumu basarisiz, ham kayit korundu: ${rawFile.name}")
                }
            } catch (e: Exception) {
                log("[VIDEO] MP4 donusum hatasi: ${e.message}")
            }
        }
    }

    private fun remuxAnnexBToMp4(rawFile: File, mp4File: File): Boolean {
        if (!rawFile.exists() || rawFile.length() < 16L) return false

        val data = FileInputStream(rawFile).use { fis ->
            BufferedInputStream(fis).readBytes()
        }
        if (data.isEmpty()) return false

        val units = parseAnnexBNalus(data)
        if (units.isEmpty()) return false

        var sps: ByteArray? = null
        var pps: ByteArray? = null
        for (nal in units) {
            val type = nal[0].toInt() and 0x1F
            if (type == 7 && sps == null) sps = nal
            if (type == 8 && pps == null) pps = nal
            if (sps != null && pps != null) break
        }
        if (sps == null || pps == null) return false

        if (mp4File.exists()) {
            mp4File.delete()
        }

        var muxer: MediaMuxer? = null
        return try {
            val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height)
            format.setByteBuffer("csd-0", ByteBuffer.wrap(withStartCode(sps)))
            format.setByteBuffer("csd-1", ByteBuffer.wrap(withStartCode(pps)))
            format.setInteger(MediaFormat.KEY_FRAME_RATE, 30)

            muxer = MediaMuxer(mp4File.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            val trackIndex = muxer.addTrack(format)
            muxer.start()

            val frameIntervalUs = 33_333L
            var ptsUs = 0L

            val sampleNalList = ArrayList<ByteArray>(8)
            var hasVcl = false
            var sampleHasKey = false

            fun writeSample() {
                if (!hasVcl || sampleNalList.isEmpty()) return
                val buffer = buildAvccSample(sampleNalList)
                val info = MediaCodec.BufferInfo().apply {
                    offset = 0
                    size = buffer.remaining()
                    presentationTimeUs = ptsUs
                    flags = if (sampleHasKey) MediaCodec.BUFFER_FLAG_KEY_FRAME else 0
                }
                muxer.writeSampleData(trackIndex, buffer, info)
                ptsUs += frameIntervalUs
                sampleNalList.clear()
                hasVcl = false
                sampleHasKey = false
            }

            for (nal in units) {
                val type = nal[0].toInt() and 0x1F

                when (type) {
                    7, 8, 9 -> {
                        if (type == 9) writeSample()
                        continue
                    }
                }

                if (type == 1 || type == 5) {
                    val firstSlice = isFirstSliceInPicture(nal)
                    if (hasVcl && firstSlice) {
                        writeSample()
                    }
                    hasVcl = true
                    if (type == 5) sampleHasKey = true
                }

                sampleNalList.add(nal)
            }
            writeSample()

            muxer.stop()
            true
        } catch (_: Exception) {
            try {
                muxer?.stop()
            } catch (_: Exception) {
            }
            false
        } finally {
            try {
                muxer?.release()
            } catch (_: Exception) {
            }
        }
    }

    private fun parseAnnexBNalus(data: ByteArray): List<ByteArray> {
        val units = ArrayList<ByteArray>(1024)
        var start = findStartCode(data, 0)
        while (start >= 0) {
            val scLen = startCodeLengthAt(data, start)
            val nalStart = start + scLen
            val next = findStartCode(data, nalStart)
            val nalEnd = if (next >= 0) next else data.size
            if (nalEnd > nalStart) {
                units.add(data.copyOfRange(nalStart, nalEnd))
            }
            if (next < 0) break
            start = next
        }
        return units
    }

    private fun findStartCode(data: ByteArray, from: Int): Int {
        var i = from.coerceAtLeast(0)
        while (i + 3 < data.size) {
            if (data[i].toInt() == 0 && data[i + 1].toInt() == 0) {
                if (data[i + 2].toInt() == 1) return i
                if (i + 3 < data.size && data[i + 2].toInt() == 0 && data[i + 3].toInt() == 1) return i
            }
            i++
        }
        return -1
    }

    private fun startCodeLengthAt(data: ByteArray, idx: Int): Int {
        return if (idx + 3 < data.size && data[idx].toInt() == 0 && data[idx + 1].toInt() == 0 &&
            data[idx + 2].toInt() == 0 && data[idx + 3].toInt() == 1
        ) {
            4
        } else {
            3
        }
    }

    private fun withStartCode(nal: ByteArray): ByteArray {
        val out = ByteArray(4 + nal.size)
        out[0] = 0
        out[1] = 0
        out[2] = 0
        out[3] = 1
        System.arraycopy(nal, 0, out, 4, nal.size)
        return out
    }

    private fun isFirstSliceInPicture(nal: ByteArray): Boolean {
        if (nal.size < 2) return true
        // first_mb_in_slice for value 0 is Exp-Golomb code '1', i.e. MSB set.
        return (nal[1].toInt() and 0x80) != 0
    }

    private fun buildAvccSample(nals: List<ByteArray>): ByteBuffer {
        var total = 0
        for (nal in nals) {
            total += 4 + nal.size
        }
        val out = ByteBuffer.allocate(total)
        for (nal in nals) {
            out.putInt(nal.size)
            out.put(nal)
        }
        out.flip()
        return out
    }

    private fun log(message: String) {
        listener?.onVideoLog(message)
    }

    private inner class H264AccessUnitAssembler(
        private val telloPacketSize: Int = 1460,
        private val maxFrameBytes: Int = 512 * 1024,
    ) {
        private val buffer = ByteArrayOutputStream(maxFrameBytes.coerceAtMost(64 * 1024))
        private val currentAccessUnit = ArrayList<ByteArray>(8)
        private var latestSps: ByteArray? = null
        private var latestPps: ByteArray? = null
        private var pendingSei: ByteArray? = null
        private var currentHasVcl = false

        fun append(packet: ByteArray, size: Int): List<ByteArray> {
            if (size <= 0) return emptyList()
            if (buffer.size() + size > maxFrameBytes) {
                reset()
            }
            buffer.write(packet, 0, size)
            return drain(flushTail = size < telloPacketSize)
        }

        fun reset() {
            buffer.reset()
            currentAccessUnit.clear()
            latestSps = null
            latestPps = null
            pendingSei = null
            currentHasVcl = false
        }

        private fun drain(flushTail: Boolean): List<ByteArray> {
            val data = buffer.toByteArray()
            if (data.isEmpty()) return emptyList()

            val out = ArrayList<ByteArray>(4)
            var start = findStartCode(data, 0)
            if (start < 0) {
                if (data.size > maxFrameBytes / 2) {
                    buffer.reset()
                }
                return emptyList()
            }

            var carryFrom = start
            while (start >= 0) {
                val scLen = startCodeLengthAt(data, start)
                val nalStart = start + scLen
                if (nalStart >= data.size) {
                    carryFrom = start
                    break
                }

                val next = findStartCode(data, nalStart)
                if (next < 0) {
                    if (flushTail) {
                        processNal(data.copyOfRange(nalStart, data.size), out)
                        carryFrom = data.size
                        flushCurrentAccessUnit(out)
                    } else {
                        carryFrom = start
                    }
                    break
                }

                if (next > nalStart) {
                    processNal(data.copyOfRange(nalStart, next), out)
                }
                carryFrom = next
                start = next
            }

            if (start < 0) {
                carryFrom = data.size
            }

            val remaining = (data.size - carryFrom).coerceAtLeast(0)
            buffer.reset()
            if (remaining > 0 && carryFrom < data.size) {
                buffer.write(data, carryFrom, remaining)
            }

            return out
        }

        private fun processNal(nal: ByteArray, out: MutableList<ByteArray>) {
            if (nal.isEmpty()) return
            val type = nal[0].toInt() and 0x1F

            when (type) {
                7 -> {
                    if (currentHasVcl) {
                        flushCurrentAccessUnit(out)
                    }
                    latestSps = nal.copyOf()
                }

                8 -> {
                    if (currentHasVcl) {
                        flushCurrentAccessUnit(out)
                    }
                    latestPps = nal.copyOf()
                }

                9 -> {
                    flushCurrentAccessUnit(out)
                }

                6 -> {
                    if (currentHasVcl) {
                        currentAccessUnit.add(withStartCode(nal))
                    } else {
                        pendingSei = nal.copyOf()
                    }
                }

                1, 5 -> {
                    val firstSlice = isFirstSliceInPicture(nal)
                    if (currentHasVcl && firstSlice) {
                        flushCurrentAccessUnit(out)
                    }
                    if (!currentHasVcl) {
                        beginAccessUnit()
                    }
                    currentAccessUnit.add(withStartCode(nal))
                    currentHasVcl = true
                }

                else -> {
                    if (currentHasVcl) {
                        currentAccessUnit.add(withStartCode(nal))
                    }
                }
            }
        }

        private fun beginAccessUnit() {
            currentAccessUnit.clear()
            latestSps?.let { currentAccessUnit.add(withStartCode(it)) }
            latestPps?.let { currentAccessUnit.add(withStartCode(it)) }
            pendingSei?.let {
                currentAccessUnit.add(withStartCode(it))
                pendingSei = null
            }
        }

        private fun flushCurrentAccessUnit(out: MutableList<ByteArray>) {
            if (!currentHasVcl || currentAccessUnit.isEmpty()) {
                currentAccessUnit.clear()
                currentHasVcl = false
                return
            }

            var total = 0
            for (part in currentAccessUnit) {
                total += part.size
            }

            val sample = ByteArray(total)
            var offset = 0
            for (part in currentAccessUnit) {
                System.arraycopy(part, 0, sample, offset, part.size)
                offset += part.size
            }

            out.add(sample)
            currentAccessUnit.clear()
            currentHasVcl = false
        }
    }
}
