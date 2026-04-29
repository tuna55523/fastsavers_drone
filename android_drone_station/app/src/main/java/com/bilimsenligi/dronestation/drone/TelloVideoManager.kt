package com.bilimsenligi.dronestation.drone

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaFormat
import android.media.MediaMuxer
import android.os.SystemClock
import android.view.Surface
import android.view.TextureView
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.net.DatagramPacket
import java.net.DatagramSocket
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
    private val remuxExecutor = Executors.newSingleThreadExecutor()

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
            outputSurface = Surface(view.surfaceTexture)
        }
        view.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(surfaceTexture: android.graphics.SurfaceTexture, w: Int, h: Int) {
                outputSurface = Surface(surfaceTexture)
                log("[VIDEO] surface hazir")
            }

            override fun onSurfaceTextureSizeChanged(surfaceTexture: android.graphics.SurfaceTexture, w: Int, h: Int) {
            }

            override fun onSurfaceTextureDestroyed(surfaceTexture: android.graphics.SurfaceTexture): Boolean {
                outputSurface?.release()
                outputSurface = null
                return true
            }

            override fun onSurfaceTextureUpdated(surfaceTexture: android.graphics.SurfaceTexture) {
            }
        }
    }

    fun start(): Boolean {
        if (running.get()) return true

        val surface = outputSurface
        if (surface == null) {
            log("[VIDEO] surface hazir degil")
            return false
        }

        return try {
            val codec = MediaCodec.createDecoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
            val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height)
            format.setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 1024 * 1024)
            codec.configure(format, surface, null, 0)
            codec.start()
            decoder = codec

            val udpSocket = DatagramSocket(listenPort).apply {
                soTimeout = 1000
                reuseAddress = true
                receiveBufferSize = 2 * 1024 * 1024
            }
            socket = udpSocket

            running.set(true)
            startReceiverLoop()
            log("[VIDEO] stream receiver basladi")
            true
        } catch (e: Exception) {
            log("[VIDEO] baslatma hatasi: ${e.message}")
            stop()
            false
        }
    }

    fun stop() {
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

                    synchronized(recordLock) {
                        if (recording.get()) {
                            try {
                                recordingStream?.write(packet.data, 0, size)
                            } catch (_: Exception) {
                            }
                        }
                    }

                    val frameEnd = size != 1460
                    feedDecoderChunk(codec, packet.data, size, frameEnd)
                    drainDecoder(codec)
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

    private fun feedDecoderChunk(codec: MediaCodec, data: ByteArray, size: Int, frameEnd: Boolean) {
        if (size <= 0) return

        try {
            val inputIndex = codec.dequeueInputBuffer(0)
            if (inputIndex >= 0) {
                val inputBuffer = codec.getInputBuffer(inputIndex)
                if (inputBuffer != null) {
                    inputBuffer.clear()
                    inputBuffer.put(data, 0, size)
                    val ptsUs = SystemClock.elapsedRealtimeNanos() / 1000L
                    val flags = if (frameEnd) 0 else MediaCodec.BUFFER_FLAG_PARTIAL_FRAME
                    codec.queueInputBuffer(inputIndex, 0, size, ptsUs, flags)
                }
            }
        } catch (_: Exception) {
        }
    }

    private fun drainDecoder(codec: MediaCodec) {
        try {
            val info = MediaCodec.BufferInfo()
            var outputIndex = codec.dequeueOutputBuffer(info, 0)
            while (outputIndex >= 0) {
                codec.releaseOutputBuffer(outputIndex, true)
                outputIndex = codec.dequeueOutputBuffer(info, 0)
            }
        } catch (_: Exception) {
        }
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
                muxer?.writeSampleData(trackIndex, buffer, info)
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
}
