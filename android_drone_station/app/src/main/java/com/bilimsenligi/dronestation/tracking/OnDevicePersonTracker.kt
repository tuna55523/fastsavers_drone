package com.bilimsenligi.dronestation.tracking

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.abs
import kotlin.math.exp

class OnDevicePersonTracker(
    private val context: Context,
    private val modelAssetName: String = "person_tracking_best_v8n_768.tflite",
) {

    data class DetectionSample(
        val tx: Float,
        val ty: Float,
        val size: Float,
        val confidence: Float,
    )

    @Volatile
    private var detector: ObjectDetector? = null

    @Volatile
    private var interpreter: Interpreter? = null

    @Volatile
    private var initTried = false

    @Volatile
    private var lastInitAttemptMs = 0L

    @Volatile
    private var inputWidth = 640

    @Volatile
    private var inputHeight = 640

    @Volatile
    private var inputType: DataType = DataType.FLOAT32

    @Volatile
    private var inputScale: Float = 1f

    @Volatile
    private var inputZeroPoint: Int = 0

    @Volatile
    private var outputType: DataType = DataType.FLOAT32

    @Volatile
    private var outputScale: Float = 1f

    @Volatile
    private var outputZeroPoint: Int = 0

    @Volatile
    private var outputShape: IntArray = intArrayOf()

    @Volatile
    private var lastInitError: String? = null

    fun isReady(): Boolean {
        ensureInit()
        return detector != null || interpreter != null
    }

    fun latestInitError(): String? = lastInitError

    fun detect(bitmap: Bitmap): DetectionSample? {
        ensureInit()
        val d = detector
        if (d != null) {
            return detectWithTaskVision(d, bitmap)
        }
        val it = interpreter ?: return null
        return detectWithInterpreter(it, bitmap)
    }

    private fun detectWithTaskVision(d: ObjectDetector, bitmap: Bitmap): DetectionSample? {
        val tensorImage = TensorImage.fromBitmap(bitmap)
        val results = d.detect(tensorImage)
        if (results.isEmpty()) return null

        val w = bitmap.width.toFloat().coerceAtLeast(1f)
        val h = bitmap.height.toFloat().coerceAtLeast(1f)
        var bestSample: DetectionSample? = null
        var bestComposite = -1f

        for (det in results) {
            val bbox = det.boundingBox
            val cat = det.categories.maxByOrNull { it.score }
            val score = (cat?.score ?: 0f).coerceIn(0f, 1f)
            val label = (cat?.label ?: "").lowercase()
            val looksPerson = label.isBlank() || label.contains("person")
            if (!looksPerson) continue

            val cxNorm = (bbox.centerX() / w).coerceIn(0f, 1f)
            val cyNorm = (bbox.centerY() / h).coerceIn(0f, 1f)
            val bwNorm = (bbox.width() / w).coerceIn(0f, 1f)
            val bhNorm = (bbox.height() / h).coerceIn(0f, 1f)
            val areaNorm = (bwNorm * bhNorm).coerceIn(0f, 1f)

            val tx = (cxNorm * 2f - 1f).coerceIn(-1f, 1f)
            val ty = (cyNorm * 2f - 1f).coerceIn(-1f, 1f)
            val composite = compositeScore(score, tx, ty, areaNorm)

            if (composite > bestComposite) {
                bestComposite = composite
                bestSample = DetectionSample(
                    tx = tx,
                    ty = ty,
                    size = areaNorm,
                    confidence = score,
                )
            }
        }
        return bestSample
    }

    private fun detectWithInterpreter(it: Interpreter, bitmap: Bitmap): DetectionSample? {
        val modelW = inputWidth.coerceAtLeast(1)
        val modelH = inputHeight.coerceAtLeast(1)
        val scaled = Bitmap.createScaledBitmap(bitmap, modelW, modelH, true)

        return try {
            val input = buildInputBuffer(scaled, inputType)
            val total = outputShape.fold(1) { acc, dim -> acc * dim.coerceAtLeast(1) }
            if (total <= 0) return null

            val bytesPerValue = when (outputType) {
                DataType.INT8, DataType.UINT8 -> 1
                else -> 4
            }
            val outBuffer = ByteBuffer.allocateDirect(total * bytesPerValue).order(ByteOrder.nativeOrder())
            it.run(input, outBuffer)
            outBuffer.rewind()

            val raw = readOutputAsFloatArray(outBuffer, total)
            decodeYoloOutput(raw, outputShape)
        } catch (_: Exception) {
            null
        } finally {
            if (scaled !== bitmap) {
                scaled.recycle()
            }
        }
    }

    private fun decodeYoloOutput(raw: FloatArray, shape: IntArray): DetectionSample? {
        if (shape.size != 3 || shape[0] != 1) return null

        val d1 = shape[1]
        val d2 = shape[2]

        var bestSample: DetectionSample? = null
        var bestComposite = -1f

        fun consider(cxNorm: Float, cyNorm: Float, bwNorm: Float, bhNorm: Float, confRaw: Float) {
            val confidence = toProb(confRaw)
            if (confidence < 0.08f) return
            val tx = (cxNorm * 2f - 1f).coerceIn(-1f, 1f)
            val ty = (cyNorm * 2f - 1f).coerceIn(-1f, 1f)
            val areaNorm = (bwNorm.coerceIn(0f, 1f) * bhNorm.coerceIn(0f, 1f)).coerceIn(0f, 1f)
            if (areaNorm < 0.0006f) return

            val composite = compositeScore(confidence, tx, ty, areaNorm)
            if (composite > bestComposite) {
                bestComposite = composite
                bestSample = DetectionSample(
                    tx = tx,
                    ty = ty,
                    size = areaNorm,
                    confidence = confidence,
                )
            }
        }

        // Layout A: [1, C, N] (Ultralytics export yaygin)
        if (d1 in 5..8 && d2 > d1) {
            for (i in 0 until d2) {
                val x = raw[i]
                val y = raw[d2 + i]
                val w = raw[2 * d2 + i]
                val h = raw[3 * d2 + i]

                val conf = if (d1 == 5) {
                    raw[4 * d2 + i]
                } else {
                    val obj = toProb(raw[4 * d2 + i])
                    var bestCls = 0f
                    for (c in 5 until d1) {
                        bestCls = maxOf(bestCls, toProb(raw[c * d2 + i]))
                    }
                    obj * bestCls
                }
                val (cx, cy, bw, bh) = normalizeBox(x, y, w, h)
                consider(cx, cy, bw, bh, conf)
            }
            return bestSample
        }

        // Layout B: [1, N, C]
        if (d2 in 5..8 && d1 > d2) {
            for (i in 0 until d1) {
                val base = i * d2
                val x = raw[base]
                val y = raw[base + 1]
                val w = raw[base + 2]
                val h = raw[base + 3]
                val conf = if (d2 == 5) {
                    raw[base + 4]
                } else {
                    val obj = toProb(raw[base + 4])
                    var bestCls = 0f
                    for (c in 5 until d2) {
                        bestCls = maxOf(bestCls, toProb(raw[base + c]))
                    }
                    obj * bestCls
                }
                val (cx, cy, bw, bh) = normalizeBox(x, y, w, h)
                consider(cx, cy, bw, bh, conf)
            }
            return bestSample
        }

        return null
    }

    private fun normalizeBox(x: Float, y: Float, w: Float, h: Float): FloatArray {
        val isNormalized = x <= 1.5f && y <= 1.5f && w <= 1.5f && h <= 1.5f
        return if (isNormalized) {
            floatArrayOf(
                x.coerceIn(0f, 1f),
                y.coerceIn(0f, 1f),
                w.coerceIn(0f, 1f),
                h.coerceIn(0f, 1f),
            )
        } else {
            floatArrayOf(
                (x / inputWidth.toFloat()).coerceIn(0f, 1f),
                (y / inputHeight.toFloat()).coerceIn(0f, 1f),
                (w / inputWidth.toFloat()).coerceIn(0f, 1f),
                (h / inputHeight.toFloat()).coerceIn(0f, 1f),
            )
        }
    }

    private fun buildInputBuffer(bitmap: Bitmap, dtype: DataType): ByteBuffer {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        return when (dtype) {
            DataType.UINT8 -> {
                val buffer = ByteBuffer.allocateDirect(w * h * 3).order(ByteOrder.nativeOrder())
                for (p in pixels) {
                    val r = ((p shr 16) and 0xFF) / 255f
                    val g = ((p shr 8) and 0xFF) / 255f
                    val b = (p and 0xFF) / 255f
                    buffer.put(quantizeToU8(r).toByte())
                    buffer.put(quantizeToU8(g).toByte())
                    buffer.put(quantizeToU8(b).toByte())
                }
                buffer.rewind()
                buffer
            }

            DataType.INT8 -> {
                val buffer = ByteBuffer.allocateDirect(w * h * 3).order(ByteOrder.nativeOrder())
                for (p in pixels) {
                    val r = ((p shr 16) and 0xFF) / 255f
                    val g = ((p shr 8) and 0xFF) / 255f
                    val b = (p and 0xFF) / 255f
                    buffer.put(quantizeToI8(r).toByte())
                    buffer.put(quantizeToI8(g).toByte())
                    buffer.put(quantizeToI8(b).toByte())
                }
                buffer.rewind()
                buffer
            }

            else -> {
                val buffer = ByteBuffer.allocateDirect(w * h * 3 * 4).order(ByteOrder.nativeOrder())
                for (p in pixels) {
                    buffer.putFloat(((p shr 16) and 0xFF) / 255f)
                    buffer.putFloat(((p shr 8) and 0xFF) / 255f)
                    buffer.putFloat((p and 0xFF) / 255f)
                }
                buffer.rewind()
                buffer
            }
        }
    }

    private fun compositeScore(confidence: Float, tx: Float, ty: Float, areaNorm: Float): Float {
        // Kalabalikta daha yakin (daha buyuk bbox) + guven yuksek hedefi sec.
        val centerPenalty = abs(tx) * 0.10f + abs(ty) * 0.08f
        return (confidence * 0.60f) + (areaNorm * 0.75f) - centerPenalty
    }

    private fun toProb(v: Float): Float {
        if (v.isNaN()) return 0f
        return if (v in 0f..1f) v else (1f / (1f + exp(-v))).coerceIn(0f, 1f)
    }

    private fun ensureInit() {
        if (detector != null || interpreter != null) return
        val now = System.currentTimeMillis()
        if (initTried && (now - lastInitAttemptMs) < 5_000L) return
        initTried = true
        lastInitAttemptMs = now
        lastInitError = null

        try {
            val base = BaseOptions.builder().setNumThreads(3).build()
            val options = ObjectDetector.ObjectDetectorOptions.builder()
                .setBaseOptions(base)
                .setScoreThreshold(0.18f)
                .setMaxResults(8)
                .build()
            detector = ObjectDetector.createFromFileAndOptions(context, modelAssetName, options)
            return
        } catch (e: Exception) {
            detector = null
            lastInitError = "TaskVision: ${e.message ?: "unknown"}"
        }

        try {
            val modelBuffer = loadModelBuffer()
            val options = Interpreter.Options().setNumThreads(4)
            val it = Interpreter(modelBuffer, options)

            val inTensor = it.getInputTensor(0)
            val inShape = inTensor.shape()
            if (inShape.size == 4) {
                inputHeight = inShape[1].coerceAtLeast(1)
                inputWidth = inShape[2].coerceAtLeast(1)
            }
            inputType = inTensor.dataType()
            val inQ = inTensor.quantizationParams()
            inputScale = if (inQ.scale > 0f) inQ.scale else 1f / 255f
            inputZeroPoint = inQ.zeroPoint

            val outTensor = it.getOutputTensor(0)
            outputShape = outTensor.shape()
            outputType = outTensor.dataType()
            val outQ = outTensor.quantizationParams()
            outputScale = if (outQ.scale > 0f) outQ.scale else 1f
            outputZeroPoint = outQ.zeroPoint
            interpreter = it
        } catch (e: Exception) {
            interpreter = null
            val prev = lastInitError ?: ""
            val suffix = "Interpreter: ${e.message ?: "unknown"}"
            lastInitError = if (prev.isBlank()) suffix else "$prev | $suffix"
        }
    }

    private fun readOutputAsFloatArray(outBuffer: ByteBuffer, total: Int): FloatArray {
        val out = FloatArray(total)
        when (outputType) {
            DataType.INT8 -> {
                for (i in 0 until total) {
                    val q = outBuffer.get().toInt()
                    out[i] = (q - outputZeroPoint) * outputScale
                }
            }

            DataType.UINT8 -> {
                for (i in 0 until total) {
                    val q = outBuffer.get().toInt() and 0xFF
                    out[i] = (q - outputZeroPoint) * outputScale
                }
            }

            else -> {
                for (i in 0 until total) {
                    out[i] = outBuffer.float
                }
            }
        }
        return out
    }

    private fun quantizeToU8(v01: Float): Int {
        val scale = inputScale.takeIf { it > 0f } ?: (1f / 255f)
        val q = (v01.coerceIn(0f, 1f) / scale + inputZeroPoint).toInt()
        return q.coerceIn(0, 255)
    }

    private fun quantizeToI8(v01: Float): Int {
        val scale = inputScale.takeIf { it > 0f } ?: (1f / 255f)
        val q = (v01.coerceIn(0f, 1f) / scale + inputZeroPoint).toInt()
        return q.coerceIn(-128, 127)
    }

    private fun loadModelBuffer(): ByteBuffer {
        return try {
            val fd = context.assets.openFd(modelAssetName)
            FileInputStream(fd.fileDescriptor).channel.use { ch ->
                ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        } catch (_: Exception) {
            val bytes = context.assets.open(modelAssetName).use { it.readBytes() }
            ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
                put(bytes)
                rewind()
            }
        }
    }
}
