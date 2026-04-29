package com.bilimsenligi.dronestation.tracking

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import kotlin.math.abs

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
    private var initTried = false

    @Volatile
    private var lastInitAttemptMs = 0L

    fun isReady(): Boolean {
        ensureInit()
        return detector != null
    }

    fun detect(bitmap: Bitmap): DetectionSample? {
        ensureInit()
        val d = detector ?: return null

        val tensorImage = TensorImage.fromBitmap(bitmap)
        val results = d.detect(tensorImage)
        if (results.isEmpty()) return null

        val w = bitmap.width.toFloat().coerceAtLeast(1f)
        val h = bitmap.height.toFloat().coerceAtLeast(1f)

        var bestSample: DetectionSample? = null
        var bestScore = -1f

        for (det in results) {
            val bbox = det.boundingBox
            val cat = det.categories.maxByOrNull { it.score }
            val score = cat?.score ?: 0f
            val label = (cat?.label ?: "").lowercase()

            // Person-only modelde label bazen bos olabilir; bu yuzden score odakli gidiyoruz.
            val looksPerson = label.isBlank() || label.contains("person")
            if (!looksPerson) continue

            val cx = bbox.centerX().coerceIn(0f, w)
            val cy = bbox.centerY().coerceIn(0f, h)
            val tx = ((cx / w) * 2f - 1f).coerceIn(-1f, 1f)
            val ty = ((cy / h) * 2f - 1f).coerceIn(-1f, 1f)

            val areaNorm = ((bbox.width() * bbox.height()) / (w * h)).coerceIn(0f, 1f)

            // Center'a yakin ve guvenli bbox oncelikli olsun.
            val centerPenalty = abs(tx) * 0.25f + abs(ty) * 0.20f
            val composite = score - centerPenalty

            if (composite > bestScore) {
                bestScore = composite
                bestSample = DetectionSample(
                    tx = tx,
                    ty = ty,
                    size = areaNorm,
                    confidence = score.coerceIn(0f, 1f),
                )
            }
        }

        return bestSample
    }

    private fun ensureInit() {
        if (detector != null) return
        val now = System.currentTimeMillis()
        if (initTried && (now - lastInitAttemptMs) < 5_000L) return
        initTried = true
        lastInitAttemptMs = now

        try {
            val base = BaseOptions.builder().setNumThreads(3).build()
            val options = ObjectDetector.ObjectDetectorOptions.builder()
                .setBaseOptions(base)
                .setScoreThreshold(0.30f)
                .setMaxResults(5)
                .build()

            detector = ObjectDetector.createFromFileAndOptions(context, modelAssetName, options)
        } catch (_: Exception) {
            detector = null
        }
    }
}
