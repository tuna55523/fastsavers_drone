package com.bilimsenligi.dronestation.tracking

import com.bilimsenligi.dronestation.control.GamepadMapper
import kotlin.math.abs
import kotlin.math.sign

class TrackingAssistMixer {

    data class Gains(
        val yawKp: Float = 18f,
        val udKp: Float = 10f,
        val fbKp: Float = 12f,
        val desiredSize: Float = 0.22f,
<<<<<<< HEAD
        val sizeDeadzone: Float = 0.08f,
        val maxFbAssistAbs: Int = 18,
        val minConfidence: Float = 0.50f,
        val deadzoneX: Float = 0.12f,
        val deadzoneY: Float = 0.16f,
=======
        val sizeDeadzone: Float = 0.03f,
        val maxFbAssistAbs: Int = 42,
        val minConfidence: Float = 0.16f,
        val deadzoneX: Float = 0.04f,
        val deadzoneY: Float = 0.05f,
>>>>>>> 3364bd317ce1848cb9738406d856bd60f04d06c2
        /**
         * Tracking sample akisi kesildiginde komutlarin ne kadar hizli sifira sonecegini belirler.
         * Ornek: staleTimeoutMs=900 ise, 300ms'de yaklasik sifira iner.
         */
        val fadeOutMs: Long = 300L,
    )

    private var gains = Gains()

    fun setGains(gains: Gains) {
        this.gains = gains
    }

    fun mix(
        manual: GamepadMapper.RcCommand,
        trackingStatus: TrackingManager.TrackingStatus,
        sample: TrackingTelemetryReceiver.TrackingSample?,
        nowMs: Long,
        staleTimeoutMs: Long,
    ): GamepadMapper.RcCommand {
        if (!trackingStatus.enabled || sample == null || sample.confidence < gains.minConfidence) {
            return manual
        }

        val ageMs = nowMs - sample.timestampMs
        if (ageMs > staleTimeoutMs) {
            return manual
        }

        if (sample.targetId >= 0 && trackingStatus.targetIndex > 0) {
            val matchesZeroBased = sample.targetId == trackingStatus.targetIndex
            val matchesOneBased = sample.targetId == (trackingStatus.targetIndex + 1)
            if (!matchesZeroBased && !matchesOneBased) {
                return manual
            }
        }

        val fadeMs = gains.fadeOutMs.coerceIn(0L, staleTimeoutMs)
        val alpha = when {
            fadeMs <= 0L -> 1f
            ageMs <= fadeMs -> 1f
            else -> {
                val denom = (staleTimeoutMs - fadeMs).coerceAtLeast(1L).toFloat()
                (1f - ((ageMs - fadeMs).toFloat() / denom)).coerceIn(0f, 1f)
            }
        }

        val yawAssist = if (abs(sample.tx) < gains.deadzoneX) 0 else (sample.tx * gains.yawKp * alpha).toInt()
        val udAssist = if (abs(sample.ty) < gains.deadzoneY) 0 else (-sample.ty * gains.udKp * alpha).toInt()

        val sizeError = gains.desiredSize - sample.size
        val sizeErrorDz = if (abs(sizeError) < gains.sizeDeadzone) 0f else {
            val trimmed = abs(sizeError) - gains.sizeDeadzone
            sign(sizeError) * trimmed
        }
        val fbAssist = (sizeErrorDz * gains.fbKp * alpha)
            .toInt()
            .coerceIn(-gains.maxFbAssistAbs, gains.maxFbAssistAbs)

        // Manual forward/back control remains primary; tracking only assists.
        val mixedFb = (manual.forwardBack + fbAssist).coerceIn(-100, 100)

        val mixedYaw = (manual.yaw + yawAssist).coerceIn(-100, 100)
        val mixedUd = (manual.upDown + udAssist).coerceIn(-100, 100)

        return GamepadMapper.RcCommand(
            leftRight = manual.leftRight,
            forwardBack = mixedFb,
            upDown = mixedUd,
            yaw = mixedYaw,
        )
    }
}
