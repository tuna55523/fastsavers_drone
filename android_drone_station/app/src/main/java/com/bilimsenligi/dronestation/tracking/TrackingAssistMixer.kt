package com.bilimsenligi.dronestation.tracking

import com.bilimsenligi.dronestation.control.GamepadMapper
import kotlin.math.abs

class TrackingAssistMixer {

    data class Gains(
        val yawKp: Float = 38f,
        val udKp: Float = 28f,
        val fbKp: Float = 26f,
        val desiredSize: Float = 0.22f,
        val minConfidence: Float = 0.30f,
        val deadzoneX: Float = 0.04f,
        val deadzoneY: Float = 0.05f,
    )

    private var gains = Gains()

    fun setGains(gains: Gains) {
        this.gains = gains
    }

    fun mix(
        manual: GamepadMapper.RcCommand,
        trackingEnabled: Boolean,
        sample: TrackingTelemetryReceiver.TrackingSample?,
    ): GamepadMapper.RcCommand {
        if (!trackingEnabled || sample == null || sample.confidence < gains.minConfidence) {
            return manual
        }

        val yawAssist = if (abs(sample.tx) < gains.deadzoneX) 0 else (sample.tx * gains.yawKp).toInt()
        val udAssist = if (abs(sample.ty) < gains.deadzoneY) 0 else (-sample.ty * gains.udKp).toInt()

        val sizeError = gains.desiredSize - sample.size
        val fbAssist = (sizeError * gains.fbKp).toInt()

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
