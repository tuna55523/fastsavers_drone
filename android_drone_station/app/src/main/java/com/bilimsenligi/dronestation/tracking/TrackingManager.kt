package com.bilimsenligi.dronestation.tracking

class TrackingManager {

    data class TrackingStatus(
        val enabled: Boolean,
        val targetIndex: Int,
    )

    @Volatile
    private var enabled = false

    @Volatile
    private var targetIndex = 0

    fun start() {
        enabled = true
    }

    fun stop() {
        enabled = false
    }

    fun toggle(): Boolean {
        enabled = !enabled
        return enabled
    }

    fun nextTarget(): Int {
        targetIndex += 1
        return targetIndex
    }

    fun status(): TrackingStatus {
        return TrackingStatus(
            enabled = enabled,
            targetIndex = targetIndex,
        )
    }
}
