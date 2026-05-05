package com.bilimsenligi.dronestation.control

import android.view.InputDevice
import android.view.KeyEvent
import android.view.MotionEvent
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.roundToInt

class GamepadMapper {

    enum class Action {
        TAKEOFF_LAND_TOGGLE,
        FLIP_FORWARD,
        FLIP_BACK,
        FLIP_LEFT,
        FLIP_RIGHT,
        PHOTO_CAPTURE,
        VIDEO_TOGGLE,
        VIDEO_PANEL_EXIT,
        TRACKING_START,
        TRACKING_STOP,
        TRACKING_TARGET_NEXT,
    }

    data class RcCommand(
        val leftRight: Int,
        val forwardBack: Int,
        val upDown: Int,
        val yaw: Int,
    )

    private var leftStickY: Float = 0f
    private var rightStickX: Float = 0f
    private var leftTriggerAxis: Float = 0f
    private var rightTriggerAxis: Float = 0f
    private var hasLeftTriggerAxis: Boolean = false
    private var hasRightTriggerAxis: Boolean = false
    private var leftTriggerBaseline: Float? = null
    private var rightTriggerBaseline: Float? = null

    private var l2Pressed: Boolean = false
    private var r2Pressed: Boolean = false
    // Once we've seen an analog trigger axis fire, we trust *only* the axis until it goes
    // stale. PS4 controllers send both AXIS_RTRIGGER motion events and KEYCODE_BUTTON_R2
    // key events; if the keyUp is delayed (or arrives on a different deviceId), the button
    // latch would otherwise override the (correctly zero) axis and drive the drone full-
    // forward. The "axis active" flag closes that fallback path while analog data is live.
    private var leftTriggerAxisActive: Boolean = false
    private var rightTriggerAxisActive: Boolean = false
    // Farkli Android cihazlari kolu birden fazla deviceId olarak raporlayabiliyor.
    // Tek bir id'ye kilitlenmek keyUp/motion event kaybina ve "tek yone gitme"ye yol aciyordu.
    // Bu nedenle sadece tip (gamepad/joystick) kontrolu yapip tum uyumlu deviceId'leri kabul ediyoruz.
    private var activeMotionDeviceId: Int? = null
    private var activeKeyDeviceId: Int? = null
    private var lastMotionEventMs: Long = 0L
    private var lastKeyEventMs: Long = 0L

    private val axisDeadzone = 0.16f
    private val triggerDeadzone = 0.06f
    // Kisalan timeout: kayip keyUp/motion event oldugunda tek yone gitme daha hizli sifirlanir.
    private val analogInputTimeoutMs = 240L
    private val buttonInputTimeoutMs = 650L
    // Son-gun guvenli kontrol profili: anlik "full" tepkileri azalt.
    private val maxForwardBack = 42
    private val maxUpDown = 30
    private val maxYawByRightStick = 34

    fun onGenericMotionEvent(event: MotionEvent): Boolean {
        if (event.action != MotionEvent.ACTION_MOVE) return false
        if (!isGamepadInputDevice(event.device, event.source)) return false
        if (!acceptMotionDevice(event.deviceId)) return false
        lastMotionEventMs = System.currentTimeMillis()

        // Yeni istek: sol joystick yukari/asagi (ud), sag joystick saga/sola (yaw).
        leftStickY = readCenteredAxis(event, MotionEvent.AXIS_Y)
        rightStickX = readCenteredAxis(event, MotionEvent.AXIS_RX)

        rightTriggerAxis = readTriggerAxisFromBestSource(
            event,
            MotionEvent.AXIS_RTRIGGER,
            MotionEvent.AXIS_GAS,
            isRight = true,
        )
        leftTriggerAxis = readTriggerAxisFromBestSource(
            event,
            MotionEvent.AXIS_LTRIGGER,
            MotionEvent.AXIS_BRAKE,
            isRight = false,
        )
        hasRightTriggerAxis = hasAxis(event, MotionEvent.AXIS_RTRIGGER) || hasAxis(event, MotionEvent.AXIS_GAS)
        hasLeftTriggerAxis = hasAxis(event, MotionEvent.AXIS_LTRIGGER) || hasAxis(event, MotionEvent.AXIS_BRAKE)
        // Latch "this controller delivers analog trigger data" the first time we see a
        // non-zero value. From that point on, button latches are ignored for forward/back.
        if (rightTriggerAxis > 0f) rightTriggerAxisActive = true
        if (leftTriggerAxis > 0f) leftTriggerAxisActive = true
        return true
    }

    fun onKeyDown(event: KeyEvent): List<Action> {
        if (!isGamepadInputDevice(event.device, event.source)) return emptyList()
        if (!acceptKeyDevice(event.deviceId)) return emptyList()
        lastKeyEventMs = System.currentTimeMillis()

        val actions = mutableListOf<Action>()
        val firstPress = event.repeatCount == 0

        when (event.keyCode) {
            // L1/R1 bilincli olarak bos birakildi.
            KeyEvent.KEYCODE_BUTTON_L1 -> {}
            KeyEvent.KEYCODE_BUTTON_R1 -> {}
            KeyEvent.KEYCODE_BUTTON_L2 -> l2Pressed = true
            KeyEvent.KEYCODE_BUTTON_R2 -> r2Pressed = true

            KeyEvent.KEYCODE_DPAD_UP -> if (firstPress) actions.add(Action.FLIP_FORWARD)
            KeyEvent.KEYCODE_DPAD_DOWN -> if (firstPress) actions.add(Action.FLIP_BACK)
            KeyEvent.KEYCODE_DPAD_LEFT -> if (firstPress) actions.add(Action.FLIP_LEFT)
            KeyEvent.KEYCODE_DPAD_RIGHT -> if (firstPress) actions.add(Action.FLIP_RIGHT)

            KeyEvent.KEYCODE_BUTTON_THUMBL -> if (firstPress) actions.add(Action.PHOTO_CAPTURE)
            KeyEvent.KEYCODE_BUTTON_THUMBR -> if (firstPress) actions.add(Action.VIDEO_TOGGLE)

            // PS4 Cross button maps to BUTTON_A on Android.
            KeyEvent.KEYCODE_BUTTON_A -> if (firstPress) actions.add(Action.TAKEOFF_LAND_TOGGLE)

            // PS4 Circle button maps to BUTTON_B on Android (bazı cihazlarda BUTTON_C gelebilir).
            KeyEvent.KEYCODE_BUTTON_B,
            KeyEvent.KEYCODE_BUTTON_C -> if (firstPress) actions.add(Action.TRACKING_START)

            // PS4 Square button maps to BUTTON_X on Android.
            KeyEvent.KEYCODE_BUTTON_X -> if (firstPress) actions.add(Action.TRACKING_STOP)

            // PS4 Triangle button maps to BUTTON_Y on Android.
            KeyEvent.KEYCODE_BUTTON_Y -> if (firstPress) actions.add(Action.TRACKING_TARGET_NEXT)

            // PS4 Options button usually maps to START on Android.
            KeyEvent.KEYCODE_BUTTON_START -> if (firstPress) actions.add(Action.VIDEO_PANEL_EXIT)
            KeyEvent.KEYCODE_MENU -> if (firstPress) actions.add(Action.VIDEO_PANEL_EXIT)
        }

        return actions
    }

    fun onKeyUp(event: KeyEvent): Boolean {
        if (!isGamepadInputDevice(event.device, event.source)) return false
        if (!acceptKeyDevice(event.deviceId)) return false
        lastKeyEventMs = System.currentTimeMillis()

        when (event.keyCode) {
            KeyEvent.KEYCODE_BUTTON_L1 -> return true
            KeyEvent.KEYCODE_BUTTON_R1 -> return true
            KeyEvent.KEYCODE_BUTTON_L2 -> l2Pressed = false
            KeyEvent.KEYCODE_BUTTON_R2 -> r2Pressed = false
            else -> return false
        }
        return true
    }

    fun currentRcCommand(): RcCommand {
        expireStaleInputs()

        // While analog trigger data is live, trust *only* the axis. Falling through to the
        // button latch when axis goes to 0 would re-trigger full-forward on every release
        // because PS4 keyUp events can lag the axis by hundreds of ms (or be lost entirely
        // when key/motion events come from different logical deviceIds).
        val forward = when {
            rightTriggerAxisActive -> shapeTrigger(rightTriggerAxis)
            // Sadece analog eksen yoksa dijital fallback kullan.
            !hasRightTriggerAxis && r2Pressed -> 0.42f
            else -> 0f
        }
        val backward = when {
            leftTriggerAxisActive -> shapeTrigger(leftTriggerAxis)
            !hasLeftTriggerAxis && l2Pressed -> 0.42f
            else -> 0f
        }

        val fbRaw = forward - backward
        val fb = scaleSigned(fbRaw, maxForwardBack)

        // Requirement: left analog helps steering while moving forward/back.
        val lr = 0

        val ud = scaleSigned(shapeSigned(-applyDeadzone(leftStickY), exponent = 1.12f), maxUpDown)

        val yaw = scaleSigned(shapeSigned(applyDeadzone(rightStickX), exponent = 1.08f), maxYawByRightStick)

        return RcCommand(
            leftRight = lr,
            forwardBack = fb,
            upDown = ud,
            yaw = yaw,
        )
    }

    fun hasManualInput(): Boolean {
        return kotlin.math.abs(leftStickY) >= axisDeadzone ||
            kotlin.math.abs(rightStickX) >= axisDeadzone ||
            leftTriggerAxis >= 0.05f ||
            rightTriggerAxis >= 0.05f ||
            (!hasLeftTriggerAxis && l2Pressed) ||
            (!hasRightTriggerAxis && r2Pressed)
    }

    fun reset() {
        leftStickY = 0f
        rightStickX = 0f
        leftTriggerAxis = 0f
        rightTriggerAxis = 0f
        hasLeftTriggerAxis = false
        hasRightTriggerAxis = false
        leftTriggerBaseline = null
        rightTriggerBaseline = null
        leftTriggerAxisActive = false
        rightTriggerAxisActive = false
        l2Pressed = false
        r2Pressed = false
        lastMotionEventMs = 0L
        lastKeyEventMs = 0L
        activeMotionDeviceId = null
        activeKeyDeviceId = null
    }

    fun onInputDeviceRemoved(deviceId: Int) {
        if (activeMotionDeviceId == deviceId || activeKeyDeviceId == deviceId) {
            reset()
        }
    }

    private fun scaleSigned(value: Float, maxAbs: Int): Int {
        val clamped = value.coerceIn(-1f, 1f)
        return (clamped * maxAbs).roundToInt().coerceIn(-maxAbs, maxAbs)
    }

    private fun centeredAxis(event: MotionEvent, axis: Int): Float {
        val device = event.device ?: return 0f
        val range = device.getMotionRange(axis, event.source) ?: return 0f
        val value = event.getAxisValue(axis)
        val flat = maxOf(range.flat, axisDeadzone)
        return if (abs(value) > flat) value else 0f
    }

    private fun readCenteredAxis(event: MotionEvent, vararg axes: Int): Float {
        for (axis in axes) {
            val value = centeredAxis(event, axis)
            if (abs(value) > 1e-4f) return value
        }
        return 0f
    }

    private fun readTriggerAxisFromBestSource(
        event: MotionEvent,
        primaryAxis: Int,
        fallbackAxis: Int,
        isRight: Boolean,
    ): Float {
        val primary = readTriggerAxisNormalized(event, primaryAxis, isRight)
        if (primary != null) return primary
        val fallback = readTriggerAxisNormalized(event, fallbackAxis, isRight)
        return fallback ?: 0f
    }

    private fun readTriggerAxisNormalized(event: MotionEvent, axis: Int, isRight: Boolean): Float? {
        val device = event.device ?: return null
        val range = device.getMotionRange(axis, event.source) ?: return null
        val raw = event.getAxisValue(axis)
        val normalized = if (range.min < 0f && range.max > 0f) {
            ((raw - range.min) / (range.max - range.min)).coerceIn(0f, 1f)
        } else {
            raw.coerceIn(0f, 1f)
        }
        val active = baselineToActivation(normalized, isRight)
        return if (active > 0.02f) active else 0f
    }

    private fun baselineToActivation(current: Float, isRight: Boolean): Float {
        val baseline = if (isRight) rightTriggerBaseline else leftTriggerBaseline
        if (baseline == null) {
            if (isRight) rightTriggerBaseline = current else leftTriggerBaseline = current
            return 0f
        }

        val upTravel = (1f - baseline).coerceAtLeast(0.08f)
        val downTravel = baseline.coerceAtLeast(0.08f)
        val deltaUp = (current - baseline) / upTravel
        val deltaDown = (baseline - current) / downTravel
        val active = maxOf(deltaUp, deltaDown).coerceIn(0f, 1f)

        // Eksende ufak oynama varsa baseline'i yavasca guncelle (drift temizligi).
        if (active < 0.04f) {
            val adjusted = baseline * 0.98f + current * 0.02f
            if (isRight) rightTriggerBaseline = adjusted else leftTriggerBaseline = adjusted
        }
        return active
    }

    private fun applyDeadzone(value: Float): Float {
        return if (abs(value) < axisDeadzone) 0f else value
    }

    private fun shapeTrigger(value: Float): Float {
        val clamped = value.coerceIn(0f, 1f)
        if (clamped <= triggerDeadzone) return 0f
        val normalized = ((clamped - triggerDeadzone) / (1f - triggerDeadzone)).coerceIn(0f, 1f)
        // Daha kontrollu tetik cevabi: kucuk basista az, sona dogru artan hiz.
        return normalized.pow(1.20f).coerceIn(0f, 1f)
    }

    private fun shapeSigned(value: Float, exponent: Float): Float {
        val clamped = value.coerceIn(-1f, 1f)
        val sign = if (clamped >= 0f) 1f else -1f
        return sign * abs(clamped).pow(exponent)
    }

    private fun hasAxis(event: MotionEvent, axis: Int): Boolean {
        val device = event.device ?: return false
        return device.getMotionRange(axis, event.source) != null
    }

    private fun acceptMotionDevice(deviceId: Int): Boolean {
        activeMotionDeviceId = deviceId
        return true
    }

    private fun acceptKeyDevice(deviceId: Int): Boolean {
        activeKeyDeviceId = deviceId
        return true
    }

    private fun expireStaleInputs() {
        val now = System.currentTimeMillis()
        if (lastMotionEventMs > 0L && now - lastMotionEventMs > analogInputTimeoutMs) {
            leftStickY = 0f
            rightStickX = 0f
            leftTriggerAxis = 0f
            rightTriggerAxis = 0f
            // Once the analog channel goes silent, drop the "axis active" latch *and* the
            // button latches together. Otherwise the button fallback kicks back in with a
            // ghost l2/r2 press still set, producing a full-throttle command from nothing.
            leftTriggerAxisActive = false
            rightTriggerAxisActive = false
            l2Pressed = false
            r2Pressed = false
            lastMotionEventMs = 0L
        }
        if (lastKeyEventMs > 0L && now - lastKeyEventMs > buttonInputTimeoutMs) {
            l2Pressed = false
            r2Pressed = false
            lastKeyEventMs = 0L
        }
    }

    private fun isGamepadInputDevice(device: InputDevice?, source: Int): Boolean {
        val effectiveSource = if (source != 0) source else (device?.sources ?: 0)
        return (effectiveSource and InputDevice.SOURCE_GAMEPAD) == InputDevice.SOURCE_GAMEPAD ||
            (effectiveSource and InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK
    }
}
