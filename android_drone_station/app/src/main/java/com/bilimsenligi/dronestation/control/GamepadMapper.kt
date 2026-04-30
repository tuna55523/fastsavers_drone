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

    private var leftStickX: Float = 0f
    private var rightStickY: Float = 0f
    private var leftTriggerAxis: Float = 0f
    private var rightTriggerAxis: Float = 0f

    private var l1Pressed: Boolean = false
    private var r1Pressed: Boolean = false
    private var l2Pressed: Boolean = false
    private var r2Pressed: Boolean = false
    // PS4 controller presents two separate logical devices on Android: one for axis/motion
    // events and one for key/button events. Tracking them separately prevents one channel
    // from being silently rejected when they arrive with different deviceIds.
    private var activeMotionDeviceId: Int? = null
    private var activeKeyDeviceId: Int? = null
    private var lastMotionEventMs: Long = 0L
    private var lastKeyEventMs: Long = 0L

    private val axisDeadzone = 0.12f
    private val triggerDeadzone = 0.05f
    private val analogInputTimeoutMs = 450L
    private val buttonInputTimeoutMs = 2500L
    private val maxForwardBack = 65
    private val maxLeftRight = 40
    private val maxUpDown = 45
    private val yawByBumper = 42

    fun onGenericMotionEvent(event: MotionEvent): Boolean {
        if (event.action != MotionEvent.ACTION_MOVE) return false
        if (!isGamepadInputDevice(event.device, event.source)) return false
        if (!acceptMotionDevice(event.deviceId)) return false
        lastMotionEventMs = System.currentTimeMillis()

        leftStickX = readCenteredAxis(
            event,
            MotionEvent.AXIS_X,
            MotionEvent.AXIS_HAT_X,
        )

        // On PS4 DualShock 4, AXIS_RZ and AXIS_Z are trigger axes that rest at -1.0.
        // Falling back to them causes a spurious ud=+45 (full-up) command with no input.
        // AXIS_RY is the correct right-stick-Y axis across all standard gamepad profiles.
        rightStickY = readCenteredAxis(event, MotionEvent.AXIS_RY)

        rightTriggerAxis = readTriggerAxis(
            event,
            MotionEvent.AXIS_RTRIGGER,
            MotionEvent.AXIS_GAS,
        )
        leftTriggerAxis = readTriggerAxis(
            event,
            MotionEvent.AXIS_LTRIGGER,
            MotionEvent.AXIS_BRAKE,
        )
        return true
    }

    fun onKeyDown(event: KeyEvent): List<Action> {
        if (!isGamepadInputDevice(event.device, event.source)) return emptyList()
        if (!acceptKeyDevice(event.deviceId)) return emptyList()
        lastKeyEventMs = System.currentTimeMillis()

        val actions = mutableListOf<Action>()
        val firstPress = event.repeatCount == 0

        when (event.keyCode) {
            KeyEvent.KEYCODE_BUTTON_L1 -> l1Pressed = true
            KeyEvent.KEYCODE_BUTTON_R1 -> r1Pressed = true
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

            // PS4 Circle button maps to BUTTON_B on Android.
            KeyEvent.KEYCODE_BUTTON_B -> if (firstPress) actions.add(Action.TRACKING_START)

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
            KeyEvent.KEYCODE_BUTTON_L1 -> l1Pressed = false
            KeyEvent.KEYCODE_BUTTON_R1 -> r1Pressed = false
            KeyEvent.KEYCODE_BUTTON_L2 -> l2Pressed = false
            KeyEvent.KEYCODE_BUTTON_R2 -> r2Pressed = false
            else -> return false
        }
        return true
    }

    fun currentRcCommand(): RcCommand {
        expireStaleInputs()

        // Prefer the analog axis for proportional control. Fall back to the digital button
        // only when no axis value is present — using maxOf would override a partial axis
        // press with full-speed whenever the button is also held.
        val forward = when {
            rightTriggerAxis > 0f -> shapeTrigger(rightTriggerAxis)
            r2Pressed -> 1f
            else -> 0f
        }
        val backward = when {
            leftTriggerAxis > 0f -> shapeTrigger(leftTriggerAxis)
            l2Pressed -> 1f
            else -> 0f
        }

        val fbRaw = forward - backward
        val fb = scaleSigned(fbRaw, maxForwardBack)

        // Requirement: left analog helps steering while moving forward/back.
        val lr = if (abs(fb) >= 4) {
            scaleSigned(shapeSigned(applyDeadzone(leftStickX), exponent = 0.88f), maxLeftRight)
        } else {
            0
        }

        // Requirement: right analog controls up/down.
        val ud = scaleSigned(shapeSigned(-applyDeadzone(rightStickY), exponent = 0.88f), maxUpDown)

        val yaw = when {
            r1Pressed && !l1Pressed -> yawByBumper
            l1Pressed && !r1Pressed -> -yawByBumper
            else -> 0
        }

        return RcCommand(
            leftRight = lr,
            forwardBack = fb,
            upDown = ud,
            yaw = yaw,
        )
    }

    fun hasManualInput(): Boolean {
        return kotlin.math.abs(leftStickX) >= axisDeadzone ||
            kotlin.math.abs(rightStickY) >= axisDeadzone ||
            leftTriggerAxis >= 0.05f ||
            rightTriggerAxis >= 0.05f ||
            l1Pressed ||
            r1Pressed ||
            l2Pressed ||
            r2Pressed
    }

    fun reset() {
        leftStickX = 0f
        rightStickY = 0f
        leftTriggerAxis = 0f
        rightTriggerAxis = 0f
        l1Pressed = false
        r1Pressed = false
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

    private fun readTriggerAxis(event: MotionEvent, vararg axes: Int): Float {
        val device = event.device ?: return 0f
        for (axis in axes) {
            val range = device.getMotionRange(axis, event.source) ?: continue
            val value = event.getAxisValue(axis)
            val normalized = if (range.min < 0f && range.max > 0f) {
                if (abs(value) <= maxOf(range.flat, triggerDeadzone)) continue
                ((value - range.min) / (range.max - range.min)).coerceIn(0f, 1f)
            } else {
                value.coerceIn(0f, 1f)
            }
            if (normalized > 0.02f) return normalized
        }
        return 0f
    }

    private fun applyDeadzone(value: Float): Float {
        return if (abs(value) < axisDeadzone) 0f else value
    }

    private fun shapeTrigger(value: Float): Float {
        val clamped = value.coerceIn(0f, 1f)
        if (clamped <= triggerDeadzone) return 0f

        val normalized = ((clamped - triggerDeadzone) / (1f - triggerDeadzone)).coerceIn(0f, 1f)
        val curved = normalized.pow(0.66f)
        return (0.08f + (curved * 0.92f)).coerceIn(0f, 1f)
    }

    private fun shapeSigned(value: Float, exponent: Float): Float {
        val clamped = value.coerceIn(-1f, 1f)
        val sign = if (clamped >= 0f) 1f else -1f
        return sign * abs(clamped).pow(exponent)
    }

    private fun acceptMotionDevice(deviceId: Int): Boolean {
        val active = activeMotionDeviceId
        return if (active == null || active == deviceId) {
            activeMotionDeviceId = deviceId
            true
        } else {
            false
        }
    }

    private fun acceptKeyDevice(deviceId: Int): Boolean {
        val active = activeKeyDeviceId
        return if (active == null || active == deviceId) {
            activeKeyDeviceId = deviceId
            true
        } else {
            false
        }
    }

    private fun expireStaleInputs() {
        val now = System.currentTimeMillis()
        if (lastMotionEventMs > 0L && now - lastMotionEventMs > analogInputTimeoutMs) {
            leftStickX = 0f
            rightStickY = 0f
            leftTriggerAxis = 0f
            rightTriggerAxis = 0f
            lastMotionEventMs = 0L
        }
        if (lastKeyEventMs > 0L && now - lastKeyEventMs > buttonInputTimeoutMs) {
            l1Pressed = false
            r1Pressed = false
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
