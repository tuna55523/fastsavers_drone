package com.bilimsenligi.dronestation.control

import android.view.InputDevice
import android.view.KeyEvent
import android.view.MotionEvent
import kotlin.math.abs
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

    private val axisDeadzone = 0.12f
    private val maxForwardBack = 55
    private val maxLeftRight = 35
    private val maxUpDown = 50
    private val yawByBumper = 45

    fun onGenericMotionEvent(event: MotionEvent): Boolean {
        if (event.action != MotionEvent.ACTION_MOVE) return false
        if ((event.source and InputDevice.SOURCE_JOYSTICK) != InputDevice.SOURCE_JOYSTICK) return false

        leftStickX = centeredAxis(event, MotionEvent.AXIS_X)

        // Different gamepads expose right stick vertical on different axes.
        val ry = centeredAxis(event, MotionEvent.AXIS_RY)
        rightStickY = if (ry != 0f) ry else centeredAxis(event, MotionEvent.AXIS_Z)

        val rTrigger = axisWithFallback(
            event,
            MotionEvent.AXIS_RTRIGGER,
            MotionEvent.AXIS_GAS,
        )
        val lTrigger = axisWithFallback(
            event,
            MotionEvent.AXIS_LTRIGGER,
            MotionEvent.AXIS_BRAKE,
        )

        rightTriggerAxis = normalizeTrigger(rTrigger)
        leftTriggerAxis = normalizeTrigger(lTrigger)
        return true
    }

    fun onKeyDown(keyCode: Int, repeatCount: Int): List<Action> {
        val actions = mutableListOf<Action>()
        val firstPress = repeatCount == 0

        when (keyCode) {
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

    fun onKeyUp(keyCode: Int): Boolean {
        when (keyCode) {
            KeyEvent.KEYCODE_BUTTON_L1 -> l1Pressed = false
            KeyEvent.KEYCODE_BUTTON_R1 -> r1Pressed = false
            KeyEvent.KEYCODE_BUTTON_L2 -> l2Pressed = false
            KeyEvent.KEYCODE_BUTTON_R2 -> r2Pressed = false
            else -> return false
        }
        return true
    }

    fun currentRcCommand(): RcCommand {
        val forward = maxOf(rightTriggerAxis, if (r2Pressed) 1f else 0f)
        val backward = maxOf(leftTriggerAxis, if (l2Pressed) 1f else 0f)

        val fbRaw = forward - backward
        val fb = scaleSigned(fbRaw, maxForwardBack)

        // Requirement: left analog helps steering while moving forward/back.
        val lr = if (abs(fb) >= 5) {
            scaleSigned(applyDeadzone(leftStickX), maxLeftRight)
        } else {
            0
        }

        // Requirement: right analog controls up/down.
        val ud = scaleSigned(-applyDeadzone(rightStickY), maxUpDown)

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

    private fun normalizeTrigger(value: Float): Float {
        val v = value.coerceIn(0f, 1f)
        return if (v < 0.02f) 0f else v
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

    private fun axisWithFallback(event: MotionEvent, primary: Int, secondary: Int): Float {
        val a = event.getAxisValue(primary)
        if (abs(a) > 1e-4f) return a
        return event.getAxisValue(secondary)
    }

    private fun applyDeadzone(value: Float): Float {
        return if (abs(value) < axisDeadzone) 0f else value
    }
}
