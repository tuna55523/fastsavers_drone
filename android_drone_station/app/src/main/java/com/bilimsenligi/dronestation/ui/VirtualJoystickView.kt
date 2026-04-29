package com.bilimsenligi.dronestation.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.hypot

class VirtualJoystickView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0,
) : View(context, attrs, defStyleAttr) {

    interface OnMoveListener {
        fun onMove(nx: Float, ny: Float, active: Boolean)
    }

    private var listener: OnMoveListener? = null

    private val basePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#66334155")
        style = Paint.Style.FILL
    }
    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#88A5B4CC")
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    private val knobPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#E5F3FF")
        style = Paint.Style.FILL
    }

    private var cx = 0f
    private var cy = 0f
    private var baseRadius = 0f
    private var knobRadius = 0f

    private var knobX = 0f
    private var knobY = 0f
    @Volatile
    private var userActive = false

    fun setOnMoveListener(listener: OnMoveListener?) {
        this.listener = listener
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        cx = w / 2f
        cy = h / 2f
        baseRadius = (minOf(w, h) * 0.48f)
        knobRadius = baseRadius * 0.34f
        resetKnob()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawCircle(cx, cy, baseRadius, basePaint)
        canvas.drawCircle(cx, cy, baseRadius, ringPaint)
        canvas.drawCircle(knobX, knobY, knobRadius, knobPaint)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE, MotionEvent.ACTION_POINTER_DOWN -> {
                userActive = true
                moveKnob(event.x, event.y, true)
                return true
            }

            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL, MotionEvent.ACTION_POINTER_UP -> {
                userActive = false
                resetKnob()
                listener?.onMove(0f, 0f, false)
                invalidate()
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    private fun moveKnob(x: Float, y: Float, active: Boolean) {
        val dx = x - cx
        val dy = y - cy
        val maxR = (baseRadius - knobRadius).coerceAtLeast(1f)
        val dist = hypot(dx, dy)

        val (clampedDx, clampedDy) = if (dist <= maxR) {
            dx to dy
        } else {
            val scale = maxR / dist
            (dx * scale) to (dy * scale)
        }

        knobX = cx + clampedDx
        knobY = cy + clampedDy

        val nx = (clampedDx / maxR).coerceIn(-1f, 1f)
        val ny = (clampedDy / maxR).coerceIn(-1f, 1f)
        listener?.onMove(nx, ny, active)
        invalidate()
    }

    private fun resetKnob() {
        knobX = cx
        knobY = cy
        invalidate()
    }

    fun isUserActive(): Boolean = userActive

    fun setVisualNormalized(nx: Float, ny: Float) {
        if (userActive) return
        val maxR = (baseRadius - knobRadius).coerceAtLeast(1f)
        val clampedNx = nx.coerceIn(-1f, 1f)
        val clampedNy = ny.coerceIn(-1f, 1f)
        knobX = cx + (clampedNx * maxR)
        knobY = cy + (clampedNy * maxR)
        invalidate()
    }
}
