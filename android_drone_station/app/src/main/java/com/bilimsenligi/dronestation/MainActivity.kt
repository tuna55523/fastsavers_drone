package com.bilimsenligi.dronestation

import android.Manifest
import android.content.Context
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.Rect
import android.hardware.input.InputManager
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.net.wifi.WifiManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Looper
import android.view.InputDevice
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.TextureView
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.bilimsenligi.dronestation.control.GamepadMapper
import com.bilimsenligi.dronestation.drone.TelloClient
import com.bilimsenligi.dronestation.drone.TelloVideoManager
import com.bilimsenligi.dronestation.tracking.OnDevicePersonTracker
import com.bilimsenligi.dronestation.tracking.TrackingAssistMixer
import com.bilimsenligi.dronestation.tracking.TrackingManager
import com.bilimsenligi.dronestation.tracking.TrackingTelemetryReceiver
import com.bilimsenligi.dronestation.ui.VirtualJoystickView
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity(),
    TelloClient.Listener,
    TelloVideoManager.Listener,
    TrackingTelemetryReceiver.Listener,
    InputManager.InputDeviceListener {

    private lateinit var statusText: TextView
    private lateinit var batteryText: TextView
    private lateinit var trackingText: TextView
    private lateinit var recordText: TextView
    private lateinit var commandText: TextView
    private lateinit var videoHintText: TextView
    private lateinit var droneWifiText: TextView
    private lateinit var controllerBtText: TextView
    private lateinit var fullScreenConnectionChip: TextView
    private lateinit var fullScreenBatteryChip: TextView
    private lateinit var fullScreenTrackingChip: TextView
    private lateinit var fullScreenControllerChip: TextView

    private lateinit var connectButton: Button
    private lateinit var chargeButton: Button
    private lateinit var takeoffLandButton: Button
    private lateinit var trackingToggleButton: Button
    private lateinit var fullScreenTrackingToggleButton: Button
    private lateinit var nextTargetButton: Button
    private lateinit var openVideoPanelButton: Button
    private lateinit var closeVideoPanelButton: Button

    private lateinit var mainLeftVirtualJoystick: VirtualJoystickView
    private lateinit var videoTexture: TextureView
    private lateinit var fullScreenTexture: TextureView
    private lateinit var mainScroll: ScrollView
    private lateinit var fullScreenPanel: FrameLayout
    private lateinit var fullScreenAxisPad: FrameLayout
    private lateinit var activeVideoTexture: TextureView
    private lateinit var leftVirtualJoystick: VirtualJoystickView
    private lateinit var mainTouchUpButton: Button
    private lateinit var mainTouchDownButton: Button
    private lateinit var mainTouchYawLeftButton: Button
    private lateinit var mainTouchYawRightButton: Button
    private lateinit var touchUpButton: Button
    private lateinit var touchDownButton: Button
    private lateinit var touchYawLeftButton: Button
    private lateinit var touchYawRightButton: Button
    private lateinit var flipForwardButton: Button
    private lateinit var flipBackButton: Button
    private lateinit var flipLeftButton: Button
    private lateinit var flipRightButton: Button

    private val telloClient = TelloClient()
    private val telloVideo = TelloVideoManager()
    private val gamepadMapper = GamepadMapper()
    private val trackingManager = TrackingManager()
    private val trackingRx = TrackingTelemetryReceiver()
    private val trackingMixer = TrackingAssistMixer()
    private lateinit var onDeviceTracker: OnDevicePersonTracker

    private val commandExecutor = Executors.newSingleThreadExecutor()
    private val rcScheduler = Executors.newSingleThreadScheduledExecutor()
    private val visionScheduler = Executors.newSingleThreadScheduledExecutor()
    private val linkWatchdogScheduler = Executors.newSingleThreadScheduledExecutor()
    private val keepAliveScheduler = Executors.newSingleThreadScheduledExecutor()
    private val uiStatusScheduler = Executors.newSingleThreadScheduledExecutor()
    private val gamepadUiScheduler = Executors.newSingleThreadScheduledExecutor()

    private val batteryFlipMin = 25
    private val reconnectStateTimeoutMs = 8500L
    private val maxReconnectAttempts = 6
    private val trackingSampleStaleMs = 900L
    // Son-gun guvenli profil: panel komutlarini daha kontrollu yap.
    private val touchStrafeMax = 28
    private val touchForwardBackMax = 40
    private val touchUpDownRate = 24
    private val touchYawRate = 22
    private val touchAxisDeadzone = 0.10f
    private val touchAxisExpoLr = 1.35
    private val touchAxisExpoFb = 1.30
    private val rcNeutralDeadband = 3
    // 50ms RC tikinde eksen basina maksimum degisim adimi.
    private val rcSlewLrStep = 6
    private val rcSlewFbStep = 7
    private val rcSlewUdStep = 5
    private val rcSlewYawStep = 5
    // Hedef sifira inerken daha hizli frenleyip drift'i azalt.
    private val rcStopLrStep = 12
    private val rcStopFbStep = 14
    private val rcStopUdStep = 10
    private val rcStopYawStep = 10
    private val rcZeroSnapThreshold = 2

    @Volatile
    private var isAirborne = false

    @Volatile
    private var isVideoRecording = false

    @Volatile
    private var latestTrackingSample: TrackingTelemetryReceiver.TrackingSample? = null

    @Volatile
    private var recoveryBusy = false

    @Volatile
    private var staleStrikeCount = 0

    @Volatile
    private var isVideoPanelOpen = false

    @Volatile
    private var touchLr = 0

    @Volatile
    private var touchFb = 0

    @Volatile
    private var touchUd = 0

    @Volatile
    private var touchYaw = 0

    @Volatile
    private var latestBatteryPercent: Int? = null

    @Volatile
    private var autoReconnectEnabled = false

    @Volatile
    private var reconnectInProgress = false

    @Volatile
    private var reconnectAttempts = 0

    @Volatile
    private var nextReconnectAtMs = 0L

    @Volatile
    private var connectedAtMs = 0L

    @Volatile
    private var activeTouchJoystickId: Int? = null

    @Volatile
    private var controllerRecoveryPending = false

    @Volatile
    private var lastSentRc = GamepadMapper.RcCommand(0, 0, 0, 0)

    @Volatile
    private var lastTrackerInitWarnMs = 0L

    private var wifiLock: WifiManager.WifiLock? = null
    private var wifiBindCallback: ConnectivityManager.NetworkCallback? = null
    private lateinit var inputManager: InputManager
    private var inputListenerRegistered = false
    private val touchUpSources = mutableSetOf<Int>()
    private val touchDownSources = mutableSetOf<Int>()
    private val touchYawLeftSources = mutableSetOf<Int>()
    private val touchYawRightSources = mutableSetOf<Int>()
    private var touchJoysticks: List<VirtualJoystickView> = emptyList()
    private var touchUpButtons: List<Button> = emptyList()
    private var touchDownButtons: List<Button> = emptyList()
    private var touchYawLeftButtons: List<Button> = emptyList()
    private var touchYawRightButtons: List<Button> = emptyList()

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { result ->
        val denied = result.filterValues { granted -> !granted }
        if (denied.isNotEmpty()) {
            toast("Bazi izinler verilmedi, kol durumu etkilenebilir")
        }
        updateLinkStatusUi()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        statusText = findViewById(R.id.statusText)
        batteryText = findViewById(R.id.batteryText)
        trackingText = findViewById(R.id.trackingText)
        recordText = findViewById(R.id.recordText)
        commandText = findViewById(R.id.commandText)
        videoHintText = findViewById(R.id.videoHintText)
        droneWifiText = findViewById(R.id.droneWifiText)
        controllerBtText = findViewById(R.id.controllerBtText)
        fullScreenConnectionChip = findViewById(R.id.fullScreenConnectionChip)
        fullScreenBatteryChip = findViewById(R.id.fullScreenBatteryChip)
        fullScreenTrackingChip = findViewById(R.id.fullScreenTrackingChip)
        fullScreenControllerChip = findViewById(R.id.fullScreenControllerChip)

        connectButton = findViewById(R.id.connectButton)
        chargeButton = findViewById(R.id.chargeButton)
        takeoffLandButton = findViewById(R.id.takeoffLandButton)
        trackingToggleButton = findViewById(R.id.trackingToggleButton)
        fullScreenTrackingToggleButton = findViewById(R.id.fullScreenTrackingToggleButton)
        nextTargetButton = findViewById(R.id.nextTargetButton)
        openVideoPanelButton = findViewById(R.id.openVideoPanelButton)
        closeVideoPanelButton = findViewById(R.id.closeVideoPanelButton)

        mainLeftVirtualJoystick = findViewById(R.id.mainLeftVirtualJoystick)
        videoTexture = findViewById(R.id.videoTexture)
        fullScreenTexture = findViewById(R.id.fullScreenTexture)
        mainScroll = findViewById(R.id.mainScroll)
        fullScreenPanel = findViewById(R.id.fullScreenPanel)
        fullScreenAxisPad = findViewById(R.id.fullScreenAxisPad)
        leftVirtualJoystick = findViewById(R.id.leftVirtualJoystick)
        mainTouchUpButton = findViewById(R.id.mainTouchUpButton)
        mainTouchDownButton = findViewById(R.id.mainTouchDownButton)
        mainTouchYawLeftButton = findViewById(R.id.mainTouchYawLeftButton)
        mainTouchYawRightButton = findViewById(R.id.mainTouchYawRightButton)
        touchUpButton = findViewById(R.id.touchUpButton)
        touchDownButton = findViewById(R.id.touchDownButton)
        touchYawLeftButton = findViewById(R.id.touchYawLeftButton)
        touchYawRightButton = findViewById(R.id.touchYawRightButton)
        flipForwardButton = findViewById(R.id.flipForwardButton)
        flipBackButton = findViewById(R.id.flipBackButton)
        flipLeftButton = findViewById(R.id.flipLeftButton)
        flipRightButton = findViewById(R.id.flipRightButton)
        activeVideoTexture = videoTexture
        inputManager = getSystemService(Context.INPUT_SERVICE) as InputManager

        telloClient.setListener(this)
        telloVideo.setListener(this)
        telloVideo.attachTextureView(activeVideoTexture)
        onDeviceTracker = OnDevicePersonTracker(this)
        trackingRx.setListener(this)
        trackingRx.start()

        connectButton.setOnClickListener {
            if (telloClient.isConnected()) {
                disconnectDrone()
            } else {
                connectDrone()
            }
        }

        chargeButton.setOnClickListener {
            refreshBatteryOnly()
        }

        openVideoPanelButton.setOnClickListener {
            enterVideoPanel()
        }

        closeVideoPanelButton.setOnClickListener {
            exitVideoPanel()
        }

        takeoffLandButton.setOnClickListener {
            handleAction(GamepadMapper.Action.TAKEOFF_LAND_TOGGLE)
        }

        trackingToggleButton.setOnClickListener {
            toggleTrackingMode()
        }

        fullScreenTrackingToggleButton.setOnClickListener {
            toggleTrackingMode()
        }

        nextTargetButton.setOnClickListener {
            val idx = trackingManager.nextTarget()
            latestTrackingSample = null
            postCommandText("Takip hedefi degisti -> #$idx")
            updateTrackingUi()
        }

        flipForwardButton.setOnClickListener {
            handleAction(GamepadMapper.Action.FLIP_FORWARD)
        }

        flipBackButton.setOnClickListener {
            handleAction(GamepadMapper.Action.FLIP_BACK)
        }

        flipLeftButton.setOnClickListener {
            handleAction(GamepadMapper.Action.FLIP_LEFT)
        }

        flipRightButton.setOnClickListener {
            handleAction(GamepadMapper.Action.FLIP_RIGHT)
        }

        ensureRuntimePermissions()
        bindTouchFlightControls()
        configureFullscreenTouchTargets()
        registerInputDeviceListenerIfNeeded()

        updateConnectionUi(false)
        updateTrackingUi()
        updateRecordUi(false)
        updateLinkStatusUi()
        updateFullscreenHud()
        takeoffLandButton.text = "Kalkis Yap (X)"
        startSchedulers()
    }

    override fun onDestroy() {
        super.onDestroy()
        autoReconnectEnabled = false
        reconnectInProgress = false
        unregisterInputDeviceListener()
        trackingRx.stop()
        telloVideo.stop()
        telloClient.disconnect()
        rcScheduler.shutdownNow()
        visionScheduler.shutdownNow()
        linkWatchdogScheduler.shutdownNow()
        keepAliveScheduler.shutdownNow()
        uiStatusScheduler.shutdownNow()
        gamepadUiScheduler.shutdownNow()
        resetTouchManual()
        releaseWifiLock()
        unbindNetwork()
        commandExecutor.shutdownNow()
    }

    override fun onResume() {
        super.onResume()
        registerInputDeviceListenerIfNeeded()
        ensureRuntimePermissions()
        updateLinkStatusUi()
    }

    override fun onPause() {
        super.onPause()
        gamepadMapper.reset()
        resetTouchManual()
        lastSentRc = GamepadMapper.RcCommand(0, 0, 0, 0)
    }

    override fun onGenericMotionEvent(event: MotionEvent): Boolean {
        val handled = gamepadMapper.onGenericMotionEvent(event)
        return handled || super.onGenericMotionEvent(event)
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        if (keyCode == KeyEvent.KEYCODE_BACK && isVideoPanelOpen) {
            exitVideoPanel()
            return true
        }

        val actions = gamepadMapper.onKeyDown(event)
        val handledByMapper = actions.isNotEmpty() || (isContinuousGamepadKey(keyCode) && isGamepadKeyEvent(event))
        if (actions.isNotEmpty()) {
            for (action in actions) {
                handleAction(action)
            }
        }
        return handledByMapper || super.onKeyDown(keyCode, event)
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent): Boolean {
        val handled = gamepadMapper.onKeyUp(event)
        return handled || super.onKeyUp(keyCode, event)
    }

    override fun onInputDeviceAdded(deviceId: Int) {
        if (!isPhysicalGamepad(deviceId)) return
        runOnUiThread { updateLinkStatusUi() }
        scheduleControllerLinkRefresh("kol baglandi", resetMapper = true)
    }

    override fun onInputDeviceRemoved(deviceId: Int) {
        gamepadMapper.onInputDeviceRemoved(deviceId)
        runOnUiThread { updateLinkStatusUi() }
        scheduleControllerLinkRefresh("kol ayrildi", resetMapper = false)
    }

    override fun onInputDeviceChanged(deviceId: Int) {
        if (!isPhysicalGamepad(deviceId)) return
        runOnUiThread { updateLinkStatusUi() }
        scheduleControllerLinkRefresh("kol guncellendi", resetMapper = false)
    }

    override fun onLog(message: String) {
        runOnUiThread { commandText.text = "Son komut: $message" }
    }

    override fun onState(state: TelloClient.TelloState) {
        latestBatteryPercent = state.batteryPercent
        runOnUiThread {
            val batText = state.batteryPercent?.let { "$it%" } ?: "-"
            batteryText.text = "Batarya: $batText"
            updateFullscreenHud()
        }
    }

    override fun onVideoLog(message: String) {
        postCommandText(message)
    }

    override fun onTrackingSample(sample: TrackingTelemetryReceiver.TrackingSample) {
        latestTrackingSample = sample
    }

    override fun onTrackingLog(message: String) {
        postCommandText(message)
    }

    override fun onSnapshotSaved(file: File) {
        runOnUiThread { toast("Foto kaydedildi: ${file.name}") }
    }

    override fun onRecordStarted(file: File) {
        runOnUiThread {
            updateRecordUi(true)
            toast("Video kaydi basladi: ${file.name}")
        }
    }

    override fun onRecordStopped(file: File) {
        runOnUiThread {
            updateRecordUi(false)
            toast("Video kaydi durdu: ${file.name}")
        }
    }

    private fun connectDrone() {
        connectButton.isEnabled = false
        statusText.text = "Baglanti: baglaniyor..."
        updateFullscreenHud()
        autoReconnectEnabled = false
        reconnectInProgress = false
        reconnectAttempts = 0
        nextReconnectAtMs = 0L

        commandExecutor.execute {
            bindToWifiNetwork(forceRebind = true)
            acquireWifiLock()

            val ok = telloClient.connect()
            val battery = if (ok) telloClient.queryBattery() else null

            if (ok) {
                val videoOk = telloVideo.start()
                runOnUiThread {
                    videoHintText.text = if (videoOk) "Video akis aktif" else "Video baslatilamadi"
                }
            } else {
                releaseWifiLock()
                unbindNetwork()
            }

            runOnUiThread {
                updateConnectionUi(ok)
                if (ok && battery != null) batteryText.text = "Batarya: $battery%"
                if (ok) {
                    staleStrikeCount = 0
                    reconnectAttempts = 0
                    nextReconnectAtMs = 0L
                    connectedAtMs = System.currentTimeMillis()
                    latestBatteryPercent = battery
                    lastSentRc = GamepadMapper.RcCommand(0, 0, 0, 0)
                    autoReconnectEnabled = true
                } else {
                    autoReconnectEnabled = false
                }
                connectButton.isEnabled = true
                updateLinkStatusUi()
                toast(if (ok) "Tello baglandi" else "Tello baglanamadi")
            }
        }
    }

    private fun disconnectDrone() {
        autoReconnectEnabled = false
        reconnectInProgress = false
        reconnectAttempts = 0
        nextReconnectAtMs = 0L
        commandExecutor.execute {
            if (telloClient.isConnected()) {
                telloClient.streamOff()
            }
            telloVideo.stop()
            telloClient.disconnect()
            gamepadMapper.reset()
            isAirborne = false
            isVideoRecording = false
            staleStrikeCount = 0
            connectedAtMs = 0L
            latestBatteryPercent = null
            latestTrackingSample = null
            lastSentRc = GamepadMapper.RcCommand(0, 0, 0, 0)
            resetTouchManual()
            releaseWifiLock()
            unbindNetwork()

            runOnUiThread {
                updateConnectionUi(false)
                batteryText.text = "Batarya: -"
                takeoffLandButton.text = "Kalkis Yap (X)"
                updateRecordUi(false)
                updateLinkStatusUi()
                videoHintText.text = "Video bekleniyor..."
            }
        }
    }

    private fun refreshBatteryOnly() {
        commandExecutor.execute {
            if (!telloClient.isConnected()) {
                runOnUiThread { toast("Once Tello baglantisini ac") }
                return@execute
            }
            val battery = telloClient.queryBattery()
            runOnUiThread {
                if (battery == null) {
                    toast("Sarj bilgisi alinamadi")
                } else {
                    latestBatteryPercent = battery
                    batteryText.text = "Batarya: $battery%"
                    postCommandText("Sarj guncellendi: $battery%")
                    updateFullscreenHud()
                }
            }
        }
    }

    private fun updateConnectionUi(connected: Boolean) {
        if (connected) {
            statusText.text = "Baglanti: Aktif"
            connectButton.text = "Baglantiyi Kes"
        } else {
            statusText.text = "Baglanti: Kapali"
            connectButton.text = "Drone'a Baglan"
        }
        updateFullscreenHud()
    }

    private fun updateTrackingUi() {
        val st = trackingManager.status()
        val mode = if (st.enabled) "Acik" else "Kapali"
        trackingText.text = "Takip: $mode | Hedef #${st.targetIndex}"
        trackingToggleButton.text = if (st.enabled) "Takibi Durdur (Kare)" else "Takibi Baslat (Yuvarlak)"
        fullScreenTrackingToggleButton.text = if (st.enabled) "Takibi Durdur" else "Takibi Baslat"
        updateFullscreenHud()
    }

    private fun toggleTrackingMode() {
        val enabled = trackingManager.toggle()
        latestTrackingSample = null
        lastSentRc = GamepadMapper.RcCommand(0, 0, 0, 0)
        postCommandText(if (enabled) "Takip baslatildi" else "Takip durduruldu")
        updateTrackingUi()
    }

    private fun updateRecordUi(recording: Boolean) {
        recordText.text = "Kayit: ${if (recording) "Acik" else "Kapali"}"
    }

    private fun startSchedulers() {
        rcScheduler.scheduleAtFixedRate(
            {
                try {
                    if (!telloClient.isConnected()) return@scheduleAtFixedRate
                    val gamepadManual = gamepadMapper.currentRcCommand()
                    val touchManual = currentTouchRcCommand()
                    val manual = mergeManualInput(gamepadManual, touchManual)
                    val trackingStatus = trackingManager.status()
                    val mixed = trackingMixer.mix(
                        manual = manual,
                        trackingStatus = trackingStatus,
                        sample = latestTrackingSample,
                        nowMs = System.currentTimeMillis(),
                        staleTimeoutMs = trackingSampleStaleMs,
                    )
                    val rc = stabilizeRcCommand(mixed)
                    if (!isAirborne &&
                        rc.leftRight == 0 &&
                        rc.forwardBack == 0 &&
                        rc.upDown == 0 &&
                        rc.yaw == 0
                    ) {
                        lastSentRc = GamepadMapper.RcCommand(0, 0, 0, 0)
                        return@scheduleAtFixedRate
                    }
                    val smooth = applyRcSlew(rc)
                    telloClient.sendRcControl(smooth.leftRight, smooth.forwardBack, smooth.upDown, smooth.yaw)
                } catch (_: Exception) {
                }
            },
            0,
            50,
            TimeUnit.MILLISECONDS,
        )

        visionScheduler.scheduleAtFixedRate(
            {
                try {
                    if (!trackingManager.status().enabled) return@scheduleAtFixedRate
                    if (!telloVideo.isRunning()) return@scheduleAtFixedRate
                    if (!onDeviceTracker.isReady()) {
                        val now = System.currentTimeMillis()
                        if (now - lastTrackerInitWarnMs > 3000L) {
                            lastTrackerInitWarnMs = now
                            val why = onDeviceTracker.latestInitError()
                            if (!why.isNullOrBlank()) {
                                postCommandText("Takip modeli hazir degil: $why")
                            }
                        }
                        return@scheduleAtFixedRate
                    }

                    val now = System.currentTimeMillis()
                    val externalFresh = latestTrackingSample?.let { now - it.timestampMs <= 700L } == true
                    if (externalFresh) return@scheduleAtFixedRate

                    val bmp = activeVideoTexture.bitmap ?: return@scheduleAtFixedRate
                    try {
                        val sample = onDeviceTracker.detect(bmp) ?: return@scheduleAtFixedRate
                        latestTrackingSample = TrackingTelemetryReceiver.TrackingSample(
                            tx = sample.tx,
                            ty = sample.ty,
                            size = sample.size,
                            confidence = sample.confidence,
                            targetId = trackingManager.status().targetIndex,
                            timestampMs = now,
                        )
                    } finally {
                        bmp.recycle()
                    }
                } catch (_: Exception) {
                }
            },
            1,
            250,
            TimeUnit.MILLISECONDS,
        )

        linkWatchdogScheduler.scheduleAtFixedRate(
            {
                try {
                    if (!telloClient.isConnected()) {
                        if (autoReconnectEnabled) {
                            scheduleReconnectAttempt("baglanti kapali")
                        }
                        return@scheduleAtFixedRate
                    }

                    val st = telloClient.latestState()
                    val ageMs = if (st == null) Long.MAX_VALUE else (System.currentTimeMillis() - st.receivedAtMs)
                    val connectedAge = System.currentTimeMillis() - connectedAtMs

                    if (ageMs > 4500L) {
                        if (st == null && connectedAge < 15_000L) {
                            return@scheduleAtFixedRate
                        }
                        staleStrikeCount += 1
                        runOnUiThread {
                            statusText.text = "Baglanti: Zayif"
                            updateFullscreenHud()
                        }

                        if (staleStrikeCount >= 2 && !recoveryBusy && !reconnectInProgress) {
                            recoveryBusy = true
                            commandExecutor.execute {
                                try {
                                    bindToWifiNetwork(forceRebind = true)
                                    telloClient.sendCommandNoWait("command")
                                    telloClient.restartStateListener()
                                    if (!telloVideo.isRunning()) {
                                        telloClient.streamOn()
                                        telloVideo.start()
                                    }
                                } catch (_: Exception) {
                                } finally {
                                    recoveryBusy = false
                                }
                            }
                        }
                        if (ageMs > reconnectStateTimeoutMs && autoReconnectEnabled) {
                            scheduleReconnectAttempt("state timeout")
                        }
                    } else {
                        staleStrikeCount = 0
                        reconnectAttempts = 0
                        nextReconnectAtMs = 0L
                        runOnUiThread {
                            if (statusText.text.toString() != "Baglanti: Aktif") {
                                statusText.text = "Baglanti: Aktif"
                                updateFullscreenHud()
                            }
                        }
                    }
                } catch (_: Exception) {
                }
            },
            2,
            2,
            TimeUnit.SECONDS,
        )

        keepAliveScheduler.scheduleAtFixedRate(
            {
                try {
                    if (!telloClient.isConnected()) return@scheduleAtFixedRate
                    if (!gamepadMapper.hasManualInput() && !hasTouchManualInput()) {
                        telloClient.sendCommandNoWait("command")
                    }
                } catch (_: Exception) {
                }
            },
            3,
            7,
            TimeUnit.SECONDS,
        )

        uiStatusScheduler.scheduleAtFixedRate(
            {
                runOnUiThread {
                    updateLinkStatusUi()
                }
            },
            0,
            1,
            TimeUnit.SECONDS,
        )

        gamepadUiScheduler.scheduleAtFixedRate(
            {
                runOnUiThread {
                    updateTouchControlVisuals()
                }
            },
            0,
            50,
            TimeUnit.MILLISECONDS,
        )
    }

    private fun handleAction(action: GamepadMapper.Action) {
        commandExecutor.execute {
            if (!telloClient.isConnected() &&
                action != GamepadMapper.Action.TRACKING_START &&
                action != GamepadMapper.Action.TRACKING_STOP &&
                action != GamepadMapper.Action.TRACKING_TARGET_NEXT &&
                action != GamepadMapper.Action.VIDEO_PANEL_EXIT
            ) {
                runOnUiThread { toast("Once Tello baglantisini ac") }
                return@execute
            }

            when (action) {
                GamepadMapper.Action.TAKEOFF_LAND_TOGGLE -> {
                    if (!isAirborne) {
                        val ok = telloClient.takeoff()
                        if (ok) isAirborne = true
                        postCommandText(if (ok) "Kalkis OK" else "Kalkis basarisiz")
                    } else {
                        val ok = telloClient.land()
                        if (ok) isAirborne = false
                        postCommandText(if (ok) "Inis OK" else "Inis basarisiz")
                    }
                    runOnUiThread { takeoffLandButton.text = if (isAirborne) "Inis Yap (X)" else "Kalkis Yap (X)" }
                }

                GamepadMapper.Action.FLIP_FORWARD -> {
                    if (!isAirborne) postCommandText("Takla icin drone havada olmali")
                    else if (!canPerformFlip()) postCommandText("Takla iptal: batarya en az $batteryFlipMin% olmali")
                    else postCommandText(if (telloClient.flipForward()) "On takla OK" else "On takla basarisiz")
                }

                GamepadMapper.Action.FLIP_BACK -> {
                    if (!isAirborne) postCommandText("Takla icin drone havada olmali")
                    else if (!canPerformFlip()) postCommandText("Takla iptal: batarya en az $batteryFlipMin% olmali")
                    else postCommandText(if (telloClient.flipBack()) "Arka takla OK" else "Arka takla basarisiz")
                }

                GamepadMapper.Action.FLIP_LEFT -> {
                    if (!isAirborne) postCommandText("Takla icin drone havada olmali")
                    else if (!canPerformFlip()) postCommandText("Takla iptal: batarya en az $batteryFlipMin% olmali")
                    else postCommandText(if (telloClient.flipLeft()) "Sol takla OK" else "Sol takla basarisiz")
                }

                GamepadMapper.Action.FLIP_RIGHT -> {
                    if (!isAirborne) postCommandText("Takla icin drone havada olmali")
                    else if (!canPerformFlip()) postCommandText("Takla iptal: batarya en az $batteryFlipMin% olmali")
                    else postCommandText(if (telloClient.flipRight()) "Sag takla OK" else "Sag takla basarisiz")
                }

                GamepadMapper.Action.PHOTO_CAPTURE -> {
                    val outDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
                    val file = telloVideo.saveSnapshot(outDir)
                    if (file == null) postCommandText("L3: Foto alinamadi (video akis aktif mi?)")
                    else postCommandText("L3: Foto kaydedildi -> ${file.name}")
                }

                GamepadMapper.Action.VIDEO_TOGGLE -> {
                    if (!isVideoRecording) {
                        val outDir = getExternalFilesDir(Environment.DIRECTORY_MOVIES)
                        val file = telloVideo.startRecording(outDir)
                        if (file == null) {
                            postCommandText("R3: Video kaydi baslatilamadi")
                        } else {
                            isVideoRecording = true
                            postCommandText("R3: Video kaydi basladi -> ${file.name}")
                        }
                    } else {
                        val file = telloVideo.stopRecording()
                        isVideoRecording = false
                        postCommandText("R3: Video kaydi durdu -> ${file?.name ?: "-"}")
                    }
                    runOnUiThread { updateRecordUi(isVideoRecording) }
                }

                GamepadMapper.Action.VIDEO_PANEL_EXIT -> {
                    runOnUiThread { exitVideoPanel() }
                }

                GamepadMapper.Action.TRACKING_START -> {
                    trackingManager.start()
                    latestTrackingSample = null
                    runOnUiThread { updateTrackingUi() }
                    postCommandText("Insan takibi baslatildi")
                }

                GamepadMapper.Action.TRACKING_STOP -> {
                    trackingManager.stop()
                    latestTrackingSample = null
                    lastSentRc = GamepadMapper.RcCommand(0, 0, 0, 0)
                    if (telloClient.isConnected()) {
                        telloClient.sendRcControl(0, 0, 0, 0)
                    }
                    runOnUiThread { updateTrackingUi() }
                    postCommandText("Insan takibi durduruldu")
                }

                GamepadMapper.Action.TRACKING_TARGET_NEXT -> {
                    val idx = trackingManager.nextTarget()
                    latestTrackingSample = null
                    runOnUiThread { updateTrackingUi() }
                    postCommandText("Takip hedefi degisti -> #$idx")
                }
            }
        }
    }

    private fun readBatteryForSafety(): Int? {
        val cached = latestBatteryPercent
        if (cached != null) return cached
        val queried = telloClient.queryBattery()
        if (queried != null) {
            latestBatteryPercent = queried
        }
        return queried
    }

    private fun canPerformFlip(): Boolean {
        val battery = readBatteryForSafety() ?: return false
        return battery >= batteryFlipMin
    }

    private fun scheduleReconnectAttempt(reason: String) {
        if (!autoReconnectEnabled || reconnectInProgress) return

        val now = System.currentTimeMillis()
        if (now < nextReconnectAtMs) return
        if (reconnectAttempts >= maxReconnectAttempts) {
            autoReconnectEnabled = false
            runOnUiThread {
                updateConnectionUi(false)
                connectButton.isEnabled = true
            }
            postCommandText("Reconnect durduruldu: deneme limiti asildi ($reason)")
            return
        }

        reconnectInProgress = true
        reconnectAttempts += 1
        val attempt = reconnectAttempts

        runOnUiThread {
            statusText.text = "Baglanti: Yeniden baglaniyor ($attempt/$maxReconnectAttempts)"
            connectButton.isEnabled = false
            updateFullscreenHud()
        }

        commandExecutor.execute {
            val ok = performReconnectAttempt()
            reconnectInProgress = false

            if (ok) {
                staleStrikeCount = 0
                reconnectAttempts = 0
                nextReconnectAtMs = 0L
                autoReconnectEnabled = true
                runOnUiThread {
                    updateConnectionUi(true)
                    connectButton.isEnabled = true
                    updateLinkStatusUi()
                }
                postCommandText("Yeniden baglandi")
                return@execute
            }

            val backoffMs = reconnectBackoffMs(attempt)
            nextReconnectAtMs = System.currentTimeMillis() + backoffMs
            runOnUiThread {
                updateConnectionUi(false)
                statusText.text = "Baglanti: Yeniden denenecek"
                connectButton.isEnabled = true
                updateFullscreenHud()
            }
            postCommandText("Reconnect basarisiz ($reason), ${backoffMs / 1000}s sonra tekrar denenecek")
        }
    }

    private fun performReconnectAttempt(): Boolean {
        return try {
            telloVideo.stop()
            telloClient.disconnect()
            Thread.sleep(200)

            bindToWifiNetwork(forceRebind = true)
            acquireWifiLock()

            val ok = telloClient.connect()
            if (!ok) {
                connectedAtMs = 0L
                return false
            }
            connectedAtMs = System.currentTimeMillis()

            val videoOk = telloVideo.start()
            val battery = telloClient.queryBattery()
            latestBatteryPercent = battery

            runOnUiThread {
                if (battery != null) {
                    batteryText.text = "Batarya: $battery%"
                }
                videoHintText.text = if (videoOk) "Video akis aktif" else "Video baslatilamadi"
            }
            true
        } catch (_: Exception) {
            false
        }
    }

    private fun reconnectBackoffMs(attempt: Int): Long {
        return when {
            attempt <= 1 -> 1500L
            attempt == 2 -> 3000L
            attempt == 3 -> 5000L
            else -> 8000L
        }
    }

    private fun bindTouchFlightControls() {
        touchJoysticks = listOf(mainLeftVirtualJoystick, leftVirtualJoystick)
        touchUpButtons = listOf(mainTouchUpButton, touchUpButton)
        touchDownButtons = listOf(mainTouchDownButton, touchDownButton)
        touchYawLeftButtons = listOf(mainTouchYawLeftButton, touchYawLeftButton)
        touchYawRightButtons = listOf(mainTouchYawRightButton, touchYawRightButton)

        touchJoysticks.forEach { joystick ->
            bindTouchJoystick(joystick)
        }
        touchUpButtons.forEach { button -> bindHoldButton(button, touchUpSources) }
        touchDownButtons.forEach { button -> bindHoldButton(button, touchDownSources) }
        touchYawLeftButtons.forEach { button -> bindHoldButton(button, touchYawLeftSources) }
        touchYawRightButtons.forEach { button -> bindHoldButton(button, touchYawRightSources) }

        updateTouchControlVisuals()
    }

    private fun bindTouchJoystick(joystick: VirtualJoystickView) {
        joystick.setOnMoveListener(object : VirtualJoystickView.OnMoveListener {
            override fun onMove(nx: Float, ny: Float, active: Boolean) {
                val sourceId = joystick.id
                if (!active) {
                    if (activeTouchJoystickId == sourceId) {
                        activeTouchJoystickId = null
                        touchLr = 0
                        touchFb = 0
                        mirrorTouchJoystick(sourceId, 0f, 0f)
                    }
                    return
                }

                activeTouchJoystickId = sourceId
                val x = shapeTouchAxis(nx, deadzone = touchAxisDeadzone, exponent = touchAxisExpoLr)
                val y = shapeTouchAxis(ny, deadzone = touchAxisDeadzone, exponent = touchAxisExpoFb)

                touchLr = (x * touchStrafeMax).toInt().coerceIn(-100, 100)
                touchFb = (-y * touchForwardBackMax).toInt().coerceIn(-100, 100)
                mirrorTouchJoystick(sourceId, x, y)
            }
        })
    }

    private fun bindHoldButton(button: Button, holdSources: MutableSet<Int>) {
        button.setOnTouchListener { _, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN, MotionEvent.ACTION_POINTER_DOWN -> {
                    holdSources.add(button.id)
                    updateTouchAxisState()
                    true
                }

                MotionEvent.ACTION_UP, MotionEvent.ACTION_POINTER_UP, MotionEvent.ACTION_CANCEL -> {
                    holdSources.remove(button.id)
                    updateTouchAxisState()
                    true
                }

                else -> true
            }
        }
    }

    private fun mirrorTouchJoystick(sourceId: Int, nx: Float, ny: Float) {
        touchJoysticks.forEach { joystick ->
            if (joystick.id != sourceId && !joystick.isUserActive()) {
                joystick.setVisualNormalized(nx, ny)
            }
        }
    }

    private fun updateTouchAxisState() {
        touchUd = when {
            touchUpSources.isNotEmpty() && touchDownSources.isEmpty() -> touchUpDownRate
            touchDownSources.isNotEmpty() && touchUpSources.isEmpty() -> -touchUpDownRate
            else -> 0
        }
        touchYaw = when {
            touchYawRightSources.isNotEmpty() && touchYawLeftSources.isEmpty() -> touchYawRate
            touchYawLeftSources.isNotEmpty() && touchYawRightSources.isEmpty() -> -touchYawRate
            else -> 0
        }
        updateTouchControlVisuals()
    }

    private fun currentTouchRcCommand(): GamepadMapper.RcCommand {
        return GamepadMapper.RcCommand(
            leftRight = touchLr,
            forwardBack = touchFb,
            upDown = touchUd,
            yaw = touchYaw,
        )
    }

    private fun mergeManualInput(
        gamepad: GamepadMapper.RcCommand,
        touch: GamepadMapper.RcCommand,
    ): GamepadMapper.RcCommand {
        fun pick(base: Int, over: Int): Int = if (kotlin.math.abs(over) >= 2) over else base
        return GamepadMapper.RcCommand(
            leftRight = pick(gamepad.leftRight, touch.leftRight),
            forwardBack = pick(gamepad.forwardBack, touch.forwardBack),
            upDown = pick(gamepad.upDown, touch.upDown),
            yaw = pick(gamepad.yaw, touch.yaw),
        )
    }

    private fun stabilizeRcCommand(command: GamepadMapper.RcCommand): GamepadMapper.RcCommand {
        fun deadband(value: Int): Int = if (kotlin.math.abs(value) <= rcNeutralDeadband) 0 else value
        return GamepadMapper.RcCommand(
            leftRight = deadband(command.leftRight),
            forwardBack = deadband(command.forwardBack),
            upDown = deadband(command.upDown),
            yaw = deadband(command.yaw),
        )
    }

    private fun applyRcSlew(target: GamepadMapper.RcCommand): GamepadMapper.RcCommand {
        val prev = lastSentRc
        val next = GamepadMapper.RcCommand(
            leftRight = slewAxis(prev.leftRight, target.leftRight, rcSlewLrStep, rcStopLrStep),
            forwardBack = slewAxis(prev.forwardBack, target.forwardBack, rcSlewFbStep, rcStopFbStep),
            upDown = slewAxis(prev.upDown, target.upDown, rcSlewUdStep, rcStopUdStep),
            yaw = slewAxis(prev.yaw, target.yaw, rcSlewYawStep, rcStopYawStep),
        )
        lastSentRc = next
        return next
    }

    private fun slewAxis(previous: Int, target: Int, step: Int, stopStep: Int): Int {
        val safeStep = step.coerceAtLeast(1)
        val safeStopStep = stopStep.coerceAtLeast(safeStep)
        if (target == 0) {
            val next = when {
                previous > safeStopStep -> previous - safeStopStep
                previous < -safeStopStep -> previous + safeStopStep
                else -> 0
            }
            return if (kotlin.math.abs(next) <= rcZeroSnapThreshold) 0 else next
        }
        return when {
            target > previous + safeStep -> previous + safeStep
            target < previous - safeStep -> previous - safeStep
            else -> target
        }.coerceIn(-100, 100)
    }

    private fun shapeTouchAxis(value: Float, deadzone: Float, exponent: Double): Float {
        val clamped = value.coerceIn(-1f, 1f)
        val magnitude = kotlin.math.abs(clamped)
        if (magnitude <= deadzone) return 0f

        val normalized = ((magnitude - deadzone) / (1f - deadzone)).coerceIn(0f, 1f)
        val curved = Math.pow(normalized.toDouble(), exponent).toFloat()
        return if (clamped >= 0f) curved else -curved
    }

    private fun resetTouchManual() {
        touchLr = 0
        touchFb = 0
        touchUd = 0
        touchYaw = 0
        activeTouchJoystickId = null
        touchUpSources.clear()
        touchDownSources.clear()
        touchYawLeftSources.clear()
        touchYawRightSources.clear()

        val applyUiReset = {
            touchJoysticks.forEach { joystick ->
                if (!joystick.isUserActive()) {
                    joystick.setVisualNormalized(0f, 0f)
                }
            }
            updateTouchControlVisuals()
        }
        if (Thread.currentThread() == Looper.getMainLooper().thread) {
            applyUiReset()
        } else {
            runOnUiThread { applyUiReset() }
        }
    }

    private fun enterVideoPanel() {
        if (isVideoPanelOpen) return
        isVideoPanelOpen = true
        mainScroll.visibility = View.GONE
        fullScreenPanel.visibility = View.VISIBLE
        updateFullscreenHud()
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
        setImmersiveMode(true)
        configureFullscreenTouchTargets()
        switchVideoSurface(fullScreenTexture)
    }

    private fun exitVideoPanel() {
        if (!isVideoPanelOpen) return
        isVideoPanelOpen = false
        fullScreenPanel.visibility = View.GONE
        mainScroll.visibility = View.VISIBLE
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_SENSOR_PORTRAIT
        setImmersiveMode(false)
        switchVideoSurface(videoTexture)
        resetTouchManual()
    }

    private fun switchVideoSurface(target: TextureView) {
        if (activeVideoTexture === target) return
        activeVideoTexture = target
        telloVideo.attachTextureView(target)
        if (!telloClient.isConnected()) return

        commandExecutor.execute {
            try {
                telloVideo.stop()
                telloClient.streamOn()
                val ok = telloVideo.start()
                runOnUiThread {
                    videoHintText.text = if (ok) "Video akis aktif" else "Video baslatilamadi"
                }
            } catch (_: Exception) {
            }
        }
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus && isVideoPanelOpen) {
            setImmersiveMode(true)
            configureFullscreenTouchTargets()
        } else if (!hasFocus) {
            gamepadMapper.reset()
            resetTouchManual()
        }
    }

    private fun setImmersiveMode(enabled: Boolean) {
        WindowCompat.setDecorFitsSystemWindows(window, !enabled)
        val controller = WindowInsetsControllerCompat(window, window.decorView)
        if (enabled) {
            controller.hide(WindowInsetsCompat.Type.statusBars() or WindowInsetsCompat.Type.navigationBars())
            controller.systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        } else {
            controller.show(WindowInsetsCompat.Type.statusBars() or WindowInsetsCompat.Type.navigationBars())
        }
    }

    private fun updateTouchControlVisuals() {
        val gamepad = gamepadMapper.currentRcCommand()
        val nx = (gamepad.leftRight / 34f).coerceIn(-1f, 1f)
        val ny = (-gamepad.forwardBack / 42f).coerceIn(-1f, 1f)
        touchJoysticks.forEach { joystick ->
            if (!joystick.isUserActive()) {
                joystick.setVisualNormalized(nx, ny)
            }
        }

        val upActive = touchUd > 0 || gamepad.upDown > 10
        val downActive = touchUd < 0 || gamepad.upDown < -10
        val yawLeftActive = touchYaw < 0 || gamepad.yaw < -10
        val yawRightActive = touchYaw > 0 || gamepad.yaw > 10

        touchUpButtons.forEach { button -> setControlButtonActive(button, upActive) }
        touchDownButtons.forEach { button -> setControlButtonActive(button, downActive) }
        touchYawLeftButtons.forEach { button -> setControlButtonActive(button, yawLeftActive) }
        touchYawRightButtons.forEach { button -> setControlButtonActive(button, yawRightActive) }
    }

    private fun setControlButtonActive(button: Button, active: Boolean) {
        button.alpha = if (active) 1.0f else 0.72f
        button.scaleX = if (active) 1.04f else 1.0f
        button.scaleY = if (active) 1.04f else 1.0f
    }

    private fun configureFullscreenTouchTargets() {
        fullScreenTexture.isClickable = false
        fullScreenTexture.isFocusable = false
        fullScreenTexture.isFocusableInTouchMode = false
        fullScreenPanel.isClickable = false
        fullScreenPanel.isFocusable = false

        fullScreenPanel.post {
            closeVideoPanelButton.bringToFront()
            fullScreenTrackingToggleButton.bringToFront()
            leftVirtualJoystick.bringToFront()
            fullScreenAxisPad.bringToFront()

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                fullScreenPanel.systemGestureExclusionRects = listOf(
                    expandedRectInParent(leftVirtualJoystick, fullScreenPanel, 24),
                    expandedRectInParent(fullScreenAxisPad, fullScreenPanel, 24),
                    expandedRectInParent(closeVideoPanelButton, fullScreenPanel, 12),
                    expandedRectInParent(fullScreenTrackingToggleButton, fullScreenPanel, 12),
                )
            }
        }
    }

    private fun expandedRectInParent(view: View, parent: View, extraDp: Int): Rect {
        val location = IntArray(2)
        val parentLocation = IntArray(2)
        view.getLocationOnScreen(location)
        parent.getLocationOnScreen(parentLocation)
        val extraPx = (extraDp * resources.displayMetrics.density).toInt()
        return Rect(
            location[0] - parentLocation[0] - extraPx,
            location[1] - parentLocation[1] - extraPx,
            location[0] - parentLocation[0] + view.width + extraPx,
            location[1] - parentLocation[1] + view.height + extraPx,
        )
    }

    private fun registerInputDeviceListenerIfNeeded() {
        if (inputListenerRegistered) return
        inputManager.registerInputDeviceListener(this, null)
        inputListenerRegistered = true
    }

    private fun unregisterInputDeviceListener() {
        if (!inputListenerRegistered) return
        try {
            inputManager.unregisterInputDeviceListener(this)
        } catch (_: Exception) {
        }
        inputListenerRegistered = false
    }

    private fun scheduleControllerLinkRefresh(reason: String, resetMapper: Boolean) {
        if (resetMapper) {
            gamepadMapper.reset()
        }
        runOnUiThread { updateTouchControlVisuals() }
        if (!telloClient.isConnected() || reconnectInProgress) return
        if (controllerRecoveryPending) return

        controllerRecoveryPending = true
        fullScreenPanel.postDelayed(
            {
                commandExecutor.execute {
                    try {
                        performControllerLinkRefresh(reason)
                    } finally {
                        controllerRecoveryPending = false
                    }
                }
            },
            450L,
        )
    }

    private fun performControllerLinkRefresh(reason: String) {
        try {
            bindToWifiNetwork(forceRebind = true)
            acquireWifiLock()
            telloVideo.stop()
            telloClient.sendCommandNoWait("command")
            telloClient.restartStateListener()
            telloClient.streamOn()
            val videoOk = telloVideo.start()
            val battery = telloClient.queryBattery()
            staleStrikeCount = 0
            connectedAtMs = System.currentTimeMillis()
            if (battery != null) {
                latestBatteryPercent = battery
            }

            runOnUiThread {
                if (battery != null) {
                    batteryText.text = "Batarya: $battery%"
                }
                videoHintText.text = if (videoOk) "Video akis aktif" else "Video baslatilamadi"
                updateLinkStatusUi()
            }
            postCommandText(
                if (videoOk) {
                    "Kol degisikligi toparlandi: $reason"
                } else {
                    "Kol degisti, video yeniden deneniyor"
                },
            )
        } catch (_: Exception) {
            postCommandText("Kol degisti, baglanti yenileme denemesi yapildi")
        }
    }

    private fun updateLinkStatusUi() {
        val wifiLabel = telloWifiLabel()
        val bluetoothAllowed = hasBluetoothConnectPermission()
        val controllerConnected = if (bluetoothAllowed) isGamepadConnected() else false

        droneWifiText.text = wifiLabel
        controllerBtText.text = when {
            !bluetoothAllowed -> "Kol BT: Izin Gerekli"
            controllerConnected -> "Kol BT: Bagli"
            else -> "Kol BT: Bagli degil"
        }
        updateFullscreenHud()
    }

    private fun updateFullscreenHud() {
        val trackingStatus = trackingManager.status()
        val bluetoothAllowed = hasBluetoothConnectPermission()
        val controllerConnected = if (bluetoothAllowed) isGamepadConnected() else false
        val connectionLabel = statusText.text.toString()
            .removePrefix("Baglanti:")
            .trim()
            .ifBlank { if (telloClient.isConnected()) "Aktif" else "Kapali" }

        fullScreenConnectionChip.text = "Drone $connectionLabel"
        fullScreenBatteryChip.text = latestBatteryPercent?.let { "Batarya $it%" } ?: "Batarya -"
        fullScreenTrackingChip.text = if (trackingStatus.enabled) {
            "Takip #${trackingStatus.targetIndex}"
        } else {
            "Takip Kapali"
        }
        fullScreenControllerChip.text = when {
            !bluetoothAllowed -> "Kol Izin"
            controllerConnected -> "Kol Bagli"
            else -> "Kol Yok"
        }
    }

    private fun ensureRuntimePermissions() {
        val needs = mutableListOf<String>()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED
        ) {
            needs.add(Manifest.permission.BLUETOOTH_CONNECT)
        }
        if (needs.isNotEmpty()) {
            permissionLauncher.launch(needs.toTypedArray())
        }
    }

    private fun hasBluetoothConnectPermission(): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) return true
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.BLUETOOTH_CONNECT,
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun telloWifiLabel(): String {
        val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val active = cm.activeNetwork ?: return "Drone Wi-Fi: Bagli degil"
        val caps = cm.getNetworkCapabilities(active) ?: return "Drone Wi-Fi: Bagli degil"
        if (!caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)) return "Drone Wi-Fi: Bagli degil"

        val ssid = currentSsid()
        if (ssid != null && ssid.contains("TELLO", ignoreCase = true)) {
            return "Drone Wi-Fi: Bagli ($ssid)"
        }
        return if (ssid != null) "Drone Wi-Fi: Farkli Ag ($ssid)" else "Drone Wi-Fi: Bagli"
    }

    @Suppress("DEPRECATION")
    private fun currentSsid(): String? {
        return try {
            val wm = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
            val raw = wm.connectionInfo?.ssid ?: return null
            val clean = raw.replace("\"", "").trim()
            if (clean.isBlank() || clean.equals("<unknown ssid>", ignoreCase = true)) null else clean
        } catch (_: Exception) {
            null
        }
    }

    private fun isGamepadConnected(): Boolean {
        val ids = InputDevice.getDeviceIds()
        for (id in ids) {
            if (isPhysicalGamepad(id)) return true
        }
        return false
    }

    private fun isPhysicalGamepad(deviceId: Int): Boolean {
        val dev = InputDevice.getDevice(deviceId) ?: return false
        if (dev.isVirtual) return false
        val src = dev.sources
        return (src and InputDevice.SOURCE_GAMEPAD) == InputDevice.SOURCE_GAMEPAD ||
            (src and InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK
    }

    private fun isGamepadKeyEvent(event: KeyEvent): Boolean {
        val device = event.device ?: return false
        if (device.isVirtual) return false
        val src = if (event.source != 0) event.source else device.sources
        return (src and InputDevice.SOURCE_GAMEPAD) == InputDevice.SOURCE_GAMEPAD ||
            (src and InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK
    }

    private fun hasTouchManualInput(): Boolean {
        return touchLr != 0 || touchFb != 0 || touchUd != 0 || touchYaw != 0
    }

    private fun postCommandText(message: String) {
        runOnUiThread { commandText.text = "Son komut: $message" }
    }

    private fun isContinuousGamepadKey(keyCode: Int): Boolean {
        return keyCode == KeyEvent.KEYCODE_BUTTON_L1 ||
            keyCode == KeyEvent.KEYCODE_BUTTON_R1 ||
            keyCode == KeyEvent.KEYCODE_BUTTON_L2 ||
            keyCode == KeyEvent.KEYCODE_BUTTON_R2
    }

    private fun toast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    private fun acquireWifiLock() {
        try {
            val wm = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
            if (wifiLock == null) {
                wifiLock = wm.createWifiLock(WifiManager.WIFI_MODE_FULL_HIGH_PERF, "DroneStation::WifiLock")
                wifiLock?.setReferenceCounted(false)
            }
            if (wifiLock?.isHeld == false) {
                wifiLock?.acquire()
            }
        } catch (_: Exception) {
        }
    }

    private fun releaseWifiLock() {
        try {
            if (wifiLock?.isHeld == true) {
                wifiLock?.release()
            }
        } catch (_: Exception) {
        }
    }

    private fun bindToWifiNetwork(forceRebind: Boolean = false) {
        try {
            val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
            if (forceRebind) {
                cm.bindProcessToNetwork(null)
                val existing = wifiBindCallback
                if (existing != null) {
                    try {
                        cm.unregisterNetworkCallback(existing)
                    } catch (_: Exception) {
                    }
                }
                wifiBindCallback = null
            }

            val active = cm.activeNetwork
            if (active != null) {
                val caps = cm.getNetworkCapabilities(active)
                if (caps?.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) == true) {
                    cm.bindProcessToNetwork(active)
                }
            }

            if (wifiBindCallback != null) return

            val req = NetworkRequest.Builder()
                .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
                .build()

            val callback = object : ConnectivityManager.NetworkCallback() {
                override fun onAvailable(network: Network) {
                    try {
                        cm.bindProcessToNetwork(network)
                    } catch (_: Exception) {
                    }
                }

                override fun onCapabilitiesChanged(network: Network, networkCapabilities: NetworkCapabilities) {
                    if (!networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)) return
                    try {
                        cm.bindProcessToNetwork(network)
                    } catch (_: Exception) {
                    }
                }
            }

            cm.requestNetwork(req, callback)
            wifiBindCallback = callback
        } catch (_: Exception) {
        }
    }

    private fun unbindNetwork() {
        try {
            val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
            cm.bindProcessToNetwork(null)
            val cb = wifiBindCallback
            if (cb != null) {
                try {
                    cm.unregisterNetworkCallback(cb)
                } catch (_: Exception) {
                }
            }
            wifiBindCallback = null
        } catch (_: Exception) {
        }
    }
}
