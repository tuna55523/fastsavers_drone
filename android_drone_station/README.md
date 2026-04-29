# Android Drone Station

Tello + PS4 gamepad kontrol uygulamasi.

## Bu surumde hazir olanlar
- Tello UDP baglantisi (`command`, `takeoff`, `land`, `flip`, `rc`)
- PS4 gamepad map (istenen tuslara gore)
- Periyodik `rc` komutu (20 Hz)
- Batarya sorgusu ve durum gosterimi
- Canli video stream (`udp://:11111`) decode ve ekranda gosterim
- L3 ile snapshot (JPG)
- R3 ile ham video kaydi (`.h264`)
- Takip durumu ac/kapa + hedef degistir
- Takip acikken telemetry tabanli assist (yaw/up-down/mesafe)
- Opsiyonel cihaz ici detector (TFLite model assets'e konursa aktif)

## Klasor yapisi
- `app/src/main/java/com/bilimsenligi/dronestation/drone/TelloClient.kt`
- `app/src/main/java/com/bilimsenligi/dronestation/drone/TelloVideoManager.kt`
- `app/src/main/java/com/bilimsenligi/dronestation/control/GamepadMapper.kt`
- `app/src/main/java/com/bilimsenligi/dronestation/tracking/TrackingManager.kt`
- `app/src/main/java/com/bilimsenligi/dronestation/tracking/TrackingTelemetryReceiver.kt`
- `app/src/main/java/com/bilimsenligi/dronestation/tracking/TrackingAssistMixer.kt`
- `app/src/main/java/com/bilimsenligi/dronestation/MainActivity.kt`

## Android Studio ile calistirma
1. Android Studio ac.
2. `android_drone_station` klasorunu proje olarak ac.
3. Gradle Sync yap.
4. Telefondan Tello Wi-Fi agina baglan.
5. Uygulamayi telefona yukleyip ac.
6. "Tello Baglan" butonuna bas.

## PS4 mapping (uygulanan)
- R2: Ileri
- L2: Geri
- R1: Saga don (yaw)
- L1: Sola don (yaw)
- D-pad up/down/left/right: On/arka/sol/sag takla
- Left stick X: Ileri-geri hareket sirasinda sag-sol yonlendirme
- Right stick Y: Yukari-asagi
- L3: Snapshot (JPG)
- R3: Ham video kaydi baslat/durdur (`.h264`)
- Circle: Takip baslat
- Square: Takip durdur
- Triangle: Hedef degistir
- X (Cross): Kalkis/Inis toggle

## Takip telemetry formati (opsiyonel)
Takip assist icin uygulama UDP `5005` portunda JSON bekler.

Ornek:
```json
{"tx":0.12,"ty":-0.08,"size":0.18,"conf":0.87,"id":3,"ts":1714300000000}
```
- `tx`: hedefin yatay offseti (`-1..1`, + saga)
- `ty`: hedefin dikey offseti (`-1..1`, + asagi)
- `size`: bbox alan orani (`0..1`)
- `conf`: guven (`0..1`)
- `id`: hedef id
- `ts`: zaman damgasi

## Notlar
- Android'de PS4 tus kodlari `A/B/X/Y` seklinde map oldugu icin,
  `Cross=A`, `Circle=B`, `Square=X`, `Triangle=Y` olarak ele alinir.
- R3 kaydi ham `.h264` formatinda saklar (hizli MVP icin).
- `app/src/main/assets/person_tracking_best_v8n_768.tflite` dosyasi konursa
  uygulama dis telemetry yokken cihaz ici detection ile takip assist uretebilir.
## Model export (Android)

Telefon uzerinde model calistirmak icin once `.pt` dosyasini `.tflite` formatina export edin.

Ornek komutlar:

```powershell
# FP16/FP32 benzeri (hizli deneme)
python bilimşenliğidrone\scripts\export_android_tflite.py --imgsz 640

# INT8 quantized (telefon icin onerilen)
python bilimşenliğidrone\scripts\export_android_tflite.py --imgsz 640 --int8
```

Cikti dosyalari Ultralytics export klasorunde olusur. Sonraki adimda `android_drone_station/app/src/main/assets/` altina koyup
inference modulu baglanir.

## Tracking telemetry hizli test (model olmadan)
Telefon IP'sine sahte hedef offset gonderip takip assist davranisini test edebilirsiniz.

```powershell
python bilimşenliğidrone\scripts\tracking_udp_telemetry_test.py --host <TELEFON_IP> --port 5005 --hz 10
```
