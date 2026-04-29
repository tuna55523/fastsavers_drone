# Tracking + Control Plan

## Sistem Akisi

1. Kamera frame al
2. Person detection calistir
3. Tracking ile hedef ID koru
4. Hedefi sec ve follow kontrolune gir
5. RC komut uret ve ucusa uygula

## Moduller

1. `perception`
   - Detection modeli
2. `tracking`
   - ByteTrack / DeepSORT
3. `control`
   - Follow controller (PID/MPC)
4. `state_machine`
   - SEARCH/LOCK/FOLLOW/LOST/REACQUIRE
5. `flight`
   - Drone arayuzu (sim/real)

## Hedef Secim Politikasi

1. Tek kisi varsa direkt kilit.
2. Cok kisi varsa:
   - Operator secimi varsa operator onceligi
   - Yoksa goruntu merkezine en yakin hedef
3. Hedef degisimi histerezis ile yumusatilir.

## Durum Makinasi

1. `SEARCH`
   - Kisi yoksa yaw sweep ile ara
2. `LOCK`
   - Yeni hedef bulundu, 0.5-1.0 sn stabil dogrula
3. `FOLLOW`
   - Hedef merkezde tutulur, mesafe korunur
4. `LOST`
   - Kisa sure algi yok, hover + son yone bak
5. `REACQUIRE`
   - Zaman sinirli yeniden tarama, bulursa `FOLLOW`, bulamazsa `SEARCH`

## Kontrol Kurallari

Yatay merkezleme:

1. `err_x = target_cx - frame_center_x`
2. `yaw_cmd = Kp_yaw * err_x`
3. Deadzone icinde yaw sifirlanir.

Mesafe koruma:

1. Bounding box alanina gore ileri/geri komut
2. Hedef alan bandi icindeyse ileri/geri sifir

Yumusatma:

1. RC slew limit
2. Komut timeout olursa hover

## Performans Hedefi

1. End-to-end latency <= 120 ms (hedef)
2. Takipte frame drop durumunda kontrol stabil kalmali
3. Hedef kayipta guvenli gecis garanti olmali
