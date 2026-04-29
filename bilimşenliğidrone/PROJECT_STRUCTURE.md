# Project Structure

Onerilen dizin yapisi:

```text
bilimsenligidrone/
  README.md
  PROJECT_STRUCTURE.md
  docs/
  configs/
  data/
    raw/
    labels/
    splits/
  models/
  logs/
  scripts/
  src/
    app/
    perception/
    tracking/
    control/
    flight/
    acro/
    safety/
```

## Modul Sorumluluklari

1. `src/app`
   - Ana calistirici, pipeline baglantisi
2. `src/perception`
   - Person detection modeli ve inference wrapper
3. `src/tracking`
   - Target ID devamliligi, hedef secim politikasi
4. `src/control`
   - Follow controller (yaw/forward/altitude mantigi)
5. `src/flight`
   - Drone abstraction (sim/real command bridge)
6. `src/acro`
   - Akrobasi komut yurutucusu ve pre-check
7. `src/safety`
   - Fail-safe, geofence, manual override, watchdog

## Config Dosyalari (onerilen)

1. `configs/runtime.yaml`
2. `configs/model.yaml`
3. `configs/safety.yaml`
4. `configs/acro.yaml`

## Naming Notu

Klasor adinda Turkce karakter kullaniliyor.
Dokumanlarda okunabilirlik icin `bilimsenligidrone` yazimi kullanildi.
