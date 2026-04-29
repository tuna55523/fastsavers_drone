# Test and Validation Plan

## Test Katmanlari

1. Offline video test
2. Simulasyon testi
3. Kapali alan gercek ucus
4. Acik alan kontrollu ucus

## Senaryo Seti

1. Tek kisi duz yurus
2. Tek kisi ani yon degisimi
3. Coklu kisi sahnesi
4. Kismi gorunurluk / engel arkasi
5. Dusuk isik
6. Hedef kaybi ve geri bulma
7. Akrobasi oncesi ve sonrasi takip geri kazanimi

## Metrikler

Detection:

1. mAP50
2. Precision
3. Recall

Tracking:

1. ID switch sayisi
2. Track kayip orani
3. Reacquire suresi

Control:

1. Ortalama merkez hatasi (px)
2. Overshoot miktari
3. Stabilizasyon suresi

Runtime:

1. End-to-end latency
2. Ortalama FPS
3. CPU/GPU kullanim profili

## Kabul Kriteri (v1)

1. Stabil takip testlerinde basari >= %85
2. Hedef kayip geri bulma (kisa kayip) >= %80
3. Kritik fail-safe testlerinde %100 guvenli cikis

## Test Kayit Formati

Her kosu icin:

1. Test ID
2. Tarih/Saat
3. Ortam kosulu
4. Basarili/Basarisiz
5. Notlar
6. Ilgili log dosyasi yolu
