# Training Plan

## Amac

Sadece `person` sinifinda hizli ve guvenilir detection modeli elde etmek.

## Model Secimi

1. Baslangic:
   - YOLOv8n (hiz odakli)
2. Alternatif:
   - YOLOv8s (daha yuksek dogruluk, daha yuksek maliyet)

## Egitim Konfig Onerisi

1. Input size:
   - 640 (ilk deneme)
   - 800 (zor sahneler icin deney)
2. Epoch:
   - 80-150 arasi
3. Batch:
   - GPU bellegine gore dinamik
4. Early stopping:
   - val metrigi iyilesmiyorsa durdur

## Deney Matrisi

1. Exp-A: YOLOv8n, imgsz=640
2. Exp-B: YOLOv8n, imgsz=800
3. Exp-C: YOLOv8s, imgsz=640
4. Exp-D: YOLOv8s, imgsz=800

Her deneyde kayit:

1. mAP50
2. mAP50-95
3. Precision / Recall
4. Inference FPS
5. Model boyutu

## Kabul Esikleri (Ilk Faz)

1. mAP50 >= 0.88
2. Precision >= 0.90
3. Recall >= 0.85
4. Gercek zaman FPS >= 25 (hedef cihazda)

## Checkpoint Politika

1. En iyi val mAP checkpoint sakla.
2. En iyi recall checkpoint de ayri sakla.
3. Son model ile en iyi modeli karistirma, ayri klasor tut.

## Hata Analizi

1. False positive kliplerini ayir.
2. False negative kliplerini ayir.
3. En zor sahneleri `hard_cases` listesine al.
4. Bir sonraki veri toplamada bu sahneleri onceliklendir.

## Versiyonlama

1. `v0.1`: Ilk calisir model
2. `v0.2`: Hard-case guclendirmesi
3. `v1.0`: Saha testine hazir kararli model
