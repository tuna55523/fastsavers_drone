# Master Roadmap

Bu dosya, yeni person-follow + acro projesinin ucundan ucuna yol haritasidir.

## Faz-1: Hedef ve sinirlar (1 gun)

Teslimler:

1. Operasyon senaryosu dokumani
2. Basari metrikleri tablosu
3. Guvenlik kurallari listesi

Kararlar:

1. Takip modu tek hedef mi, hedef secimli mi?
2. Kamera merkezleme toleransi kac piksel?
3. Akrobasi tetikleme manuel mi, yarim-otonom mu?

## Faz-2: Dataset stratejisi (3-7 gun)

Teslimler:

1. Cekim protokolu (mekan, isik, mesafe, aci)
2. Etiketleme kilavuzu (`class = person`)
3. `train/val/test` split raporu
4. Data kalite raporu (bulaniklik, occlusion, gece)

Hedef:

- Minimum 8 farkli sahne
- Her sahnede farkli kiyafet/arka plan
- Hareketli klipler (tracking icin)

## Faz-3: Egitim ve benchmark (3-10 gun)

Teslimler:

1. Ilk detection modeli
2. Egitim konfigurasyonu
3. Val/Test metrik raporu
4. FPS + latency olcumleri

Kabul kriteri (baslangic):

1. mAP50 >= 0.88 (person-only veri setinde)
2. Precision >= 0.90
3. GPU hedef FPS >= 25 (giris seviyesi kartta)

## Faz-4: Takip kontrolu (4-8 gun)

Teslimler:

1. Durum makinasi:
   - `SEARCH`
   - `LOCK`
   - `FOLLOW`
   - `LOST`
   - `REACQUIRE`
2. Kontrol katmani (PID/MPC secimi)
3. Takip stabilizasyon ayarlari

Kabul kriteri:

1. Hedef merkez hatasi ortalama dusuk
2. Kisa hedef kayiplarinda otomatik geri kilit
3. Ani harekette oscillation sinirli

## Faz-5: Akrobasi modulu (7-14 gun)

Teslimler:

1. Manevra API:
   - `acro_roll_left`
   - `acro_roll_right`
   - `acro_front_flip`
   - `acro_back_flip`
2. Emniyet kosul kontrolu
3. Simulasyon dogrulama raporu

Kural:

- Takip ve akrobasi bagimsiz moduller olacak.
- Akrobasi, yalnizca emniyet kapisi aciksa calisacak.

## Faz-6: Entegrasyon + saha test (5-10 gun)

Teslimler:

1. Uctan uca pipeline calisir hali
2. Test senaryolari sonuclar tablosu
3. Ucus loglari + hata analiz notlari
4. Sonraki iterasyon backlog

## Toplam Is Programi (hedef)

1. En hizli akista: 3-4 hafta
2. Daha guvenli iteratif akista: 5-7 hafta

## Kritik Riskler ve Azaltma

1. Veri dagilimi zayifligi:
   - Cok cesitli sahne toplama ve duzenli yeniden veri ekleme
2. Takipte ani kayip:
   - Reacquire paterni + zaman asimli hover
3. Akrobasi emniyeti:
   - Once sim, sonra dusuk irtifa, sonra kontrollu acik alan
