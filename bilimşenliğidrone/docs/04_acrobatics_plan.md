# Acrobatics Plan

## Temel Prensip

Akrobasi, takipten bagimsiz bir moduldur.
Takip pipeline'i kesilmeden, kontrollu sekilde akrobasi komutu verilir.

## Manevra Listesi (v1)

1. `acro_roll_left`
2. `acro_roll_right`
3. `acro_front_flip`
4. `acro_back_flip`

## Emniyet Kapilari (Gate Checks)

Akrobasi calismadan once tumu saglanmali:

1. Batarya min esik ustu
2. GPS/IMU/vision saglik durumu iyi
3. Minimum irtifa uygun
4. Cevirme alani bos (catisma riski dusuk)
5. Manuel override aktif ve hazir
6. Operator onayi var (v1)

## Uygulama Stratejisi

1. Once simulasyonda hareket profili test edilir.
2. Sonra kapali alanda dusuk riskli deneme.
3. Sonra acik alanda kontrollu deneme.
4. Basarisiz manevrada otomatik stabilize + hover.

## Manevra Yurutme Adimlari

1. Pre-check
2. Kisa hover stabilizasyonu
3. Manevra komutu
4. Post-check ve attitude normalize
5. Follow moduna geri donus

## Basari Olcutleri

1. Manevra tamamlanma orani
2. Manevra sonrasi stabilizasyon suresi
3. Manevra sonrasi hedefe geri kilit suresi

## Yasak Kosullar

1. Dusuk batarya
2. Hedef yeni kaybolmusken akrobasi
3. Sensor sapmasi / stream stale
4. Coklu emniyet uyarisi aktifken
