# Dataset Plan (Person Only)

## Hedef

Yeni model sadece insan tespitine odaklanacak.
Ana prensip: eski veri setinden bagimsiz, yeni dagilim ile toplama.

## Veri Toplama Senaryolari

1. Acik alan gunduz
2. Acik alan aksamustu
3. Kapali alan duz isik
4. Kapali alan dusuk isik
5. Tek kisi hizli hareket
6. Coklu kisi kalabalik sahne
7. Kismi gorunurluk (arkadan gecis, engel)
8. Uzak/orta/yakin mesafe

## Cekim Kurallari

1. Kamera acisi cesitli olacak (ustten, yatay, hafif capraz).
2. Arka plan tekrari minimum tutulacak.
3. Farkli kiyafet renkleri kullanilacak.
4. Hareketli video klipleri zorunlu olacak.

## Etiketleme Standardi

Siniflar:

1. `person`

Format:

1. YOLO formati (`class x_center y_center width height`)

Kurallar:

1. Tam gorunen kisi: tum govdeyi kapsa.
2. Kismi gorunen kisi: gorunen bolgeyi kapsa.
3. Cok kucuk veya ayirt edilemeyen siluet: etiketleme disi.
4. Maneken/fotograf gibi canli olmayan insan benzeri nesne: etiketleme disi.

## Veri Ayrimi (Split)

Onerilen oran:

1. Train: %70
2. Validation: %20
3. Test: %10

Kural:

1. Ayni videodan gelen frame'ler farkli splitlere dagitilmayacak.
2. Split sahne bazli yapilacak, frame bazli degil.

## Kalite Kontrol

1. Etiket kaymasi var mi?
2. Kutu boyutlari asiri dar/genis mi?
3. Dosya-etiket eslesmesi tam mi?
4. Bos etiketli frame oranlari beklenen duzeyde mi?

## Veri Miktari Baslangic Hedefi

1. En az 8-12 saat ham video
2. En az 12k-20k etiketli frame
3. Test setinde en az 1500 frame

## Artirim (Augmentation) Plani

1. Motion blur
2. Dusicik isik simulasyonu
3. Scale jitter
4. Kismi occlusion
5. Hafif noise

Not:

1. Horizontal flip sahneye uygun oldugunda acik olur.
2. Asiri artirim ile gercek dagilim bozulmayacak.
