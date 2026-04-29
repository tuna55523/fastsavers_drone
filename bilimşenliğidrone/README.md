# Bilim Senligi Drone - Person Follow + Acro

Bu klasor, ana projeden bagimsiz yeni bir calisma alanidir.
Amac: sadece insan odakli tespit + takip sistemi ve guvenli akrobatik hareket modulu gelistirmek.

## Proje Hedefi

1. Sadece `person` sinifi ile calisan yeni bir perception modeli egitmek.
2. Hedef kisiyi goruntu merkezinde tutup otonom takip etmek.
3. Hedef kaybi durumunda guvenli sekilde yeniden bulma davranisi uygulamak.
4. Ayri bir manevra motoru ile roll/flip hareketlerini guvenli kosullarda calistirmak.

## Kapsam

- Dahil:
  - Yeni dataset toplama + etiketleme
  - Detection + tracking + follow control
  - Search/lock/follow/lost/reacquire durum makinasi
  - Akrobasi modulu (simulasyon once)
  - Guvenlik, loglama, test ve benchmark
- Haric:
  - Su ustu bogulma risk analizi
  - Multi-class nesne analizi
  - Uzun mesafe BVLOS operasyonu

## Fazlar

1. Faz-1: Hedef ve operasyon politikasi netlestirme
2. Faz-2: Dataset toplama + etiketleme + split
3. Faz-3: Model egitimi ve dogrulama
4. Faz-4: Takip kontrolu ve durum makinasi
5. Faz-5: Akrobasi motoru ve emniyet kilitleri
6. Faz-6: Entegrasyon, saha testi, iterasyon

Detaylar icin `docs/` altindaki plan dosyalarina bak.

## Hedef Teknoloji (onerilen)

- Detection: YOLOv8n veya YOLOv8s (person only)
- Tracking: ByteTrack (ilk tercih), DeepSORT (alternatif)
- Ucus kontrolu: MAVSDK / PX4 arayuzu (sim + gercek)
- Simulasyon: PX4 + Gazebo veya AirSim

## Baslangic Kontrol Listesi

1. `docs/00_master_roadmap.md` oku
2. `docs/06_sprint1_daily_plan.md` uzerinden Sprint-1 baslat
3. `PROJECT_STRUCTURE.md` ye gore klasorleri aktif kullan
4. Ilk hedef: stabil person follow (akrobasi sonraki adim)
