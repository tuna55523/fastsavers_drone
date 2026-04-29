# Safety Policy

## Ana Kural

Takip veya akrobasi modu ne olursa olsun, guvenlik her zaman en ust katmandir.

## Kill Switch ve Override

1. Manuel override tusu her zaman aktif olacak.
2. Tek komutla `hover` gecisi olacak.
3. Kritik durumda `land` oncelikli fail-safe olacak.

## Operasyon Limitleri

1. Min batarya esigi asagisinda otonom hareket kapali.
2. Max hiz ve max yaw oran limitli.
3. Geofence disina cikis engellenecek.
4. Stream stale suresinde auto mode durdurulacak.

## Hedef Kayip Politikasi

1. Kisa kayip: hover + son yone bak
2. Orta kayip: kontrollu sweep ile reacquire
3. Uzun kayip: guvenli `SEARCH` davranisina don

## Akrobasi Guvenlik Politikasi

1. Akrobasi yalnizca operator onayi ile.
2. Takip durumu stabil degilse akrobasi yasak.
3. Manevra sonrasi zorunlu stabilizasyon penceresi.

## Kayit ve Izlenebilirlik

Her kritik olay loglanacak:

1. Mode degisimleri
2. Target lock/loss olaylari
3. Fail-safe tetiklemeleri
4. Akrobasi pre-check ve sonuc durumu

## Testten Ucusa Gecis Kurali

1. Simulasyonda basarisiz senaryo kalmayacak.
2. Kapali alan testleri tamamlanmadan acik alan yok.
3. Acik alanda ilk ucuslar dusuk irtifa + dusuk hiz.
