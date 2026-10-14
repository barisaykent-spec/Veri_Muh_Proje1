# BIL545 · Veri Madenciliğinde İleri Konular · 2026–2027 Güz

**Konu:** 9.19 İstanbul'da Airbnb ilanları  
**Soru (tek cümle):** _…_  
**Hedef:** Fiyat dilimi (`price` sütunundan türetilen sınıflar)  
**Grup tipi:** 5 kişi (temel kapsam + Ek paket 1 + Ek paket 2)

## Grup
| | Ad Soyad | Öğrenci no | GitHub |
|---|---|---|---|
| Üye 1 | | | |
| Üye 2 | | | |
| Üye 3 | | | |
| Üye 4 | | | |
| Üye 5 | | | |

**Grup adı:** …  
**İletişim sorumlusu:** …

## Veri kümesi ve lisans
| Alan | Bilgi |
|---|---|
| Kaynak | [Inside Airbnb](http://insideairbnb.com/get-the-data/): İstanbul, Marmara, Türkiye |
| Dosya | `listings.csv.gz` (detaylı ilan dosyası, ≈75 sütun) |
| Anlık görüntü tarihi | `2026-09-25` (dönem boyunca sabit) |
| Lisans | CC BY 4.0 |

## Verinin indirilmesi
Veri depoya konmaz. Defterin "Veriyi yükleme" hücresi dosyayı şu adresten indirir:

```
https://data.insideairbnb.com/turkey/marmara/istanbul/<SNAPSHOT>/data/listings.csv.gz
```

Inside Airbnb eski anlık görüntüleri bir süre sonra kaldırabilir. Bu durumda grubun sakladığı kopyayı Colab'a `listings.csv.gz` adıyla yükleyin.

## Defterlerin çalıştırılması
1. `proje1/BIL545_Proje1.ipynb` dosyasını Colab'da açın.
2. **Çalışma Zamanı → Tümünü çalıştır**. Sürümler ilk hücrede yazdırılır, tohum `SEED = 42`.
3. Proje 2, Proje 1'in çıktılarını (`p1_temiz_veri.csv`, `p1_hat.joblib`) kullanır.

| Klasör | İçerik | Teslim |
|---|---|---|
| `proje1/` | Proje 1 defteri (vize) | 8. hafta |
| `proje2/` | Proje 2 defteri (final) | 14. hafta |

## Sonuç özeti
_Proje 1 tesliminde doldurulacak._

## Katkı beyanı
_Ayrıntısı defterlerin sonundaki "Katkı beyanı" bölümündedir._
