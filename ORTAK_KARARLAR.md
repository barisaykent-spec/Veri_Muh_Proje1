# Ortak kararlar (herkes bireysel çalışırken bunlar sabit)

Beş kişi önce kendi başına çalışıp sonra tek defterde birleşecek. Sonuçların karşılaştırılabilmesi için aşağıdakiler **herkeste aynı** olmalı. Bir karar değişirse burada güncellenir ve herkese duyurulur.

| # | Karar | Değer | Gerekçe (kısa) |
|---|---|---|---|
| 1 | Veri | Inside Airbnb İstanbul, `listings.csv.gz`, anlık görüntü **2026-09-25** (27.232 × 90) | Dönem boyunca sabit |
| 2 | Kapsam | Yalnızca kısa dönem: `minimum_nights < 30` → 17.114 ilan | 100+ gecelik ilanların fiyatı aylık teklif; iki pazar karışıyor |
| 3 | Hedefsiz satırlar | `price` boş olanlar dışlanır → **16.558 ilan, 3.434 ev sahibi** | Hedef yok; eksiklik MAR yönünde (takvim kapalı), 1.3'te raporlanır |
| 4 | Hedef | Gecelik fiyatın üçte birlik dilimleri: **ucuz < 3.455,50 ≤ orta < 5.877,67 ≤ pahalı** (TL, çıkarım) | Dengeli 3 sınıf; uç değerlerden etkilenmez |
| 5 | Bölme | `StratifiedGroupKFold`, `groups = host_id` | Ev sahibi içi log-fiyat std 0,31, genel 0,85: rastgele bölme skoru şişirir |
| 6 | Tohum | `SEED = 42` | Tekrarlanabilirlik |
| 7 | Modele asla girmeyecek sütunlar | `price`, `price_quote_*` (5 sütun), `estimated_revenue_l365d` | Hedefi doğrudan ya da dolaylı veriyor (sızıntı) |
| 8 | Kimlik / URL sütunları | `id`, `listing_url`, `scrape_id`, `host_id`*, `host_url`, `host_profile_id`, `host_profile_url`, `picture_url`, `host_picture_url` | Öznitelik değil (*`host_id` yalnızca bölme grubu) |
| 9 | %100 boş sütunlar | 14 sütun (host_since, host_response_*, bathrooms, instant_bookable, …) | Bilgi yok; künyede listelenir |

## Bireysel çalışma düzeni
- Her üye kendi kopyasını `calisma/<ad>/BIL545_Proje1.ipynb` yolunda tutar ve **kendi hesabından** commit'ler (katkı puanının %25'i commit geçmişi).
- Birleştirme sırasında en iyi kısımlar `proje1/BIL545_Proje1.ipynb` defterine alınır; kimin hangi kısmının alındığı katkı beyanına yazılır.
