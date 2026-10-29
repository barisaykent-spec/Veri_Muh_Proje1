# Sütun sözlüğü ve işlem kaydı (Inside Airbnb İstanbul, 2026-09-25)

90 sütunun her biri: ne işe yaradığı, temel istatistikleri, projede silinip silinmediği ve hangi aksiyonun alındığı.

**Durum özeti:** Öznitelik adayı: 52 · Silindi: 35 · Tutuldu (model dışı): 2 · HEDEF: 1

| No | Sütun | Ne işe yarar | Tip | Eksik % | Farklı değer | Durum | Aksiyon / gerekçe |
|---|---|---|---|---|---|---|---|
| 1 | `id` | İlanın Airbnb numarası | int64 | 0.0 | 27232 | Tutuldu (model dışı) | ANAHTAR rolü: başka tablolarla (reviews.csv) bağlantı. Modele girmez. Weka'da W2'de silindi |
| 2 | `listing_url` | İlanın Airbnb sayfa adresi | metin | 0.0 | 27232 | Silindi | Adım 2 (B: kimlik/URL). Weka için W1'de çıkarıldı (çökertiyordu) |
| 3 | `scrape_id` | Kazıma işleminin numarası (tek değer: 20260925130056) | int64 | 0.0 | 1 | Silindi | Adım 2 (A: sabit). Weka: RemoveUseless |
| 4 | `last_scraped` | İlanın kazındığı gün (28–29 Eylül) | metin | 0.0 | 2 | Silindi | Adım 3 (toplama süreci izi). Weka: W2 Remove |
| 5 | `source` | Önceki kazımadan mı geldi bilgisi | metin | 60.8 | 1 | Silindi | Adım 3 (süreç izi). Python'da 2 değerli (dolu/boş) sayıldı; Weka RemoveUseless tek değerli sayıp sildi |
| 6 | `name` | İlan başlığı (serbest metin) | metin | 0.0 | 25571 | Silindi | Adım 3 (kuralın kaçırdığı kısa metin). Weka için W1'de çıkarıldı (kesme işareti) |
| 7 | `description` | İlan açıklaması (uzun metin) | metin | 0.6 | 17410 | Silindi | Adım 2 (D: ham metin). İleride metin özniteliği türetilebilir |
| 8 | `neighborhood_overview` | Mahalle tanıtımı | float64 | 100.0 | 0 | Silindi | Adım 2 (A: %100 boş) |
| 9 | `picture_url` | İlan fotoğrafı adresi | metin | 0.0 | 25806 | Silindi | Adım 2 (B: URL) |
| 10 | `host_id` | Ev sahibi numarası | int64 | 0.0 | 9967 | Tutuldu (model dışı) | GRUP rolü: StratifiedGroupKFold(groups=host_id) ile bölme. Modele girmez. Weka'da W2'de silindi (Weka'da gruba göre bölme yok) |
| 11 | `host_url` | Ev sahibi sayfa adresi | metin | 0.0 | 9967 | Silindi | Adım 2 (B: URL) |
| 12 | `host_profile_id` | Ev sahibi profil numarası | int64 | 0.0 | 9967 | Silindi | Adım 2 (B: _id). Weka: W2 Remove |
| 13 | `host_profile_url` | Ev sahibi profil adresi | metin | 0.0 | 9967 | Silindi | Adım 2 (B: URL) |
| 14 | `host_name` | Ev sahibinin adı (kişisel veri) | metin | 0.0 | 4330 | Silindi | Adım 3 (serbest metin, kişisel veri). Weka için W1'de çıkarıldı |
| 15 | `host_since` | Ev sahibinin kayıt tarihi | float64 | 100.0 | 0 | Silindi | Adım 2 (A: %100 boş). Yerine hosts_time_as_* var |
| 16 | `hosts_time_as_user_years` | Ev sahibinin Airbnb üyeliği (yıl) | int64 | 0.0 | 17 | Öznitelik adayı | Aksiyon yok |
| 17 | `hosts_time_as_user_months` | Üyelik süresinin ay kısmı (0–11) | int64 | 0.0 | 12 | Öznitelik adayı | Aksiyon yok |
| 18 | `hosts_time_as_host_years` | Ev sahipliği süresi (yıl) | int64 | 0.0 | 16 | Öznitelik adayı | Aksiyon yok; 1.3-d modelinde kullanıldı |
| 19 | `hosts_time_as_host_months` | Ev sahipliği süresinin ay kısmı (0–11) | int64 | 0.0 | 12 | Öznitelik adayı | Aksiyon yok |
| 20 | `host_location` | Ev sahibinin yaşadığı yer | metin | 35.9 | 351 | Öznitelik adayı | 1.3-c: %35,9 eksik, MAR. Yazım farkları var (İstanbul/Istanbul, Turkey/Türkiye) → birleştirilmeli (bekliyor) |
| 21 | `host_about` | Ev sahibinin kendini tanıtması (uzun metin) | metin | 58.6 | 3127 | Silindi | Adım 2 (D: ham metin). 1.3-b: eksikliği deneyimle ilişkili |
| 22 | `host_response_time` | Yanıt süresi (ordinal) | float64 | 100.0 | 0 | Silindi | Adım 2 (A: %100 boş) |
| 23 | `host_response_rate` | Yanıt oranı | float64 | 100.0 | 0 | Silindi | Adım 2 (A: %100 boş) |
| 24 | `host_acceptance_rate` | Rezervasyon kabul oranı | float64 | 100.0 | 0 | Silindi | Adım 2 (A: %100 boş) |
| 25 | `host_is_superhost` | Süper ev sahibi mi (t/f, asimetrik ikili) | metin | 0.0 | 2 | Öznitelik adayı | t/f → 1/0 dönüşümü bekliyor |
| 26 | `host_thumbnail_url` | Ev sahibi küçük fotoğrafı | float64 | 100.0 | 0 | Silindi | Adım 2 (A: %100 boş) |
| 27 | `host_picture_url` | Ev sahibi fotoğrafı | metin | 0.0 | 8790 | Silindi | Adım 2 (B: URL) |
| 28 | `host_neighbourhood` | Ev sahibinin mahallesi | float64 | 100.0 | 0 | Silindi | Adım 2 (A: %100 boş) |
| 29 | `host_listings_count` | Ev sahibinin toplam ilan sayısı (Airbnb'nin verdiği) | int64 | 0.0 | 83 | Öznitelik adayı | Aksiyon yok; 1.3-d modelinde kullanıldı |
| 30 | `host_total_listings_count` | Ev sahibinin tüm ilan sayısı | float64 | 100.0 | 0 | Silindi | Adım 2 (A: %100 boş) |
| 31 | `host_verifications` | Doğrulama yöntemleri (e-posta, telefon…) | float64 | 100.0 | 0 | Silindi | Adım 2 (A: %100 boş) |
| 32 | `host_has_profile_pic` | Profil fotoğrafı var mı (t/f) | metin | 0.0 | 2 | Öznitelik adayı | t/f → 1/0 dönüşümü bekliyor |
| 33 | `host_identity_verified` | Kimlik doğrulanmış mı (t/f) | metin | 0.0 | 2 | Öznitelik adayı | t/f → 1/0 dönüşümü bekliyor |
| 34 | `neighbourhood` | Ev sahibinin yazdığı semt | float64 | 100.0 | 0 | Silindi | Adım 2 (A: %100 boş) |
| 35 | `neighbourhood_cleansed` | İlçe (39 ilçe; Inside Airbnb'nin konumdan belirlediği) | metin | 0.0 | 39 | Öznitelik adayı | Kategorik; kodlama bekliyor (bire-bir uygun). Proje 2 alt grup analizi için önemli |
| 36 | `neighbourhood_group_cleansed` | İlçe grubu | float64 | 100.0 | 0 | Silindi | Adım 2 (A: %100 boş) |
| 37 | `latitude` | Enlem | float64 | 0.0 | 16932 | Öznitelik adayı | Aksiyon yok; 1.3-d modelinde kullanıldı |
| 38 | `longitude` | Boylam | float64 | 0.0 | 17767 | Öznitelik adayı | Aksiyon yok; 1.3-d modelinde kullanıldı |
| 39 | `property_type` | Mülk tipi (66 kategori model verisinde) | metin | 0.0 | 88 | Öznitelik adayı | 1.4: bire-bir / nadir birleştirme / hedef kodlama denendi; katkı sınırlı |
| 40 | `room_type` | Oda tipi (4 kategori: tüm ev, özel oda, otel, paylaşımlı) | metin | 0.0 | 4 | Öznitelik adayı | Kategorik; bire-bir kodlama bekliyor. Fiyatla güçlü ilişkili |
| 41 | `accommodates` | Kişi kapasitesi | int64 | 0.0 | 16 | Öznitelik adayı | Aksiyon yok; tek başına en güçlü öznitelik |
| 42 | `bathrooms` | Banyo sayısı (sayısal) | float64 | 100.0 | 0 | Silindi | Adım 2 (A: %100 boş). Yerine bathrooms_text var |
| 43 | `bathrooms_text` | Banyo bilgisi metin olarak ("1.5 shared baths") | metin | 0.4 | 59 | Öznitelik adayı | Metin → sayı (banyo_sayisi) + tip (özel/paylaşımlı) dönüşümü bekliyor |
| 44 | `bedrooms` | Yatak odası sayısı | float64 | 0.1 | 30 | Öznitelik adayı | 1.3-a: 0 değeri geçerli (stüdyo) olarak bırakıldı |
| 45 | `beds` | Yatak sayısı | float64 | 1.5 | 35 | Öznitelik adayı | %0,9 eksik; hatta medyanla dolduruluyor |
| 46 | `amenities` | Olanaklar listesi (["Wifi", "Klima" …], çok değerli) | metin | 0.0 | 23115 | Silindi | Adım 2 (D: liste). Weka'yı çökertiyordu. Olanak sayısı / sık olanaklar türetilebilir (bekliyor); Bölüm 3 birliktelik için aday |
| 47 | `price` | Gecelik teklif fiyatı ("$" işaretli ama TL) | metin | 9.8 | 13011 | HEDEF | Metin → sayı. 3 dilime çevrildi (fiyat_dilimi). Fiyatı boş satırlar dışlandı. Öznitelik olarak ASLA girmez |
| 48 | `price_quote_checkin_date` | Fiyat teklifinin giriş tarihi | metin | 6.2 | 248 | Silindi | Adım 3 (SIZINTI: teklifin parçası). Weka: W2 |
| 49 | `price_quote_checkout_date` | Fiyat teklifinin çıkış tarihi | metin | 6.2 | 327 | Silindi | Adım 3 (SIZINTI). Weka: W2 |
| 50 | `price_quote_total_price` | Teklifin toplam tutarı (fiyat × gece) | float64 | 9.8 | 12705 | Silindi | Adım 3 (SIZINTI). Tek başına %73 doğruluk |
| 51 | `price_quote_price_per_night` | Teklifin gecelik tutarı (= price) | float64 | 9.8 | 13011 | Silindi | Adım 3 (SIZINTI: hedefin birebir kopyası) |
| 52 | `price_quote_raw` | Teklifin ham JSON kaydı | metin | 6.2 | 20239 | Silindi | Adım 2 (D: ham metin) + sızıntı |
| 53 | `minimum_nights` | En az konaklama (gece) | int64 | 0.0 | 85 | Öznitelik adayı | KAPSAM filtresi: < 30 (ORTAK_KARARLAR). Ayrıca öznitelik |
| 54 | `maximum_nights` | En çok konaklama (gece) | int64 | 0.0 | 141 | Öznitelik adayı | 1.3-a: %87'si platform varsayılanı (365/1125); bilgi değeri düşük, not edildi |
| 55 | `minimum_minimum_nights` | Takvimdeki en düşük min. konaklama | int64 | 0.0 | 85 | Öznitelik adayı | minimum_nights ile neredeyse aynı (gereksiz tekrar olabilir) |
| 56 | `maximum_minimum_nights` | Takvimdeki en yüksek min. konaklama | int64 | 0.0 | 87 | Öznitelik adayı | Aksiyon yok |
| 57 | `minimum_maximum_nights` | Takvimdeki en düşük max. konaklama | int64 | 0.0 | 141 | Öznitelik adayı | maximum_nights ile neredeyse aynı |
| 58 | `maximum_maximum_nights` | Takvimdeki en yüksek max. konaklama | int64 | 0.0 | 141 | Öznitelik adayı | maximum_nights ile neredeyse aynı |
| 59 | `minimum_nights_avg_ntm` | Önümüzdeki 12 ay ort. min. konaklama | float64 | 0.0 | 336 | Öznitelik adayı | Aksiyon yok |
| 60 | `maximum_nights_avg_ntm` | Önümüzdeki 12 ay ort. max. konaklama | float64 | 0.0 | 449 | Öznitelik adayı | Aksiyon yok |
| 61 | `calendar_updated` | Takvim güncelleme bilgisi | float64 | 100.0 | 0 | Silindi | Adım 2 (A: %100 boş) |
| 62 | `has_availability` | Müsaitlik var mı (t / boş) | metin | 1.7 | 1 | Öznitelik adayı | 1.3-a: boş → "f" (ters yönlü kodlanmış eksik). Model verisinde neredeyse hep "t" → bilgi değeri düşük |
| 63 | `availability_30` | Önümüzdeki 30 günde müsait gün sayısı | int64 | 0.0 | 31 | Öznitelik adayı | Aksiyon yok |
| 64 | `availability_60` | Önümüzdeki 60 günde müsait gün | int64 | 0.0 | 61 | Öznitelik adayı | Aksiyon yok |
| 65 | `availability_90` | Önümüzdeki 90 günde müsait gün | int64 | 0.0 | 91 | Öznitelik adayı | Aksiyon yok |
| 66 | `availability_365` | Önümüzdeki 365 günde müsait gün | int64 | 0.0 | 366 | Öznitelik adayı | 1.3-b: 0 ise fiyat %99,8 eksik |
| 67 | `calendar_last_scraped` | Takvimin okunduğu gün | metin | 0.0 | 2 | Silindi | Adım 3 (süreç izi). Weka: W2 |
| 68 | `number_of_reviews` | Toplam yorum sayısı | int64 | 0.0 | 387 | Öznitelik adayı | 1.3-a: 0 geçerli (yeni ilan). Puanların eksik olmasının nedeni |
| 69 | `number_of_reviews_ltm` | Son 12 aydaki yorum sayısı | int64 | 0.0 | 132 | Öznitelik adayı | Aksiyon yok |
| 70 | `number_of_reviews_l30d` | Son 30 gündeki yorum sayısı | int64 | 0.0 | 24 | Öznitelik adayı | Aksiyon yok |
| 71 | `availability_eoy` | Yıl sonuna kadar müsait gün | int64 | 0.0 | 96 | Öznitelik adayı | Aksiyon yok |
| 72 | `number_of_reviews_ly` | Geçen yılki yorum sayısı | int64 | 0.0 | 118 | Öznitelik adayı | Aksiyon yok |
| 73 | `estimated_occupancy_l365d` | Tahmini doluluk (gün, son 12 ay; yorumlardan türetilmiş) | int64 | 0.0 | 88 | Öznitelik adayı | Aksiyon yok. Dikkat: revenue ile birleşince fiyatı ele veriyor; revenue silindiği için tek başına güvenli |
| 74 | `estimated_revenue_l365d` | Tahmini gelir (fiyat × doluluk) | float64 | 9.8 | 9482 | Silindi | Adım 3 (SIZINTI: gelir/doluluk = fiyat, %99,9 doğruluk) |
| 75 | `first_review` | İlk yorum tarihi | metin | 35.5 | 3240 | Öznitelik adayı | %35,5 eksik (yorumsuz ilanlar). Tarih → "ilan yaşı" gibi sayıya dönüşüm bekliyor |
| 76 | `last_review` | Son yorum tarihi | metin | 35.5 | 1875 | Öznitelik adayı | Aynı; tarih → "son yorumdan bu yana gün" dönüşümü bekliyor |
| 77 | `review_scores_rating` | Genel puan (1–5) | float64 | 35.5 | 172 | Öznitelik adayı | 1.3-a: <1 olan 1 değer NaN yapıldı. 1.3-c: MAR (yorumsuz = puansız). Medyan + add_indicator |
| 78 | `review_scores_accuracy` | Doğruluk puanı | float64 | 35.5 | 166 | Öznitelik adayı | 1.3-c: MAR; medyan + add_indicator |
| 79 | `review_scores_cleanliness` | Temizlik puanı | float64 | 35.5 | 187 | Öznitelik adayı | Aynı |
| 80 | `review_scores_checkin` | Giriş puanı | float64 | 35.5 | 136 | Öznitelik adayı | Aynı |
| 81 | `review_scores_communication` | İletişim puanı | float64 | 35.5 | 139 | Öznitelik adayı | Aynı |
| 82 | `review_scores_location` | Konum puanı | float64 | 35.5 | 159 | Öznitelik adayı | Aynı |
| 83 | `review_scores_value` | Fiyat/değer puanı | float64 | 35.5 | 173 | Öznitelik adayı | Aynı |
| 84 | `license` | Kiralama izin numarası (7464 s. Kanun) | metin | 35.0 | 5080 | Öznitelik adayı | 1.3-a: sahte "12-3456" → NaN. 1.3-b: 100+ gecede %95 eksik. 1.4: hedef kodlama sızıntısı burada gösterildi (0,59 → 0,81) |
| 85 | `instant_bookable` | Anında rezervasyon (t/f) | float64 | 100.0 | 0 | Silindi | Adım 2 (A: %100 boş) |
| 86 | `calculated_host_listings_count` | Ev sahibinin bu şehirdeki ilan sayısı (Inside Airbnb'nin hesapladığı) | int64 | 0.0 | 67 | Öznitelik adayı | Aksiyon yok; 1.3-d modelinde kullanıldı |
| 87 | `calculated_host_listings_count_entire_homes` | …bunlardan tüm ev olanlar | int64 | 0.0 | 63 | Öznitelik adayı | Aksiyon yok |
| 88 | `calculated_host_listings_count_private_rooms` | …bunlardan özel oda olanlar | int64 | 0.0 | 32 | Öznitelik adayı | Aksiyon yok |
| 89 | `calculated_host_listings_count_shared_rooms` | …bunlardan paylaşımlı oda olanlar | int64 | 0.0 | 8 | Öznitelik adayı | Aksiyon yok; %98 sıfır |
| 90 | `reviews_per_month` | Aylık ortalama yorum | float64 | 35.5 | 736 | Öznitelik adayı | %35,5 eksik (yorumsuz); medyan + add_indicator |
