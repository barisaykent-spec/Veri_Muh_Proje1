# =====================================================================
# 1.3-a — Kodlanmış eksikleri NaN'a (ya da doğru anlamına) çevirme
# Defterde 1.3'ün "# Kodunuz" hücresine yapıştırın (df hazırken).
# Bu dönüşümler veriden bir şey ÖĞRENMEZ (satır satır kural uygular),
# bu yüzden bölmeden önce bütün veride yapılması sızıntı yaratmaz.
# =====================================================================
import numpy as np

df13 = df.copy()          # ham df'yi bozmamak için kopya üzerinde çalışıyoruz
once = df13.isna().sum()  # dönüşümden önceki eksik sayıları (karşılaştırma için)

# 1) license = "12-3456" -> NaN
#    103 ilanda aynı numara; "123456" sıralı rakamlar: tipik bir yer tutucu (sahte) değer.
#    "Non-real estate listing" ve "Exempt" ise GERÇEK kategorilerdir (eksik değil), dokunmuyoruz.
df13.loc[df13["license"] == "12-3456", "license"] = np.nan

# 2) review_scores_rating < 1 -> NaN
#    Puan ölçeği 1–5; 0 imkânsız bir değer. Satırı silmek yerine sadece hatalı hücreyi boşaltıyoruz
#    (ilanın diğer 89 bilgisi doğru olabilir).
df13.loc[df13["review_scores_rating"] < 1, "review_scores_rating"] = np.nan

# 3) has_availability boş -> "f"
#    Boş olan 457 ilanın 447'sinde fiyat da yok: boş burada "bilinmiyor" değil, "müsait değil" demek.
#    Bu, ters yönde bir kodlama: eksik görünen değer aslında bilinen bir değer.
df13["has_availability"] = df13["has_availability"].fillna("f")

# DOKUNMADIKLARIMIZ (gerekçeleri yorum hücresinde):
#   bedrooms = 0          -> stüdyo / ortak oda: geçerli sıfır
#   number_of_reviews = 0 -> yeni ilan: geçerli sıfır (puanların boş olmasının nedeni)
#   maximum_nights = 365 / 1125 -> platform varsayılanı ("üst sınır koymamış"); eksik değil, not ediyoruz

sonra = df13.isna().sum()
degisen = (sonra - once)
print("Eksik sayısı değişen sütunlar:")
print(degisen[degisen != 0].to_string())
print("\nhas_availability değerleri:", df13["has_availability"].value_counts().to_dict())
