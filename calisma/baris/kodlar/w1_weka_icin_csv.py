# =====================================================================
# WEKA-1 — Weka'nın açabileceği bir CSV hazırlama
# 1.3-a hücresinden sonra çalıştırın (df13 hazırken).
# Amaç: Weka'nın CSV okuyucusunu ÇÖKERTEN sütunları ayıklamak; geri kalan
# temizlik kararlarını (boş sütunlar, kimlikler, satır filtresi, eksik
# değerler) bilerek WEKA'DA, filtrelerle vereceğiz.
# =====================================================================

weka = df13.copy()

# 1) Weka'yı çökerten sütunlar — sebebi bu sütunların İÇERİĞİ:
#    - uzun metin ve listeler: içinde "" kaçışı, virgül, satır sonu var
#      ("Index 90 out of bounds" hatası)
#    - URL'ler: her satırda farklı; Weka nominal değerleri ilk 100 satırdan öğrendiği
#      için 101. satırda yeni değer görünce çöküyor ("Read unknown nominal value")
#    - kişi/ilan adları: içlerinde kesme işareti (') var; Weka onu tırnak sanıyor
cokertenler = ["description", "host_about", "amenities", "price_quote_raw",
               "name", "host_name"] + [c for c in weka.columns if c.endswith("_url")]
weka = weka.drop(columns=cokertenler)

# 2) Kalan metinlerdeki kesme işaretlerini temizle (ör. host_location'da 2 kayıt)
for c in weka.columns:
    if not pd.api.types.is_numeric_dtype(weka[c]):
        weka[c] = weka[c].str.replace("'", "", regex=False)

# 3) Fiyatı sayıya çevir: "$4,736.32" -> 4736.32
#    (Weka bunu metin olarak okur ve 13.000 değerli bir kategori sanar.)
weka["price"] = weka["price"].str.replace(r"[$,]", "", regex=True).astype(float)

# BİLEREK BIRAKTIKLARIMIZ (Weka'da bulup temizleyeceğiz):
#   14 tamamen boş sütun, scrape_id (sabit), id ve host_id (sayısal kimlik),
#   sızıntı sütunları, uzun dönem ve fiyatsız ilanlar.

weka.to_csv("weka_adim1.csv", index=False)
print("Kaydedildi: weka_adim1.csv |", weka.shape[0], "satır,", weka.shape[1], "sütun")
print("Çıkarılan", len(cokertenler), "sütun:", cokertenler)

# Colab'dan bilgisayarına indirmek için:
from google.colab import files
files.download("weka_adim1.csv")
