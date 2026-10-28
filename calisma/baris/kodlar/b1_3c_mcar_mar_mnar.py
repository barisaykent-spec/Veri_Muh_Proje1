# =====================================================================
# 1.3-c — MCAR / MAR / MNAR tahmini ve veriyle gerekçesi
# Fikir: Eksiklik TAMAMEN rastgele (MCAR) olsaydı, değeri eksik olan ilanlarla
# dolu olan ilanlar diğer özellikler bakımından BENZER olurdu.
# Belirgin farklar varsa eksiklik gözlenen bir şeye bağlıdır (MAR ya da MNAR).
# SQL karşılığı:  SELECT (sutun IS NULL) AS eksik, AVG(x1), AVG(x2) ... GROUP BY eksik
# =====================================================================

karsilastirma_sutunlari = {
    "number_of_reviews":              "yorum sayısı (ort.)",
    "availability_365":               "yıllık müsait gün (ort.)",
    "minimum_nights":                 "en az konaklama (ort.)",
    "accommodates":                   "kişi kapasitesi (ort.)",
    "calculated_host_listings_count": "ev sahibinin ilan sayısı (ort.)",
}

def eksik_mi_dolu_mu(sutun):
    """sutun eksik olan ve dolu olan ilanları diğer özelliklerde karşılaştırır."""
    grup = df13[sutun].isna().map({True: "EKSİK", False: "dolu"})
    tablo = df13.groupby(grup)[list(karsilastirma_sutunlari)].mean().round(1).T
    tablo.index = list(karsilastirma_sutunlari.values())
    # süper ev sahibi oranı (t/f sütunu olduğu için ayrı hesaplıyoruz)
    tablo.loc["süper ev sahibi oranı (%)"] = (
        (df13["host_is_superhost"] == "t").groupby(grup).mean().mul(100).round(1))
    tablo.loc["ilan sayısı"] = grup.value_counts()
    print(f"\n=== {sutun}: eksik % {df13[sutun].isna().mean()*100:.1f} ===")
    print(tablo[["dolu", "EKSİK"]].to_string())

# Model adayı olan sütunlar arasında en çok eksiği olan iki sütun:
eksik_mi_dolu_mu("host_location")          # %35,9
eksik_mi_dolu_mu("review_scores_rating")   # %35,5
# Ek olarak hedef değişkenimiz (fiyatı eksik satırları dışarıda bırakma kararımızın gerekçesi):
eksik_mi_dolu_mu("price")                  # %9,8
