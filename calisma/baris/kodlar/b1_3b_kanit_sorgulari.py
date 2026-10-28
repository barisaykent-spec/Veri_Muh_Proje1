# =====================================================================
# 1.3-b (devam) — Grafikte gördüğümüzü "select" mantığıyla doğrulama
# SQL karşılığı:  SELECT grup, AVG(sutun IS NULL) FROM df13 GROUP BY grup
# pandas'ta:      df13["sutun"].isna().groupby(grup).mean()
# =====================================================================
import numpy as np

yorum_grubu = pd.cut(df13["number_of_reviews"], [-1, 0, 5, 30, 10_000],
                     labels=["0", "1-5", "6-30", "30+"])
gece_grubu  = np.where(df13["minimum_nights"] >= 100, "100+ gece", "100'den az")

def eksik_orani_gruba_gore(sutun, grup, grup_adi):
    tablo = (df13[sutun].isna().groupby(grup, observed=True).mean() * 100).round(1)
    print(f"{sutun:<22} eksik % | {grup_adi}: {tablo.to_dict()}")

print("--- Blok 1: puan sütunları ---")
eksik_orani_gruba_gore("review_scores_rating", yorum_grubu, "yorum sayısı")

print("\n--- Blok 2: fiyat sütunları ---")
eksik_orani_gruba_gore("price", df13["availability_365"] == 0, "takvim tamamen kapalı mı")
eksik_orani_gruba_gore("price", gece_grubu, "en az konaklama")

print("\n--- license ---")
eksik_orani_gruba_gore("license", gece_grubu, "en az konaklama")

print("\n--- Dağınık görünenler: host_about, host_location ---")
eksik_orani_gruba_gore("host_about", yorum_grubu, "yorum sayısı")
eksik_orani_gruba_gore("host_about", df13["host_is_superhost"], "süper ev sahibi")
eksik_orani_gruba_gore("host_location", yorum_grubu, "yorum sayısı")
