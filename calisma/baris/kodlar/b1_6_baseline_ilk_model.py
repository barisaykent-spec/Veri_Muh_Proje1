# =====================================================================
# 1.6 — Baseline ve ilk model (sızıntısız hat içinde)
# 1.3-d hücresinden sonra çalıştırın (veri, y, gruplar hazır olmalı). Süre: Colab'da ~4-5 dk.
# Önce bekleyen TİP DÖNÜŞÜMLERİNİ yapıyoruz (Adım 4). Bunlar satır satır kural
# uygular, veriden bir şey ÖĞRENMEZ -> bölmeden önce yapılması sızıntı değildir.
# =====================================================================
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

# ---------------------------------------------------------------------
# 1) Tip dönüşümleri (Adım 4)
# ---------------------------------------------------------------------
def tipleri_donustur(d):
    d = d.copy()
    # a) t/f -> 1/0
    for c in ["host_is_superhost", "host_has_profile_pic", "host_identity_verified", "has_availability"]:
        d[c] = d[c].map({"t": 1, "f": 0})

    # b) bathrooms_text -> sayı + tip   ("1.5 shared baths" -> 1.5 ve "paylaşımlı")
    bt = d["bathrooms_text"].str.lower()
    d["banyo_sayisi"] = pd.to_numeric(bt.str.extract(r"([\d.]+)")[0], errors="coerce")
    d.loc[bt.str.contains("half", na=False) & d["banyo_sayisi"].isna(), "banyo_sayisi"] = 0.5
    d["banyo_paylasimli"] = bt.str.contains("shared", na=False).astype(int)

    # c) tarihler -> kazıma tarihine göre gün sayısı (tahmin anında bilinen bilgi)
    kazima = pd.Timestamp("2026-09-25")
    d["ilk_yorumdan_gun"] = (kazima - pd.to_datetime(d["first_review"])).dt.days
    d["son_yorumdan_gun"] = (kazima - pd.to_datetime(d["last_review"])).dt.days

    # d) amenities -> olanak sayısı  ('["Wifi", "Kitchen"]' -> 2)
    d["olanak_sayisi"] = d["amenities"].str.count('",') + d["amenities"].str.contains('"').astype(int)
    return d

veri16 = tipleri_donustur(veri)

# ---------------------------------------------------------------------
# 2) Öznitelik listeleri
# ---------------------------------------------------------------------
sayisal = [
    "hosts_time_as_user_years", "hosts_time_as_host_years", "host_listings_count",
    "calculated_host_listings_count", "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms", "calculated_host_listings_count_shared_rooms",
    "host_is_superhost", "host_has_profile_pic", "host_identity_verified",
    "latitude", "longitude", "accommodates", "bedrooms", "beds", "banyo_sayisi", "banyo_paylasimli",
    "olanak_sayisi", "minimum_nights", "maximum_nights", "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
    "has_availability", "availability_30", "availability_60", "availability_90", "availability_365",
    "availability_eoy", "number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d",
    "number_of_reviews_ly", "estimated_occupancy_l365d", "reviews_per_month",
    "ilk_yorumdan_gun", "son_yorumdan_gun",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value",
]
kategorik = ["room_type", "neighbourhood_cleansed", "property_type", "license"]
X16 = veri16[sayisal + kategorik].copy()
X16[kategorik] = X16[kategorik].fillna("YOK")
print(f"Öznitelik: {len(sayisal)} sayısal + {len(kategorik)} kategorik | satır: {len(X16)}")

# ---------------------------------------------------------------------
# 3) Hatlar
# ---------------------------------------------------------------------
hazirla = ColumnTransformer([
    ("sayisal", Pipeline([("doldur", SimpleImputer(strategy="median", add_indicator=True)),
                          ("olcekle", StandardScaler())]), sayisal),
    # 1.4'ün sonucu: nadirleri birleştiren bire-bir en iyi sızıntısız sonucu vermişti
    ("kategorik", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30), kategorik),
])

hatlar = {
    # Baseline 1: hep en sık sınıfı söyler (Weka'daki ZeroR'ın karşılığı)
    "Dummy (en sık sınıf)":      Pipeline([("hazirla", hazirla), ("model", DummyClassifier(strategy="most_frequent"))]),
    # Baseline 2: sınıf oranlarına göre rastgele tahmin eder
    "Dummy (oranlara göre rastgele)": Pipeline([("hazirla", hazirla), ("model", DummyClassifier(strategy="stratified", random_state=42))]),
    # İlk model
    "Lojistik regresyon":        Pipeline([("hazirla", hazirla), ("model", LogisticRegression(max_iter=3000))]),
}

# ---------------------------------------------------------------------
# 4) Değerlendirme: ev sahibine göre 5 kat x 5 tekrar, iki ölçüt
# ---------------------------------------------------------------------
olcutler = {"makro_F1": "f1_macro", "dogruluk": "accuracy"}
sonuc16 = {}
for ad, hat in hatlar.items():
    birikim = {k: [] for k in olcutler}
    for tohum in range(5):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=tohum)
        r = cross_validate(hat, X16, y, groups=gruplar, cv=cv, scoring=olcutler)
        for k in olcutler:
            birikim[k].extend(r[f"test_{k}"])
    sonuc16[ad] = {f"{k} ort": np.mean(v) for k, v in birikim.items()} | \
                  {f"{k} std": np.std(v) for k, v in birikim.items()}

tablo16 = pd.DataFrame(sonuc16).T[["makro_F1 ort", "makro_F1 std", "dogruluk ort", "dogruluk std"]].round(4)
print("\nEv sahibine göre 5x5 çapraz doğrulama:")
print(tablo16.to_string())

# ---------------------------------------------------------------------
# 5) Grafik
# ---------------------------------------------------------------------
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 3))
ax.barh(tablo16.index[::-1], tablo16["makro_F1 ort"][::-1], xerr=tablo16["makro_F1 std"][::-1],
        color=["#3b6ea5", "#9aa5b1", "#9aa5b1"], height=0.5, capsize=4)
ax.axvline(1/3, color="#999999", linestyle="--", linewidth=1)
ax.set_xlabel("Makro F1 (ortalama ± std, 25 skor)")
ax.set_xlim(0, 0.8)
ax.set_title("Baseline ve ilk model (ev sahibine göre bölme)")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.show()
print("Açıklama: Gri çubuklar tahmin yapmayan baseline'lar; kesikli çizgi üç dengeli sınıfta şans düzeyi (1/3).")
