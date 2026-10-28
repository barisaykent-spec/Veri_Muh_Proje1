# =====================================================================
# 1.3-d — add_indicator deneyi: "eksikti" bayrağı modele bir şey katıyor mu?
# 1.3-c hücresinden sonra yeni bir hücreye yapıştırın (df13 hazırken).
# =====================================================================
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

# ---- 1) Modelleme verisi (ORTAK_KARARLAR: kısa dönem + fiyatı olanlar) ----
fiyat = df13["price"].str.replace(r"[$,]", "", regex=True).astype(float)   # "$4,736.32" -> 4736.32
veri  = df13[(df13["minimum_nights"] < 30) & fiyat.notna()].copy()
fiyat = fiyat[veri.index]

# Hedef: sabit sınırlarla üç dilim (ORTAK_KARARLAR, karar 4)
y = pd.cut(fiyat, [-np.inf, 3455.50, 5877.67, np.inf], right=False,
           labels=["ucuz", "orta", "pahali"]).astype(str)
gruplar = veri["host_id"]                       # bölme ev sahibine göre
print("Satır:", len(veri), "| Sınıflar:", y.value_counts().to_dict())

# ---- 2) Öznitelikler: şimdilik yalnız sayısal adaylar (kategorikler 1.4'te) ----
sayisal_oznitelikler = [
    "accommodates", "bedrooms", "beds", "latitude", "longitude",
    "minimum_nights", "maximum_nights",
    "availability_30", "availability_60", "availability_90", "availability_365",
    "number_of_reviews", "number_of_reviews_ltm", "reviews_per_month",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value",
    "host_listings_count", "calculated_host_listings_count", "hosts_time_as_host_years",
]
X = veri[sayisal_oznitelikler]
print("Eksik içeren öznitelik sayısı:", int((X.isna().mean() > 0).sum()))

# ---- 3) İki hat: tek fark add_indicator ----
#   Pipeline = adımları sırayla bağlayan zincir. cross_val_score her katta
#   zinciri YALNIZ eğitim verisiyle kurar (medyan da eğitim verisinden hesaplanır)
#   -> sızıntı yok. 1.5'te bunun neden önemli olduğunu deneyle göreceğiz.
def hat(add_indicator):
    return Pipeline([
        ("doldur",  SimpleImputer(strategy="median", add_indicator=add_indicator)),
        ("olcekle", StandardScaler()),
        ("model",   LogisticRegression(max_iter=2000)),
    ])

# ---- 4) Değerlendirme: ev sahibine göre 5 kat x 5 tekrar = 25 skor ----
def skorla(model, X):
    skorlar = []
    for tohum in range(5):                       # 5 farklı karıştırma = tekrarlı ÇD
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=tohum)
        skorlar.extend(cross_val_score(model, X, y, groups=gruplar, cv=cv,
                                       scoring="f1_macro"))
    return np.array(skorlar)

sonuc = {}
sonuc["A: bayraksız"]     = skorla(hat(False), X)
sonuc["B: add_indicator"] = skorla(hat(True),  X)

# ---- 5) Yan deney: yorum sayısı sütunları OLMADAN ----
#   Puanın eksik olduğunu model zaten number_of_reviews = 0'dan anlayabilir.
#   Bu sütunları çıkarırsak bayrağın tek başına katkısını görürüz.
X_yorumsuz = X.drop(columns=["number_of_reviews", "number_of_reviews_ltm", "reviews_per_month"])
sonuc["C: yorum sayısı yok, bayraksız"]     = skorla(hat(False), X_yorumsuz)
sonuc["D: yorum sayısı yok, add_indicator"] = skorla(hat(True),  X_yorumsuz)

tablo = pd.DataFrame({k: {"ortalama": v.mean(), "std": v.std()} for k, v in sonuc.items()}).T.round(4)
print("\nMakro F1 (ev sahibine göre 5x5 çapraz doğrulama):")
print(tablo.to_string())

# ---- 6) Grafik: hata çubuklarıyla (yönerge 3.4: ortalama ± std) ----
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.barh(tablo.index[::-1], tablo["ortalama"][::-1], xerr=tablo["std"][::-1],
        color="#3b6ea5", height=0.5, capsize=4)
ax.set_xlabel("Makro F1 (ortalama ± std, 25 skor)")
ax.set_xlim(0.30, max(0.60, tablo["ortalama"].max() + 0.05))
ax.axvline(1/3, color="#999999", linestyle="--", linewidth=1)
ax.set_ylim(-0.6, len(tablo) - 0.2)
ax.text(1/3 + 0.003, len(tablo) - 0.45, "şans düzeyi ≈ 0,33", color="#666666", fontsize=8, va="center")
ax.set_title("add_indicator etkisi — lojistik regresyon, sayısal öznitelikler")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.show()
print("Açıklama: A–B ve C–D çiftleri yalnızca 'eksikti' bayrağının varlığıyla ayrışıyor.")
