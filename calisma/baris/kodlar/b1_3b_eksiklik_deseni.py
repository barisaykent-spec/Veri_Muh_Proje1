# =====================================================================
# 1.3-b — Eksiklik desenini görselleştirme
# 1.3-a hücresinden sonra yeni bir hücreye yapıştırın (df13 hazırken).
# =====================================================================
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Kısmen eksik olan sütunlar (tamamen boş ya da tamamen dolu olanlar desen göstermez)
eksik = df13.isna().mean()
kismi = eksik[(eksik > 0) & (eksik < 1)].sort_values(ascending=False)

# ---- Grafik 1: Sütun başına eksik oranı ----
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(kismi.index[::-1], kismi.values[::-1] * 100, color="#3b6ea5", height=0.6)
ax.set_xlabel("Eksik oranı (%)")
ax.set_ylabel("Sütun")
ax.set_title("Kısmen eksik sütunlar: eksik oranı (27.232 ilan)")
ax.grid(axis="x", color="#e5e5e5"); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.show()
print("Açıklama: Puan sütunlarının hepsi aynı oranda (%35,5) eksik; fiyat ve fiyattan türeyen "
      "sütunlar da kendi aralarında aynı oranda (%9,8) eksik.")

# ---- Grafik 2: Eksiklik matrisi (her satır bir ilan, koyu = eksik) ----
# Satırları anlamlı bir sırayla diziyoruz: önce fiyatı eksik olanlar, sonra yorum sayısına göre.
# Böylece "birlikte eksik olan" sütunlar dikey bloklar halinde görünür.
sirali = (df13.assign(_fiyat_yok=df13["price"].isna())
               .sort_values(["_fiyat_yok", "number_of_reviews"], ascending=[False, True]))
ornek = sirali.iloc[::10]                       # her 10 ilandan biri: ~2.700 satır, okunaklı
matris = ornek[kismi.index].isna().to_numpy()

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(matris, aspect="auto", interpolation="nearest",
          cmap=ListedColormap(["#f2f2f2", "#1f3b5c"]))   # açık = dolu, koyu = eksik
ax.set_xticks(range(len(kismi)))
ax.set_xticklabels(kismi.index, rotation=90, fontsize=8)
ax.set_xlabel("Sütun")
ax.set_ylabel("İlan (önce fiyatı eksik olanlar, sonra yorum sayısı artan)")
ax.set_title("Eksiklik matrisi — koyu hücre = eksik değer")
ax.set_yticks([])
plt.tight_layout(); plt.show()
print("Açıklama: Puan sütunları aynı satırlarda birlikte eksik (yatay koyu bant), "
      "fiyat sütunları da ayrı bir blok halinde birlikte eksik.")

# ---- Sayısal kanıt: puanların eksik olduğu satırlarda yorum sayısı kaç? ----
puan_yok = df13["review_scores_accuracy"].isna()
print("\nPuan eksik olan ilanlarda number_of_reviews = 0 oranı:",
      round((df13.loc[puan_yok, "number_of_reviews"] == 0).mean(), 3))
print("Puan dolu olan ilanlarda number_of_reviews = 0 oranı:",
      round((df13.loc[~puan_yok, "number_of_reviews"] == 0).mean(), 3))
