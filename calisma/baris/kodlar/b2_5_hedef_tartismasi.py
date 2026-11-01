# =====================================================================
# 2.5 (tartışma) — Hedefimiz ne kadar sağlam? Keyfi sınırlar başarıyı nasıl etkiler?
# Önkoşul: defterdeki "Model tablosu ve ortak değerlendirme hattı" hücreleri
# çalışmış olmalı (hedefli, X, y, gruplar, CV_GRUP, on_isleme, lojistik, SEED,
# SINIFLAR, ESIKLER). Süre: Colab'da ~3-4 dk.
#
# Soru: Etiket (fiyat dilimi) bizim seçtiğimiz 2750 / 5100 sınırlarıyla üretildi.
#   (A) Modelin hataları nerede? Ciddi mi (ucuz<->pahalı) yoksa komşu mu?
#   (B) Sınırları değiştirirsek F1 ne olur? Modelin öğrendiği şey sınıra bağlı mı?
# Model: 1.6'daki lojistik regresyon (hızlı; amaç model değil HEDEFİ sorgulamak).
# =====================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, r2_score, confusion_matrix
from sklearn.model_selection import cross_val_predict

fiyat = hedefli["fiyat"].to_numpy()
# cross_val_predict her ilanı tam bir kez test etmeli -> CV_GRUP'un ilk tekrarı (5 kat) kullanılır
CV_TEK = CV_GRUP[:5]
hat_lr = lambda: Pipeline([("on", on_isleme(kodlama="nadir", add_indicator=True)), ("model", lojistik())])

# ---------------------------------------------------------------------
# A) Hata yapısı: karışıklık matrisi ve sınıra uzaklık
# ---------------------------------------------------------------------
tahmin = cross_val_predict(hat_lr(), X, y, cv=CV_TEK, n_jobs=-1)
print(f"Makro F1 (tek tekrar): {f1_score(y, tahmin, average='macro'):.3f}\n")

km = pd.DataFrame(confusion_matrix(y, tahmin), index=[f"gerçek {s}" for s in SINIFLAR],
                  columns=[f"tahmin {s}" for s in SINIFLAR])
display(km)

fark = np.abs(y - tahmin)              # 0 = doğru, 1 = komşu sınıf, 2 = uç hata (ucuz<->pahalı)
hata_turu = pd.Series(fark).map({0: "doğru", 1: "komşu sınıf hatası", 2: "uç hata (ucuz↔pahalı)"})
display(hata_turu.value_counts(normalize=True).round(3).rename("oran").to_frame())

# Fiyatın en yakın sınıra oransal (log) uzaklığı: 2.749 TL ile 2.751 TL farklı sınıfta!
uzaklik = np.min([np.abs(np.log(fiyat / e)) for e in ESIKLER], axis=0)
bant = pd.cut(uzaklik, [-0.01, 0.05, 0.10, 0.20, 0.40, np.inf],
              labels=["<%5", "%5–10", "%10–20", "%20–40", ">%40"])
tablo_bant = (pd.DataFrame({"bant": bant, "hata": fark > 0})
              .groupby("bant", observed=True)["hata"].agg(ilan_sayisi="size", hata_orani="mean"))
display(tablo_bant.round(3))

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar(tablo_bant.index.astype(str), tablo_bant["hata_orani"], color="#4C72B0")
ax.axhline(1 - f1_score(y, tahmin, average="micro"), color="#999999", ls="--", lw=1)
ax.set(xlabel="Fiyatın en yakın sınıra uzaklığı", ylabel="Hata oranı",
       title="Hatalar sınırların etrafında yığılıyor")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.show()
print("Grafik: kesikli çizgi genel hata oranı. Sınıra yakın ilanlarda model neredeyse yazı-tura atıyor;\n"
      "bu bir model zayıflığından çok etiketin keskin sınırından kaynaklanıyor.")

# ---------------------------------------------------------------------
# B) Sınır duyarlılığı: F1 sınıra bağlı mı, öğrenilen sinyal sınıra bağlı mı?
#    Spearman = modelin "beklenen sınıf"ı (olasılıklarla ağırlıklı) ile gerçek fiyatın sıra korelasyonu
#    -> ilanları ucuzdan pahalıya ne kadar doğru diziyor; sınır seçiminden bağımsız bir ölçü.
# ---------------------------------------------------------------------
sinir_setleri = {
    "Ekip (2750 / 5100)":   ESIKLER,
    "%10 aşağı":            [e * 0.9 for e in ESIKLER],
    "%10 yukarı":           [e * 1.1 for e in ESIKLER],
    "Yuvarlak (2500 / 5000)": [2500, 5000],
    "2 sınıf (medyan)":     [np.median(fiyat)],
    "4 sınıf (çeyrekler)":  list(np.quantile(fiyat, [0.25, 0.50, 0.75])),
}
satirlar = []
for ad, sinir in sinir_setleri.items():
    y_s = np.digitize(fiyat, sinir)          # pd.cut(..., right=False) ile aynı: sınıra eşit fiyat üst sınıfa
    olas = cross_val_predict(hat_lr(), X, y_s, cv=CV_TEK, method="predict_proba", n_jobs=-1)
    beklenen = olas @ np.arange(olas.shape[1])
    satirlar.append({"sınır seti": ad, "sınıf sayısı": len(sinir) + 1,
                     "makro F1": f1_score(y_s, olas.argmax(1), average="macro"),
                     "şans düzeyi": 1 / (len(sinir) + 1),
                     "Spearman (sıralama gücü)": spearmanr(beklenen, fiyat).correlation})

# Hiç sınır kullanmayan karşılaştırma: log(fiyat) için regresyon
log_fiyat = np.log(fiyat)
reg = cross_val_predict(Pipeline([("on", on_isleme(kodlama="nadir", add_indicator=True)), ("model", Ridge())]),
                        X, log_fiyat, cv=CV_TEK, n_jobs=-1)
satirlar.append({"sınır seti": "Sınırsız: log(fiyat) regresyonu", "sınıf sayısı": np.nan,
                 "makro F1": np.nan, "şans düzeyi": np.nan,
                 "Spearman (sıralama gücü)": spearmanr(reg, fiyat).correlation})
tablo_sinir = pd.DataFrame(satirlar).set_index("sınır seti")
display(tablo_sinir.round(3))
print(f"Regresyon: R² = {r2_score(log_fiyat, reg):.3f} · medyan mutlak yüzde hata = "
      f"%{np.median(np.abs(np.exp(reg) / fiyat - 1)) * 100:.1f}")
print("Okuma: makro F1 sınır seçimine göre oynuyor (özellikle sınıf sayısıyla); sıralama gücü ise\n"
      "her sınır setinde ve sınırsız regresyonda aynı -> modelin öğrendiği sinyal sınırdan bağımsız.")
