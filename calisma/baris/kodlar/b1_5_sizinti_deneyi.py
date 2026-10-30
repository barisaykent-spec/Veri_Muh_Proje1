# =====================================================================
# 1.5 — Sızıntı deneyi: ön işleme bütün veride mi, Pipeline içinde mi?
# 1.3-d hücresinden sonra çalıştırın (veri, y, gruplar, sayisal_oznitelikler hazır olmalı).
# Ek olarak ikinci bir sızıntı türünü de ölçüyoruz: bölmede ev sahibini yok saymak.
# =====================================================================
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score

X = veri[sayisal_oznitelikler]

# ---- Değerlendirme: 5 kat x 5 tekrar; bölme türü seçilebilir ----
def skorla_cv(model, X, gruba_gore=True):
    skorlar = []
    for tohum in range(5):
        if gruba_gore:   # DOĞRU: aynı ev sahibinin ilanları hep aynı katta
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=tohum)
            s = cross_val_score(model, X, y, groups=gruplar, cv=cv, scoring="f1_macro")
        else:            # YANLIŞ: ev sahibi yok sayılır, ilanlar rastgele dağıtılır
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=tohum)
            s = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
        skorlar.extend(s)
    return np.array(skorlar)

# ---- İki model: biri doğrusal, biri ezberlemeye yatkın ----
modeller = {
    "Lojistik regresyon": lambda: LogisticRegression(max_iter=2000),
    "Karar ağacı":        lambda: DecisionTreeClassifier(min_samples_leaf=5, random_state=42),
}

sonuclar = {}
for model_adi, yeni_model in modeller.items():

    # (1) DOĞRU: ön işleme Pipeline içinde -> her katta medyan ve ölçek YALNIZ eğitim verisinden
    dogru_hat = Pipeline([
        ("doldur",  SimpleImputer(strategy="median", add_indicator=True)),
        ("olcekle", StandardScaler()),
        ("model",   yeni_model()),
    ])

    # (2) YANLIŞ: ön işleme BÜTÜN veride (test katları dahil) bir kez yapılıyor,
    #     model sonra bu "önceden hazırlanmış" veride çapraz doğrulanıyor
    X_on_islenmis = StandardScaler().fit_transform(
        SimpleImputer(strategy="median", add_indicator=True).fit_transform(X))

    sonuclar[(model_adi, "DOĞRU: Pipeline + ev sahibine göre bölme")] = skorla_cv(dogru_hat, X, True)
    sonuclar[(model_adi, "YANLIŞ-1: ön işleme bütün veride")]         = skorla_cv(yeni_model(), X_on_islenmis, True)
    sonuclar[(model_adi, "YANLIŞ-2: rastgele bölme (ev sahibi yok sayıldı)")] = skorla_cv(dogru_hat, X, False)

tablo15 = pd.DataFrame({k: {"ortalama": v.mean(), "std": v.std()} for k, v in sonuclar.items()}).T.round(4)
tablo15.index.names = ["model", "kurulum"]
# doğru kuruluma göre fark
tablo15["fark (doğruya göre)"] = tablo15.groupby(level="model")["ortalama"].transform(
    lambda s: (s - s.iloc[0]).round(4))
print("Makro F1 (5x5 çapraz doğrulama):")
print(tablo15.to_string())

# ---- Grafik: hata çubuklarıyla ----
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
for ax, model_adi in zip(axes, modeller):
    t = tablo15.loc[model_adi]
    renk = ["#3b6ea5" if k.startswith("DOĞRU") else "#c0392b" for k in t.index]
    ax.barh(range(len(t))[::-1], t["ortalama"], xerr=t["std"], color=renk, height=0.55, capsize=4)
    ax.set_yticks(range(len(t))[::-1])
    ax.set_yticklabels([k.split(":")[0] for k in t.index])
    ax.set_title(model_adi)
    ax.set_xlabel("Makro F1 (ortalama ± std, 25 skor)")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
axes[0].set_xlim(0.45, 0.70)
fig.suptitle("Sızıntı deneyi — mavi: doğru kurulum, kırmızı: sızıntılı kurulumlar")
plt.tight_layout(); plt.show()
print("Açıklama: YANLIŞ-1 = medyan ve ölçek bütün veriden öğrenildi; "
      "YANLIŞ-2 = aynı ev sahibinin ilanları hem eğitimde hem testte.")
