# =====================================================================
# 2.1 Model karşılaştırması
# Önkoşul: defterdeki "Model tablosu ve ortak değerlendirme hattı" hücreleri
# çalışmış olmalı (X, y, CV_GRUP, on_isleme, lojistik, SEED, SINIFLAR).
# Bütün modeller AYNI 25 bölmede (CV_GRUP) ve AYNI ön işleme hattında koşar;
# böylece farklar yalnız modelden gelir.
# =====================================================================
import time
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

OLCUTLER_21 = ["f1_macro", "balanced_accuracy"]   # ana ölçüt + ikincil ölçüt

def hat_kur(model, yogun=False):
    """Ortak ön işleme (1.3 ve 1.4 kararları) + model.
    yogun=True: HistGradientBoosting seyrek (sparse) matris kabul etmediği için çıktı yoğun yapılır."""
    on = on_isleme(kodlama="nadir", add_indicator=True)
    if yogun:
        on.set_params(sparse_threshold=0)
    return Pipeline([("on", on), ("model", model)])

# Hiperparametreler bilerek SABİT ve önceden seçildi (ayar yok): aynı ÇD'de hem ayar yapıp
# hem skor raporlamak optimist yanlılık üretir; ayar işi 2.3'te iç içe ÇD ile yapılacak.
MODELLER_21 = {
    "Dummy · en sık sınıf":        hat_kur(DummyClassifier(strategy="most_frequent")),
    "Dummy · sınıf oranlarıyla":    hat_kur(DummyClassifier(strategy="stratified", random_state=SEED)),
    "Lojistik regresyon (1.6)":     hat_kur(lojistik()),
    "Karar ağacı · sınırsız":       hat_kur(DecisionTreeClassifier(random_state=SEED)),
    "Karar ağacı · yaprak ≥ 20":    hat_kur(DecisionTreeClassifier(min_samples_leaf=20, random_state=SEED)),
    "k-NN · k = 15 (ölçekli)":      hat_kur(KNeighborsClassifier(n_neighbors=15)),
    "Rastgele orman · 300 ağaç":    hat_kur(RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                                   n_jobs=-1, random_state=SEED)),
    "Gradyan artırma (HistGB)":     hat_kur(HistGradientBoostingClassifier(random_state=SEED), yogun=True),
}

sonuc_21, skorlar_21 = [], {}
for ad, hat in MODELLER_21.items():
    t0 = time.time()
    s = cross_validate(hat, X, y, cv=CV_GRUP, scoring=OLCUTLER_21,
                       return_train_score=True, n_jobs=-1)
    skorlar_21[ad] = s["test_f1_macro"]                     # 25 değer → kutu grafiği ve 2.4
    sonuc_21.append({
        "model": ad,
        "makro_F1_ort": s["test_f1_macro"].mean(),
        "makro_F1_std": s["test_f1_macro"].std(),
        "dengeli_dogruluk_ort": s["test_balanced_accuracy"].mean(),
        "egitim_makro_F1": s["train_f1_macro"].mean(),       # eğitim–test farkı = ezberleme göstergesi
        "fark_egitim_test": s["train_f1_macro"].mean() - s["test_f1_macro"].mean(),
        "sure_sn": time.time() - t0,
    })
    print(f"{ad:<30s} makro F1 = {s['test_f1_macro'].mean():.3f} ± {s['test_f1_macro'].std():.3f}"
          f"   ({time.time() - t0:5.0f} sn)")

tablo_21 = pd.DataFrame(sonuc_21).set_index("model")
display(tablo_21.round(3))

# Kutu grafiği: her kutu 25 bölmenin makro F1 değerleri (5 kat × 5 tekrar)
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.boxplot(list(skorlar_21.values()), vert=False, tick_labels=list(skorlar_21.keys()), showmeans=True)
ax.axvline(tablo_21.loc["Dummy · sınıf oranlarıyla", "makro_F1_ort"], color="grey", ls=":", lw=1)
ax.invert_yaxis()
ax.set(xlabel="Makro F1 (gruplu tekrarlı ÇD, 25 bölme)", title="2.1 · Model karşılaştırması")
plt.tight_layout(); plt.show()
print("Grafik: kutu = çeyrekler arası aralık, çizgi = medyan, üçgen = ortalama; noktalı çizgi rastgele tahmin düzeyi.")

# Eşli karşılaştırma: aynı 25 bölmede iki modelin farkı (bölmeler ortak olduğu için farka bakmak daha adil)
for a_, b_ in [("Rastgele orman · 300 ağaç", "Lojistik regresyon (1.6)"),
               ("Gradyan artırma (HistGB)", "Rastgele orman · 300 ağaç"),
               ("k-NN · k = 15 (ölçekli)", "Karar ağacı · yaprak ≥ 20")]:
    d = skorlar_21[a_] - skorlar_21[b_]
    print(f"{a_} − {b_}: {d.mean():+.3f} ± {d.std():.3f} (25 bölmenin {int((d > 0).sum())}'inde önde)")
print("Not: bölmeler aynı veriden üretildiği için bu 25 fark bağımsız değil; anlamlılık testi 2.4'te (gruplu 5×2 ÇD).")
