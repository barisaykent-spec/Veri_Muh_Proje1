from statsmodels.stats.contingency_tables import mcnemar
# =====================================================================
# 2.4 İstatistiksel test: iki modelin farkı tesadüf olabilir mi?
# Önkoşul: ortak hat + 2.1 (MODELLER_21, skorlar_21).
# Dietterich (1998): aynı veriden üretilmiş bölmelerin skorları bağımsız değildir; 25 skora sıradan
# t-testi uygulamak "fark yokken var" deme (I. tür hata) oranını şişirir. Önerileri: 5×2 ÇD t-testi
# (model 10 kez eğitilebiliyorsa) ya da McNemar (yalnız bir kez eğitilebiliyorsa).
# Bizde ek koşul: bütün bölmeler EV SAHİBİNE GÖRE gruplu (1.5).
# =====================================================================
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

KARSILASTIRMALAR = [   # (A, B): en iyi iki model + iki kontrol çifti
    ("Gradyan artırma (HistGB)", "Rastgele orman · 300 ağaç"),     # en iyi iki model (2.1'de fark +0,005)
    ("Rastgele orman · 300 ağaç", "Lojistik regresyon (1.6)"),     # açık fark beklenen çift (+0,035)
    ("k-NN · k = 15 (ölçekli)", "Karar ağacı · yaprak ≥ 20"),      # farklı aileler, skorlar neredeyse aynı
]
MODELLER_24 = sorted({m for cift in KARSILASTIRMALAR for m in cift})

# ---- 1) Gruplu 5×2 ÇD: 5 tekrar × 2 kat; her katta bütün modeller AYNI bölmede eğitilir (eşleştirilmiş)
BOLME_5x2 = []
for r in range(5):
    sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=SEED + r)
    BOLME_5x2.append(list(sgkf.split(np.zeros(len(y)), y, gruplar)))
assert all(len(set(gruplar[a]) & set(gruplar[b])) == 0 for (a, b), _ in BOLME_5x2)

skor_5x2 = {m: np.zeros((5, 2)) for m in MODELLER_24}
for r, katlar in enumerate(BOLME_5x2):
    for j, (tr, te) in enumerate(katlar):
        for m in MODELLER_24:
            tahmin = clone(MODELLER_21[m]).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            skor_5x2[m][r, j] = f1_score(y[te], tahmin, average="macro")
    print(f"5×2 ÇD tekrar {r + 1}/5 tamam")

def t_5x2(p):                                    # Dietterich (1998), serbestlik derecesi 5
    s2 = ((p - p.mean(1, keepdims=True)) ** 2).sum(1)
    t = p[0, 0] / np.sqrt(s2.mean())
    return t, 2 * stats.t.sf(abs(t), df=5)

def f_5x2(p):                                    # Alpaydın (1999): 10 farkın hepsini kullanan birleşik F testi
    s2 = ((p - p.mean(1, keepdims=True)) ** 2).sum(1)
    f = (p ** 2).sum() / (2 * s2.sum())
    return f, stats.f.sf(f, 10, 5)

def t_saf(d):                                    # YANLIŞ: 25 skoru bağımsız sayan eşli t-testi (karşılaştırma için)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return t, 2 * stats.t.sf(abs(t), df=len(d) - 1)

def t_duzeltilmis(d, test_orani=1 / 4):          # Nadeau & Bengio (2003): örtüşen eğitim kümeleri için düzeltme
    t = d.mean() / np.sqrt((1 / len(d) + test_orani) * d.var(ddof=1))
    return t, 2 * stats.t.sf(abs(t), df=len(d) - 1)

# ---- 2) McNemar: tek bir gruplu ayırma (eğitim 2/3, test 1/3), modeller bir kez eğitilir
tr_h, te_h = next(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y, gruplar))
dogru_h = {m: clone(MODELLER_21[m]).fit(X.iloc[tr_h], y[tr_h]).predict(X.iloc[te_h]) == y[te_h] for m in MODELLER_24}

# ---- 3) Bütün testler, her çift için
satirlar = []
for a_, b_ in KARSILASTIRMALAR:
    p = skor_5x2[a_] - skor_5x2[b_]                     # 5 × 2 fark matrisi
    d25 = skorlar_21[a_] - skorlar_21[b_]               # 2.1'deki 25 gruplu bölmenin farkları
    ia, ib = dogru_h[a_], dogru_h[b_]
    tablo2x2 = [[int(( ia &  ib).sum()), int(( ia & ~ib).sum())],
                [int((~ia &  ib).sum()), int((~ia & ~ib).sum())]]
    mc = mcnemar(tablo2x2, exact=False, correction=True)
    satirlar.append({
        "karşılaştırma (A − B)": f"{a_}  −  {b_}",
        "fark (25 bölme)": d25.mean(), "fark (5×2)": p.mean(),
        "p · saf t (25, YANLIŞ)": t_saf(d25)[1],
        "p · düzeltilmiş t (25)": t_duzeltilmis(d25)[1],
        "p · 5×2 t (Dietterich)": t_5x2(p)[1],
        "p · 5×2 F (Alpaydın)": f_5x2(p)[1],
        "McNemar: yalnız A doğru": tablo2x2[0][1], "McNemar: yalnız B doğru": tablo2x2[1][0],
        "p · McNemar": mc.pvalue,
    })
test_tablosu = pd.DataFrame(satirlar).set_index("karşılaştırma (A − B)")
display(test_tablosu.round(4))
print(f"McNemar ayırması: eğitim {len(tr_h):,} · test {len(te_h):,} ilan (ev sahibi kesişimi yok: "
      f"{len(set(gruplar[tr_h]) & set(gruplar[te_h])) == 0}). McNemar doğruluk üzerinden çalışır (doğru/yanlış), makro F1 değil.")
print("Not: 3 çift × 5 test = çoklu karşılaştırma; tek bir p < 0,05 tek başına kanıt sayılmamalı. "
      "Karar için önceden seçilen test: gruplu 5×2 ÇD t (Dietterich), destek: 5×2 F.")

# ---- 4) Grafik: her çiftin 10 farkı (5×2) ve 25 farkı (2.1)
fig, ax = plt.subplots(1, len(KARSILASTIRMALAR), figsize=(14, 3.8), sharey=False)
for a, (a_, b_) in zip(ax, KARSILASTIRMALAR):
    a.scatter(np.zeros(25) + np.random.default_rng(0).uniform(-0.08, 0.08, 25), skorlar_21[a_] - skorlar_21[b_],
              s=14, color="#8da0cb", label="25 bölme (5×5, 2.1)")
    a.scatter(np.ones(10) + np.random.default_rng(1).uniform(-0.08, 0.08, 10), (skor_5x2[a_] - skor_5x2[b_]).ravel(),
              s=18, color="#C44E52", label="10 kat (5×2)")
    a.axhline(0, color="grey", lw=1)
    a.set(xticks=[0, 1], xticklabels=["5×5", "5×2"], xlim=(-0.5, 1.5), ylabel="Makro F1 farkı (A − B)",
          title=f"{a_.split(' ·')[0].split(' (')[0]} − {b_.split(' ·')[0].split(' (')[0]}")
ax[0].legend(fontsize=7, loc="lower left")
plt.tight_layout(); plt.show()
print("Grafik: her nokta bir katta A ile B'nin makro F1 farkı; sıfır çizgisinin iki yanına dağılım farkın tutarlılığını gösterir.")
