from sklearn.model_selection import GridSearchCV
# =====================================================================
# 2.3 Optimist yanlılık: GridSearchCV.best_score_ ile iç içe ÇD skorunun karşılaştırması
# Önkoşul: ortak hat + 2.2 (agac_on_isleme). Ayarlanan model: karar ağacı (hızlı ve ayara duyarlı).
# İç ÇD de EV SAHİBİNE GÖRE gruplu: ayar aşamasında da aynı ev sahibi hem eğitimde hem doğrulamada olmamalı.
# =====================================================================
import time
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

IZGARA = {"model__max_depth": [4, 6, 8, 12, 16, None],
          "model__min_samples_leaf": [1, 5, 20, 50]}              # 6 × 4 = 24 aday
ADAYLAR = [dict(zip(IZGARA, d)) for d in
           [(a, b) for a in IZGARA["model__max_depth"] for b in IZGARA["model__min_samples_leaf"]]]

def agac_ayar_hatti():
    return Pipeline([("on", agac_on_isleme()), ("model", DecisionTreeClassifier(random_state=SEED))])

def ic_cv(tohum=SEED):
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=tohum)

# ---- A) Yanlış ama yaygın: bütün veride ayar yap, best_score_'u rapor et
t0 = time.time()
gs = GridSearchCV(agac_ayar_hatti(), IZGARA, cv=ic_cv(), scoring="f1_macro", n_jobs=-1)
gs.fit(X, y, groups=gruplar)
print(f"A) GridSearchCV (gruplu 5 kat, 24 aday) best_score_ = {gs.best_score_:.3f} · seçilen: {gs.best_params_}  ({time.time()-t0:.0f} sn)")

# ---- B) Doğru: iç içe ÇD. Dış kat yalnız SKOR için; ayar her dış eğitim katının İÇİNDE yapılır.
#      Dış bölme = CV_GRUP'un ilk turu (5 kat, her ilan bir kez test edilir).
#      Analiz için her dış katta 24 adayın hepsinin dış test skorunu da saklıyoruz (C bölümü).
ic_skor, dis_skor, secilen, ic_en_iyi, dis_secilen = [], [], [], [], []
for k, (tr, te) in enumerate(CV_GRUP[:5]):
    t0 = time.time()
    g = GridSearchCV(agac_ayar_hatti(), IZGARA, cv=ic_cv(SEED + 100 + k), scoring="f1_macro", n_jobs=-1)
    g.fit(X.iloc[tr], y[tr], groups=gruplar[tr])                  # ayar: yalnız dış eğitim katı
    ic_skor.append(g.cv_results_["mean_test_score"])               # 24 adayın iç ÇD skoru
    # 24 adayın her birini dış eğitim katında eğitip dış testte ölç (yalnız analiz için; seçimde kullanılmıyor)
    dis_skor.append(np.array([f1_score(y[te], agac_ayar_hatti().set_params(**p).fit(X.iloc[tr], y[tr]).predict(X.iloc[te]),
                                       average="macro") for p in ADAYLAR]))
    j = int(np.argmax(ic_skor[-1]))
    secilen.append(ADAYLAR[j]); ic_en_iyi.append(ic_skor[-1][j]); dis_secilen.append(dis_skor[-1][j])
    print(f"   dış kat {k+1}: iç en iyi {ic_en_iyi[-1]:.3f} → dış test {dis_secilen[-1]:.3f} · seçilen {secilen[-1]}  ({time.time()-t0:.0f} sn)")
ic_skor, dis_skor = np.array(ic_skor), np.array(dis_skor)
ic_en_iyi, dis_secilen = np.array(ic_en_iyi), np.array(dis_secilen)
print(f"B) İç içe ÇD skoru = {dis_secilen.mean():.3f} ± {dis_secilen.std():.3f}  "
      f"(iç ÇD'deki en iyi skorların ortalaması {ic_en_iyi.mean():.3f}; her katta şişme {np.mean(ic_en_iyi - dis_secilen):+.3f})")

# ---- C) Şişme aday sayısıyla büyür mü? Izgaradan rastgele k aday seç; iç ÇD'ye göre en iyiyi al;
#      "iç skor − aynı adayın dış test skoru" farkını k'ya göre ortala (her k için 300 rastgele alt küme × 5 dış kat).
rng = np.random.default_rng(SEED)
egri = []
for kk in [1, 2, 4, 8, 12, 16, 24]:
    farklar = []
    for _ in range(300 if kk < 24 else 1):
        alt = rng.choice(len(ADAYLAR), kk, replace=False)
        for f in range(5):
            j = alt[np.argmax(ic_skor[f, alt])]
            farklar.append(ic_skor[f, j] - dis_skor[f, j])
    egri.append({"aday sayısı": kk, "şişme (iç − dış)": np.mean(farklar)})
egri = pd.DataFrame(egri)

# ---- D) Bölme yanlışsa ayar da yanlış: iç ÇD SATIRA göre (ev sahibini bilmeden) yapılırsa
from sklearn.model_selection import StratifiedKFold
gs_satir = GridSearchCV(agac_ayar_hatti(), IZGARA, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                        scoring="f1_macro", n_jobs=-1).fit(X, y)
gercek_satir = degerlendir(agac_ayar_hatti().set_params(**gs_satir.best_params_), X,
                           ad=f"Satıra göre seçilen ağaç {gs_satir.best_params_} · gruplu ÇD")
gercek_grup = degerlendir(agac_ayar_hatti().set_params(**gs.best_params_), X,
                          ad=f"Gruplu seçilen ağaç {gs.best_params_} · gruplu ÇD")

# ---- E) Rastgele ormanın OOB skoru: satıra göre "iç" tahmin (yerine koyarak örneklemede dışarıda kalan ilanlar)
oob, test_rf = [], []
for tr, te in CV_GRUP[:5]:
    rf = Pipeline([("on", on_isleme()), ("model", RandomForestClassifier(n_estimators=100, min_samples_leaf=2, oob_score=True,
                                                                        n_jobs=-1, random_state=SEED))]).fit(X.iloc[tr], y[tr])
    oob.append(f1_score(y[tr], rf.named_steps["model"].oob_decision_function_.argmax(1), average="macro"))
    test_rf.append(f1_score(y[te], rf.predict(X.iloc[te]), average="macro"))
oob, test_rf = np.array(oob), np.array(test_rf)

# ---- Özet tablo: "rapor edilen" skor ile "gerçekçi" skor
ozet = pd.DataFrame([
    {"kaynak": "Ayar + rapor aynı ÇD'de (best_score_, gruplu iç ÇD)", "rapor edilen": gs.best_score_,
     "gerçekçi (iç içe / gruplu)": dis_secilen.mean()},
    {"kaynak": "Ayar satıra göre ÇD'de (best_score_)", "rapor edilen": gs_satir.best_score_,
     "gerçekçi (iç içe / gruplu)": gercek_satir["f1_macro_ort"]},
    {"kaynak": "Rastgele orman OOB skoru (satıra göre)", "rapor edilen": oob.mean(),
     "gerçekçi (iç içe / gruplu)": test_rf.mean()},
]).set_index("kaynak")
ozet["şişme"] = ozet["rapor edilen"] - ozet["gerçekçi (iç içe / gruplu)"]
display(ozet.round(3))
print(f"Satıra göre seçilen ağaç, gruplu seçilen ağaca göre gruplu ÇD'de {gercek_satir['f1_macro_ort'] - gercek_grup['f1_macro_ort']:+.3f} "
      f"(daha kötü olduğu bölme: {int((gercek_satir['_skorlar'] < gercek_grup['_skorlar']).sum())}/25)")

# ---- Grafik: (sol) 24 adayın iç ve dış skoru · (sağ) şişme ~ aday sayısı
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].scatter(ic_skor.mean(0), dis_skor.mean(0), color="#4C72B0")
lim = [min(ic_skor.mean(0).min(), dis_skor.mean(0).min()) - 0.01, max(ic_skor.mean(0).max(), dis_skor.mean(0).max()) + 0.01]
ax[0].plot(lim, lim, color="grey", ls=":", lw=1)
ax[0].set(xlabel="İç ÇD skoru (ayar sırasında görülen)", ylabel="Dış test skoru (görülmemiş ev sahipleri)",
          title="24 aday · 5 dış katın ortalaması")
ax[1].plot(egri["aday sayısı"], egri["şişme (iç − dış)"], marker="o", color="#C44E52")
ax[1].axhline(0, color="grey", lw=1)
ax[1].set(xlabel="Denenen aday sayısı", ylabel="Seçilen adayda iç − dış skor", title="Seçim yanlılığı aday sayısıyla büyüyor mu?")
plt.tight_layout(); plt.show()
print("Grafik: solda noktalı çizgi iç = dış; altında kalan aday ayar sırasında olduğundan iyi görünmüş. "
      "Sağda her aday sayısı için 300 rastgele alt küme × 5 dış kat ortalaması.")
display(egri.round(4).set_index("aday sayısı").T)
