# =====================================================================
# 2.2 Karar ağacının yorumu: kök ve ilk iki düzey
# Önkoşul: ortak hat hücreleri (X, y, veri, CV_GRUP, SAYISAL, KATEGORIK, YUKSEK_KARD, NADIR_ESIK, SEED, SINIFLAR)
# =====================================================================
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# 1) Ağaç için ÖLÇEKSİZ ön işleme. Ağaç "kapasite > 3,5?" gibi eşik soruları sorar; ölçekleme soruların
#    cevabını değiştirmez ama eşikleri okunamaz yapar (ör. "kapasite ≤ 0,37"). Doldurma ve kodlama aynı.
def agac_on_isleme():
    return ColumnTransformer([
        ("sayisal", SimpleImputer(strategy="median", add_indicator=True), SAYISAL),
        ("kategorik", OneHotEncoder(handle_unknown="ignore"), KATEGORIK),
        ("yuksek_kard", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=NADIR_ESIK), [YUKSEK_KARD]),
    ], verbose_feature_names_out=False)          # sütun adları önek almasın: "kapasite", "oda_tipi_Private room"

def agac_hatti():
    # 2.1'deki "yaprak ≥ 20" ağacı: her yaprakta en az 20 ilan
    return Pipeline([("on", agac_on_isleme()),
                     ("model", DecisionTreeClassifier(min_samples_leaf=20, random_state=SEED))])

# Kontrol: ölçeksiz hat 2.1'deki ölçekli ağaçla aynı skoru vermeli (ağaç ölçekten etkilenmez)
_ = degerlendir(agac_hatti(), X, ad="Karar ağacı · yaprak ≥ 20 · ölçeksiz hat (2.1 ile aynı olmalı)")

# 2) Yorum için bütün veride bir ağaç. Bu ağaç YALNIZ yapıyı okumak için; skor raporunda kullanılmıyor.
agac = agac_hatti().fit(X, y)
ad = list(agac.named_steps["on"].get_feature_names_out())
t = agac.named_steps["model"].tree_
print(f"\nBütün veride ağaç: derinlik {agac.named_steps['model'].get_depth()}, yaprak {agac.named_steps['model'].get_n_leaves()}")
print(export_text(agac.named_steps["model"], feature_names=ad, class_names=SINIFLAR, max_depth=2, decimals=2))

# 3) Eşikleri okunur hâle getirme: log1p ile dönüştürülmüş sütunlarda eşiği geri çevir (expm1)
def okunur_kural(sutun, esik, sol):
    if sutun.startswith("log_"):
        ham = np.expm1(esik)
        birim = {"log_min_gece": "gece", "log_ev_sahibi_ilan": "ilan", "log_aciklama_uzunlugu": "karakter"}.get(sutun, "")
        return f"{sutun[4:]} {'≤' if sol else '>'} {ham:.1f} {birim}".strip()
    if sutun.startswith("missingindicator_"):
        return f"{sutun[17:]} {'dolu' if sol else 'EKSİK'}"
    if sutun not in SAYISAL:                       # bire-bir kodlanmış kategori (0/1), eşik hep 0,5
        return f"{sutun} {'DEĞİL' if sol else 'EVET'}"
    return f"{sutun} {'≤' if sol else '>'} {esik:.2f}"

def soru(dugum):                                   # düğümün sorduğu soru, okunur biçimde
    return okunur_kural(ad[t.feature[dugum]], t.threshold[dugum], True)

# 4) Kök ve ilk iki düzeydeki her düğüm: kaç ilan, dilim dağılımı, çoğunluk
satirlar = []
def gez(dugum, derinlik, yol):
    oran = t.value[dugum][0] / t.value[dugum][0].sum()          # düğümdeki sınıf oranları
    satirlar.append({"düzey": derinlik, "kural": " ∧ ".join(yol) if yol else "(kök: bütün ilanlar)",
                     "ilan": int(t.n_node_samples[dugum]),
                     **{f"% {s}": round(100 * o, 1) for s, o in zip(SINIFLAR, oran)},
                     "çoğunluk": SINIFLAR[int(np.argmax(oran))],
                     "sonraki soru": soru(dugum) if derinlik < 2 else ""})
    if derinlik < 2:
        s, e = ad[t.feature[dugum]], t.threshold[dugum]
        gez(t.children_left[dugum], derinlik + 1, yol + [okunur_kural(s, e, True)])
        gez(t.children_right[dugum], derinlik + 1, yol + [okunur_kural(s, e, False)])
gez(0, 0, [])
dugum_tablosu = pd.DataFrame(satirlar)
display(dugum_tablosu)

# 5) Kararlılık: aynı soruları her eğitim katında da soruyor mu? (25 bölmenin eğitim katları)
kararlilik = []
for tr, _te in CV_GRUP:
    m = agac_hatti().fit(X.iloc[tr], y[tr])
    tt, aa = m.named_steps["model"].tree_, m.named_steps["on"].get_feature_names_out()
    L, R = tt.children_left[0], tt.children_right[0]
    kararlilik.append({"kök": aa[tt.feature[0]], "kök eşiği": tt.threshold[0],
                       "sol çocuk": aa[tt.feature[L]], "sağ çocuk": aa[tt.feature[R]]})
kararlilik = pd.DataFrame(kararlilik)
print("Kök ve birinci düzey, 25 eğitim katında (kaç kez seçildi):")
for k in ["kök", "sol çocuk", "sağ çocuk"]:
    print(f"  {k:<10s}", kararlilik[k].value_counts().to_dict())
kok = kararlilik.loc[kararlilik["kök"] == "log_min_gece", "kök eşiği"]
if len(kok):
    print(f"  Kök eşiği (en az konaklama, gece): {np.expm1(kok.min()):.1f} – {np.expm1(kok.max()):.1f}")

# 6) Alan bilgisi için bağlam: kök sorusunun iki tarafında fiyat
uzun = veri["minimum_nights"] > np.expm1(t.threshold[0])
print(f"\nKökün sağı (en az konaklama > {np.expm1(t.threshold[0]):.0f} gece): {uzun.mean():.1%} ilan; "
      f"medyan gecelik fiyat {veri.loc[uzun, 'fiyat'].median():,.0f} TL (sol: {veri.loc[~uzun, 'fiyat'].median():,.0f} TL)")
print("En az konaklamanın en sık değerleri (sağ taraf):", veri.loc[uzun, "minimum_nights"].value_counts().head(4).to_dict())
bosluk = veri["minimum_nights"].between(15, 99)
print(f"Kök eşiklerinin düştüğü 15–99 gece aralığındaki ilan: {bosluk.sum()} ({bosluk.mean():.1%}) → "
      "eşik bu boşlukta nereye konsa da ayrım pratikte aynı: '≤ 14 gece' ile '≥ 100 gece'")

# 7) Görsel: ilk iki düzey
fig, ax = plt.subplots(figsize=(14, 5.5))
plot_tree(agac.named_steps["model"], max_depth=2, feature_names=ad, class_names=SINIFLAR,
          filled=True, impurity=True, proportion=True, rounded=True, fontsize=8, ax=ax)
ax.set_title("2.2 · Karar ağacı (yaprak ≥ 20), kök ve ilk iki düzey · renk = çoğunluk dilimi")
plt.tight_layout(); plt.show()
print("Grafik: her kutuda soru, Gini safsızlığı, ilan oranı ve dilim oranları (ucuz, orta, pahalı). "
      "log_min_gece eşiği log ölçekte; okunur karşılığı yukarıdaki tabloda.")
