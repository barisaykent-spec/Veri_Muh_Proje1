import joblib
# =====================================================================
# 2.5 Proje 2'ye aktarılacak model ve çıktılar
# Önkoşul: ortak hat + 2.1 (MODELLER_21, tablo_21) + 2.3 (ozet) + 2.4 (test_tablosu)
# Proje 2 defteri şunları okuyor: pd.read_csv("p1_temiz_veri.csv"), joblib.load("p1_hat.joblib")
# Veri dosyaları GitHub'a konmaz (.gitignore: *.csv, *.joblib); küçük protokol dosyası (JSON) konabilir.
# =====================================================================
import json, os
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score

# ---- 1) Seçim: rastgele orman, 2.1'deki SABİT ayarlarla (300 ağaç, yaprak ≥ 2)
# Gerekçe: 2.4'te HistGB − RF farkı (+0,005) sağlam testlerin hiçbirinde anlamlı değil → skor seçimi belirlemiyor.
# RF: ayarsız kararlı, iyi bilinen, E1.3 ve Proje 2 açıklanabilirlik (permütasyon önemi) için doğal aday.
# Ayar yapılmadı: 2.3'te ayarın kazancı ve seçim yanlılığı çok küçüktü; sabit ayar 2.1 skorunu yanlılıksız bırakır.
SECILEN = "Rastgele orman · 300 ağaç"
hat = clone(MODELLER_21[SECILEN])

# ---- 2) Kat dışı tahminler: her ilan, kendi ev sahibini HİÇ görmemiş bir modelle tahmin edilir (CV_GRUP'un ilk turu)
# Proje 2'de alt grup / küme / aykırı gözlem hata analizleri BUNLARLA yapılmalı, bütün veride eğitilmiş modelin
# kendi eğitim ilanlarına tahminiyle değil (o tahminler ezber içerir).
oof_olasilik = cross_val_predict(clone(hat), X, y, cv=CV_GRUP[:5], method="predict_proba", n_jobs=-1)
oof_tahmin = oof_olasilik.argmax(1)
print(f"Kat dışı tahmin (CV_GRUP ilk tur, 5 kat): makro F1 = {f1_score(y, oof_tahmin, average='macro'):.3f} "
      f"(2.1, 25 bölme: {tablo_21.loc[SECILEN, 'makro_F1_ort']:.3f} ± {tablo_21.loc[SECILEN, 'makro_F1_std']:.3f})")

# ---- 3) Son model: bütün 24.553 ilanla eğitilir (yeni veriye uygulanacak model budur; skoru 2.1'deki ÇD skorudur)
hat.fit(X, y)
egitim_ici = f1_score(y, hat.predict(X), average="macro")
print(f"Son model bütün veride eğitildi · kendi eğitim verisindeki makro F1 = {egitim_ici:.3f} (RAPOR EDİLMEZ: ezber içerir)")
joblib.dump(hat, "p1_hat.joblib", compress=3)

# ---- 4) Temizlenmiş veri: model girdileri (X'in sütunları, aynı adlar) + hedef + bölme grubu + bağlam + kat dışı tahminler
df_temiz = pd.concat([
    veri[["id", "host_id", "fiyat", "minimum_nights", "neighbourhood_cleansed"]].rename(
        columns={"neighbourhood_cleansed": "ilce_ham"}),
    X,
    pd.DataFrame({"fiyat_dilimi": veri["fiyat_dilimi"].astype(str), "y": y,
                  "p1_oof_tahmin": oof_tahmin,
                  **{f"p1_oof_olasilik_{s}": oof_olasilik[:, k] for k, s in enumerate(SINIFLAR)},
                  "p1_oof_dogru": (oof_tahmin == y).astype(int)}, index=X.index),
], axis=1)
df_temiz.to_csv("p1_temiz_veri.csv", index=False)

# ---- 5) Protokol: Proje 2'nin aynı kurulumu yeniden üretebilmesi için her şey tek dosyada
protokol = {
    "veri": {"kaynak": VERI_URL, "anlik_goruntu": "2026-09-25", "ilan": int(len(X)), "ev_sahibi": int(len(np.unique(gruplar))),
             "kapsam": KAPSAM, "senaryo": SENARYO},
    "hedef": {"sinirlar_TL": ESIKLER_MODEL, "siniflar": SINIFLAR, "kodlar": {s: k for k, s in enumerate(SINIFLAR)},
              "aralik": "sol kapalı: [alt, üst)"},
    "oznitelikler": {"sayisal": SAYISAL, "kategorik": KATEGORIK, "yuksek_kardinalite": YUKSEK_KARD,
                     "on_isleme": "medyan doldurma + add_indicator + StandardScaler; bire-bir kodlama; mulk_tipi nadir (<30) birleşik"},
    "model": {"ad": SECILEN, "parametreler": {k: v for k, v in hat.named_steps["model"].get_params().items()
                                               if k in ("n_estimators", "min_samples_leaf", "max_features", "random_state")}},
    "degerlendirme": {"bolme": "StratifiedGroupKFold(groups=host_id), 5 kat × 5 tekrar, tohum 42–46 (CV_GRUP)",
                      "olcut": "f1_macro (ikincil: balanced_accuracy)",
                      "makro_F1_ort": round(float(tablo_21.loc[SECILEN, "makro_F1_ort"]), 4),
                      "makro_F1_std": round(float(tablo_21.loc[SECILEN, "makro_F1_std"]), 4),
                      "dengeli_dogruluk_ort": round(float(tablo_21.loc[SECILEN, "dengeli_dogruluk_ort"]), 4),
                      "baseline_stratified": round(float(tablo_21.loc["Dummy · sınıf oranlarıyla", "makro_F1_ort"]), 4),
                      "oof_makro_F1_ilk_tur": round(float(f1_score(y, oof_tahmin, average="macro")), 4)},
    "surumler": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                 "numpy": np.__version__},
    "uyarilar": ["Hata analizleri p1_oof_* sütunlarıyla yapılmalı; p1_hat.joblib'in eğitim ilanlarına tahmini ezber içerir.",
                 "Yeni bölmelerde host_id gruplaması korunmalı (1.5: satıra göre bölme ağaç/ormanda +0,08–0,10 şişirir).",
                 "Kök ayrım 'en az konaklama ≤ 20 gece' (2.2): 100 gecelik teklifler ayrı bir alt grup olarak incelenmeli.",
                 "HistGB − RF farkı anlamlı değil (2.4); model seçimi skora dayanmıyor.",
                 "CV_GRUP bölmeleri scikit-learn sürümüne bağlı; 1.9.1 dışında sayılar ~0,005 oynayabilir."]}
with open("p1_protokol.json", "w", encoding="utf-8") as f:
    json.dump(protokol, f, ensure_ascii=False, indent=2)

# ---- 6) Kontrol: kaydedilenler geri okunduğunda aynı sonucu veriyor mu? (Proje 2'nin yapacağı gibi)
hat2 = joblib.load("p1_hat.joblib")
df2 = pd.read_csv("p1_temiz_veri.csv", low_memory=False)
ayni = (hat2.predict(df2[X.columns]) == hat.predict(X)).mean()
print(f"Geri okuma kontrolü: tahminlerin {ayni:.1%}'i aynı · CSV {df2.shape[0]:,} × {df2.shape[1]} · "
      f"kat dışı makro F1 (CSV'den) = {f1_score(df2['y'], df2['p1_oof_tahmin'], average='macro'):.3f}")
assert ayni == 1.0

dosyalar = pd.DataFrame([{"dosya": d, "boyut (MB)": round(os.path.getsize(d) / 1e6, 1), "GitHub'a": g, "içerik": i}
    for d, g, i in [("p1_hat.joblib", "hayır (.gitignore)", "ön işleme + RF, bütün veride eğitilmiş Pipeline"),
                    ("p1_temiz_veri.csv", "hayır (.gitignore)", "24 girdi + hedef + host_id + bağlam + kat dışı tahminler"),
                    ("p1_protokol.json", "evet", "sınırlar, öznitelikler, bölme, skorlar, sürümler, uyarılar")]]).set_index("dosya")
display(dosyalar)

# ---- 7) Kat dışı hata: Proje 2'ye ilk ipucu (alt gruplarda doğruluk)
alt = pd.DataFrame({"oda tipi": X["oda_tipi"], "en az konaklama": np.where(veri["minimum_nights"] >= 100, "≥ 100 gece", "< 100 gece"),
                    "doğru": oof_tahmin == y})
display(pd.concat([alt.groupby(k)["doğru"].agg(["size", "mean"]).rename(columns={"size": "ilan", "mean": "doğruluk"})
                   .assign(alt_grup=k).set_index("alt_grup", append=True).swaplevel() for k in ["oda tipi", "en az konaklama"]]).round(3))
print("Bu tablo yalnız bir ön bakış; alt grup analizi Proje 2 Bölüm 3'ün konusu.")
