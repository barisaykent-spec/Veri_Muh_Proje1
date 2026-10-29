# =====================================================================
# 1.4 — Kategorik kodlama: bire-bir, nadirleri birleştiren bire-bir,
#        kat dışı hedef kodlama ve hedef kodlamanın YANLIŞ uygulanması
# 1.3-d hücresinden sonra çalıştırın (veri, y, gruplar, sayisal_oznitelikler,
# skorla() hazır olmalı).
# Süre: Colab'da yaklaşık 10–15 dakika (license için bire-bir kodlama yavaş).
# =====================================================================
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning
# license için bire-bir kodlama binlerce sütun üretir; lojistik regresyon bazı katlarda
# yakınsama uyarısı verir. Sonucu değiştirmediği için uyarıları gizliyoruz (yorumda not edilmeli).
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---- 0) Hangi sütunlar? ----
# property_type: 66 kategori, 36'sında 10'dan az ilan var -> "nadir kategori" sorununun tipik örneği
# license     : binlerce farklı değer -> gerçekten yüksek kardinaliteli; hedef kodlama sızıntısını en net gösteren sütun
for c in ["property_type", "license"]:
    sayilar = veri[c].fillna("YOK").value_counts()
    print(f"{c:<14} farklı değer: {len(sayilar):>5} | 10'dan az ilanı olan: {(sayilar < 10).sum():>5} "
          f"| tek ilanlık: {(sayilar == 1).sum():>5}")

# Kategorik sütunlardaki boşları "YOK" adlı bir kategori yapıyoruz (eksik de bir bilgi)
for c in ["property_type", "license"]:
    veri[c] = veri[c].fillna("YOK")

# ---- 1) Hat kurucu: sayısal kısım hep aynı, sadece kategorik kodlayıcı değişiyor ----
sayisal_hat = Pipeline([
    ("doldur",  SimpleImputer(strategy="median", add_indicator=True)),
    ("olcekle", StandardScaler()),
])

def hat_kur(kodlayici, kategorik_sutun):
    adimlar = [("sayisal", sayisal_hat, sayisal_oznitelikler)]
    if kodlayici is not None:
        # hedef kodlamanın çıktısı da ölçeklensin diye kodlayıcıyı bir mini hatta koyuyoruz
        adimlar.append(("kategorik",
                        Pipeline([("kodla", kodlayici), ("olcekle", StandardScaler(with_mean=False))]),
                        [kategorik_sutun]))
    return Pipeline([
        ("hazirla", ColumnTransformer(adimlar)),
        ("model",   LogisticRegression(max_iter=3000)),
    ])

# ---- 2) Dört kodlama yöntemi ----
def kodlayicilar():
    return {
        # a) Bire-bir (one-hot): her kategori ayrı bir 0/1 sütunu
        "bire-bir":                  OneHotEncoder(handle_unknown="ignore"),
        # b) Nadirleri birleştiren bire-bir: eğitim verisinde 30'dan az görülen kategoriler "diğer" olur
        "bire-bir + nadir birleştir": OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30),
        # c) Kat dışı hedef kodlama: kategoriyi, o kategorideki sınıf oranlarıyla temsil eder.
        #    TargetEncoder eğitim verisinin İÇİNDE de çapraz doğrulama yapar (kat dışı / out-of-fold),
        #    böylece bir ilan kendi etiketini "göremez".
        "kat dışı hedef kodlama":     TargetEncoder(target_type="multiclass", random_state=42),
    }

# ---- 3) YANLIŞ hedef kodlama: oranları BÜTÜN veriden (test dahil) hesaplamak ----
def yanlis_hedef_kodlama(kategorik_sutun):
    """Her kategori için ucuz/orta/pahalı oranlarını tüm 16.558 ilandan hesaplar.
    Test katındaki ilanların etiketleri de bu oranlara girer -> sızıntı."""
    oranlar = pd.crosstab(veri[kategorik_sutun], y, normalize="index")      # kategori x sınıf oranı
    kodlu = oranlar.reindex(veri[kategorik_sutun]).to_numpy()
    X_yanlis = veri[sayisal_oznitelikler].copy()
    for i, sinif in enumerate(oranlar.columns):
        X_yanlis[f"te_{sinif}"] = kodlu[:, i]
    return X_yanlis

# ---- 4) Deney ----
sonuclar = {}
X_tum = veri[sayisal_oznitelikler + ["property_type", "license"]]
sonuclar[("—", "kategorik yok (taban)")] = skorla(hat_kur(None, None), X_tum)

for sutun in ["property_type", "license"]:
    for ad, kodlayici in kodlayicilar().items():
        sonuclar[(sutun, ad)] = skorla(hat_kur(kodlayici, sutun), X_tum)
    # yanlış uygulama: hedef kodlanmış sütunlar zaten "sayısal" hale geldi
    X_yanlis = yanlis_hedef_kodlama(sutun)
    yanlis_hat = Pipeline([("doldur", SimpleImputer(strategy="median", add_indicator=True)),
                           ("olcekle", StandardScaler()),
                           ("model", LogisticRegression(max_iter=3000))])
    sonuclar[(sutun, "YANLIŞ: hedef kodlama bütün veride")] = skorla(yanlis_hat, X_yanlis)

tablo14 = pd.DataFrame({k: {"ortalama": v.mean(), "std": v.std()} for k, v in sonuclar.items()}).T.round(4)
tablo14.index.names = ["sütun", "kodlama"]
print("\nMakro F1 (lojistik regresyon, ev sahibine göre 5x5 ÇD):")
print(tablo14.to_string())

# ---- 5) Grafik ----
import matplotlib.pyplot as plt
etiketler = [f"{s} · {k}" if s != "—" else k for s, k in tablo14.index]
renkler = ["#c0392b" if "YANLIŞ" in k else "#3b6ea5" for _, k in tablo14.index]
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(etiketler[::-1], tablo14["ortalama"][::-1], xerr=tablo14["std"][::-1],
        color=renkler[::-1], height=0.55, capsize=4)
ax.set_xlabel("Makro F1 (ortalama ± std, 25 skor)")
ax.set_xlim(0.50, min(1.0, tablo14["ortalama"].max() + 0.05))
ax.set_title("Kategorik kodlama yöntemleri\n(kırmızı: hedef kodlamanın yanlış, sızıntılı uygulanması)")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.show()
print("Açıklama: Mavi çubuklar sızıntısız hatlar; kırmızı çubuk, hedef kodlama oranları test "
      "etiketleri de dahil bütün veriden hesaplandığında elde edilen (geçersiz) skor.")
