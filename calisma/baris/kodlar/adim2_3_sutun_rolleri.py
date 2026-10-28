# =====================================================================
# BIL545 Proje 1 — Barış'ın çalışma notları
# Adım 2 ve Adım 3: Sütunları gruplama ve her sütuna rol verme
#
# Kullanım (Colab): defterde "Veriyi yükleme" hücresi çalıştıktan sonra
# (yani df hazırken) aşağıdaki iki hücreyi sırayla yeni hücrelere yapıştırın.
# Beklenen sonuç: A=15, B=8, D=4, kalan=63; roller toplamı 90.
# =====================================================================

# ---------------------------------------------------------------------
# HÜCRE 1 — ADIM 2: Mekanik kurallarla gruplama (A / B / D)
# ---------------------------------------------------------------------
import pandas as pd

# Her sütun için üç temel ölçü
eksik_orani  = df.isna().mean()                 # 0..1 arası eksik oranı
farkli       = df.nunique(dropna=False)         # NaN'ı da bir değer sayarak farklı değer sayısı
benzersizlik = df.nunique() / len(df)           # farklı değer / satır sayısı (1'e yakınsa kimlik benzeri)

kalan = list(df.columns)                         # henüz bir gruba atanmamış sütunlar

# A — Bilgisiz: tamamen boş ya da tek değerli (NaN dahil)
A_bilgisiz = [c for c in kalan if eksik_orani[c] == 1 or farkli[c] == 1]
kalan = [c for c in kalan if c not in A_bilgisiz]

# B — Kimlik / URL: neredeyse her satırda farklı, ya da adı kimlik/URL olduğunu söylüyor
B_kimlik = [c for c in kalan
            if benzersizlik[c] > 0.95
            or c == "id"
            or c.endswith("_id")
            or c.endswith("_url")]
kalan = [c for c in kalan if c not in B_kimlik]

# D — Ham metin / liste: sayısal OLMAYAN sütunlarda, boşlar hariç ortalama uzunluk > 50
#     (Not: "is_string_dtype" pandas 2.x'te NaN içeren sütunlarda yanlış sonuç veriyordu;
#      bu yüzden "sayısal değil mi?" diye soruyoruz — her sürümde aynı çalışır.)
def ort_uzunluk(s):
    return s.dropna().astype(str).str.len().mean()

metin_sutunlari = [c for c in kalan if not pd.api.types.is_numeric_dtype(df[c])]
D_metin = [c for c in metin_sutunlari if ort_uzunluk(df[c]) > 50]
kalan = [c for c in kalan if c not in D_metin]

# Özet
print(f"A bilgisiz : {len(A_bilgisiz):>2}  {A_bilgisiz}")
print(f"B kimlik   : {len(B_kimlik):>2}  {B_kimlik}")
print(f"D metin    : {len(D_metin):>2}  {D_metin}")
print(f"Kalan      : {len(kalan):>2}")
print("Toplam     :", len(A_bilgisiz) + len(B_kimlik) + len(D_metin) + len(kalan))

# Kalan sütunları elle inceleyebilmek için küçük bir tablo
kalan_tablo = pd.DataFrame({
    "tip":     df[kalan].dtypes.astype(str),
    "eksik_%": (eksik_orani[kalan] * 100).round(1),
    "farkli":  df[kalan].nunique(),
    "ornek":   [str(df[c].dropna().iloc[0])[:30] for c in kalan],
})
kalan_tablo


# ---------------------------------------------------------------------
# HÜCRE 2 — ADIM 3: Her sütuna bir rol ver
# ---------------------------------------------------------------------

# 1) Özel rolü olan üç sütun: modele GİRMEZLER ama tabloda kalırlar
HEDEF   = "price"     # tahmin edeceğimiz şey; fiyat dilimini bundan üreteceğiz
GRUP    = "host_id"   # aynı ev sahibinin ilanları hem eğitimde hem testte olmasın diye bölmede kullanılacak
ANAHTAR = "id"        # ilanın kimliği; reviews.csv gibi başka tablolara bağlanmak için

# 2) SIZINTI: değerini hesaplamak için fiyatı bilmek gerekiyor
#    Test sorusu: "Yeni bir ilanın fiyatını bilmeden bu sütunu bilebilir miyim?" -> Hayır
C_sizinti = [
    "price_quote_price_per_night",  # fiyatın birebir aynısı
    "price_quote_total_price",      # fiyat x gece sayısı
    "price_quote_checkin_date",     # fiyat teklifinin giriş tarihi: teklifin parçası
    "price_quote_checkout_date",    # fiyat teklifinin çıkış tarihi: teklifin parçası
    "estimated_revenue_l365d",      # fiyat x doluluk: dolaylı olarak fiyatı içeriyor
]

# 3) TOPLAMA SÜRECİNİN İZLERİ: evin değil, verinin nasıl toplandığının bilgisi
S_surec = [
    "last_scraped",            # kazıyıcının ilanı ziyaret ettiği gün (28 veya 29 Eylül)
    "calendar_last_scraped",   # takvimin okunduğu gün
    "source",                  # "önceki kazımadan mı geldi" bilgisi
]

# 4) KURALIN KAÇIRDIĞI SERBEST METİNLER: kısa ama yine de serbest yazı
T_metin = [
    "name",       # ilan başlığı, 25.571 farklı değer
    "host_name",  # ev sahibinin adı; kişisel veri, modele katkısı yok
]

# 5) Geriye kalanlar ÖZNİTELİK adayıdır
cikarilacaklar = C_sizinti + S_surec + T_metin + [HEDEF]
E_aday = [c for c in kalan if c not in cikarilacaklar]

# 6) Her sütunun rolünü tek bir tabloda topla
rol = {}
for c in A_bilgisiz: rol[c] = "at: bilgisiz"
for c in B_kimlik:   rol[c] = "at: kimlik/URL"
for c in D_metin:    rol[c] = "at (şimdilik): ham metin"
for c in T_metin:    rol[c] = "at (şimdilik): ham metin"
for c in C_sizinti:  rol[c] = "at: SIZINTI"
for c in S_surec:    rol[c] = "at: toplama süreci izi"
for c in E_aday:     rol[c] = "ÖZNİTELİK adayı"
rol[HEDEF]   = "HEDEF"                 # bu üçü yukarıdaki genel rolün üzerine yazılır
rol[GRUP]    = "GRUP (bölme)"
rol[ANAHTAR] = "ANAHTAR (bağlantı)"

roller = pd.Series(rol, name="rol")
print("Toplam sütun:", len(roller))    # 90 olmalı -> hiçbir sütunu atlamadık
print(roller.value_counts())
# Beklenen: ÖZNİTELİK adayı 52 · bilgisiz 15 · kimlik/URL 6 · ham metin 6 ·
#           SIZINTI 5 · süreç izi 3 · HEDEF 1 · GRUP 1 · ANAHTAR 1
