# Devir notu — BIL545 Proje 1 (Barış ile çalışma oturumu özeti)

Son güncelleme: 2026-11-02 (Bölüm 1 V2'den kopyalandı, sürüm 1.9.1'e geçildi, 2.1 deftere eklendi). Yeni bir sohbet açılınca Claude'un önce bu dosyayı okuması yeterli.

## Ders ve proje
- BIL545 Veri Madenciliğinde İleri Konular (Dr. Selma Güzel, Üsküdar Ü.), 2026–27 Güz. 5 kişilik grup → temel kapsam + Ek paket 1 + Ek paket 2.
- Konu 9.19 İstanbul Airbnb (Inside Airbnb, anlık görüntü **2026-09-25**, 27.232 × 90). Hedef: **gecelik fiyat dilimi** (ucuz/orta/pahalı).
- Puanlama: başarım değil kurulumun doğruluğu (%30), gerekçe (%25), dürüst raporlama (%25), yorum (%15), teknik düzen (%5). Sızıntı = kurulum puanı sıfır.
- Barış'ın sorumluluğu: **Bölüm 2 (sınıflandırma ve dürüst değerlendirme)**. Bölüm 1'i Kübra yazdı (aşağıda).

## Barış hakkında / çalışma biçimi
- Finans altyapılı analist; tablo sever; SQL/select alışkanlığı var. Türkçe, isimle hitap ("Barış").
- **Öğrenmek istiyor**: adım adım, önce kavram, sonra kod; Python + Weka paralel. Kodu verirken satır satır açıklama + yorum taslağı istiyor.
- Ortam (2026-11-02 kararı): Python 3.13, **pandas 3.0.5, scikit-learn 1.9.1** (Kübra'nın V2 defteriyle aynı). Eski karar 1.6.1'di.
  - Sürüm farkı `CV_GRUP` bölmelerini değiştiriyor: 1.9.1'de her test katının sınıf oranı birebir eşit (std 0,000), 1.6.1'de ±0,02. Skorlar ~0,002–0,006 oynuyor, sıralama değişmiyor.
  - 1.3 kontrol hücresindeki "eğitim katı eşikleriyle hiçbir test etiketi değişmiyor" yalnız 1.9.1'de doğru (1.6.1'de ort. %1, en fazla %4 değişiyor). Varsayım defterinde sürüm belirtilmeli.

## Depolar ve dosyalar
- Grup deposu: github.com/barisaykent-spec/Veri_Muh_Proje1 (public). Barış'ın denemeleri `calisma/baris/`.
  - `calisma/baris/kodlar/` : adim2_3 (sütun rolleri), b1_3a–d (eksiklik), b1_4 (kodlama), b1_5 (sızıntı), b1_6 (baseline), w1 (Weka CSV), b2_1 (model karşılaştırması)
  - `calisma/baris/belgeler/sutun_sozlugu.md|csv` : 90 sütunun açıklaması, durumu, aksiyonu
  - `ORTAK_KARARLAR.md` : ilk kararlar (kapsam kısmı artık GEÇERSİZ, aşağıya bak)
- **Barış'ın defteri `calisma/baris/BIL545_Proje1.ipynb`**: Hazırlık + Bölüm 1, Kübra'nın V2 defterinden birebir kopyalandı (çıktılarıyla, 2026-11-02). Bölüm 2 burada yazılıyor. Önceki Bölüm 1 sürümü git geçmişinde (485905e).
- Kübra'nın defteri (Bölüm 1, ekip ana hattı): github.com/kubragiritkopuz/MachineLearningCourseDokuments → **`BIL545_Proje1_AirbnbIstanbulV2.ipynb`** (eski: `BIL545_Proje1_20261007.ipynb`)
  - V2 değişiklikleri: `model_tablosu(d, senaryo)`; `KAPSAM="hepsi"`, `SENARYO="mevcut_ilan"`; 24 öznitelik (uzun_donem, puan_deger, kaynak çıkarıldı); veri tablosu `hedefli` yerine `veri`; `add_indicator=True` varsayılan; `lr_hatti()` eklendi.
  - İçinde hazır fonksiyonlar: `model_tablosu(d, senaryo)`, `on_isleme(kodlama, add_indicator, ...)`, `lojistik()`, `grup_cv()`, `CV_GRUP`, `CV_SATIR`, `degerlendir()`, `tablo()`.

## Ekip kararları (güncel)
| Konu | Karar |
|---|---|
| Kapsam | **Tüm fiyatlı ilanlar (24.553)**; uzun dönem `uzun_donem` / `log_min_gece` ile işaretli (ölçüm etkisi yorumda yazılacak) |
| Hedef sınırları | **2.750 / 5.100 TL** (üçte birliklere yakın sabit sınırlar) |
| Bölme | `StratifiedGroupKFold(groups=host_id)`, 5 kat × 5 tekrar (tohum 42…46) = `CV_GRUP` |
| Ana ölçüt | **Makro F1** (ikincil: dengeli doğruluk) |
| Ön işleme | Hepsi Pipeline içinde (`on_isleme`); satır içi dönüşümler `model_tablosu`'nda |
| Önerilen (resmî onay bekliyor) | müsaitlik/doluluk/son dönem yorum sütunları çıkarılsın (ters nedensellik), `license` → 4 durum, `reviews_per_month` yorumsuzsa 0. **Not (2026-11-01):** üçü de Kübra'nın `model_tablosu` / `kodlanmis_eksikleri_duzelt` fonksiyonlarında zaten uygulanmış; 2.1 bu hâliyle koşturuldu. Ekipte yalnız kararın kapatılması kaldı |

## Önemli bulgular (Bölüm 1)
- Sızıntı sütunları: `price_quote_*`, `estimated_revenue_l365d` (gelir ÷ doluluk = fiyat, %99,9).
- 1.4: hedef kodlama bütün veride → `license` 0,59→0,81, `host_id` 0,62→0,86 (sahte).
- 1.5: ön işleme sızıntısı ≈ 0; **bölme sızıntısı büyük** (ağaç/orman +0,08–0,10; LR +0,01) ve model sıralamasını tersine çeviriyor.
- 1.6: LR makro F1 ≈ 0,63 (tüm veri) / 0,61 (kısa dönem); Dummy 0,16 / 0,33.
- Kübra'nın defterinde yorumlardaki sayılar çıktıyla uyuşmuyor (1.3, 1.5, 1.6) → düzeltilmeli.
- Weka: ham CSV açılmıyor (metin/URL/kesme işareti); `bufferSize=30000` gerekli. W1–W4 adımları kısa dönem kapsamına göre yapıldı → tüm veri kapsamına göre yenilenecek.

## 2.1 sonuçları (2026-11-02, V2 model tablosu, sklearn 1.9.1; defterde 2.1 hücresi + `calisma/baris/kodlar/b2_1_model_karsilastirma.py`)
Ortak hat (`on_isleme(kodlama="nadir", add_indicator=True)`) + `CV_GRUP` 25 bölme. Hiperparametreler sabit (ayar 2.3'te). Parantez içi: 1.6.1 sonucu.

| Model | Makro F1 | Eğitim − test |
|---|---|---|
| Dummy en sık / oranlarla | 0,169 / 0,335 (0,163 / 0,334) | ≈0 |
| Lojistik regresyon | 0,630 ± 0,011 (0,628) | 0,02 |
| Karar ağacı sınırsız / yaprak ≥ 20 | 0,567 / 0,604 (0,564 / 0,599) | 0,43 / 0,14 |
| k-NN k=15 ölçekli | 0,605 ± 0,010 (0,602) | 0,12 |
| Rastgele orman 300 ağaç | 0,665 ± 0,016 (0,663) | 0,29 |
| HistGradientBoosting | 0,670 ± 0,016 (0,667) | 0,19 |

- RF > LR 25/25 bölmede (+0,035). HistGB − RF = +0,005 ± 0,007 (18/25) → kazanan 2.4'teki testle belirlenecek. k-NN ≈ ağaç ≥ 20 (+0,001, 15/25).
- Ek deney (1.6.1, eski tablo, tek bölme): iki sınırsız ağaç test ilanlarının %44'ünde farklı dilim söylüyor, iki RF %5'inde. RF eğitim F1 0,957 / OOB 0,750 / gruplu test 0,661 → OOB satıra göre olduğu için bu veride şişik (1.5 bölme sızıntısıyla aynı); 2.3'te kullanılabilir.
- Yorum hücresi henüz yazılmadı: Barış yazacak (eğitim–test farkının ağaç vs. RF'de farklı okunması; makro F1 gerekçesi).

## 2.2 sonuçları (2026-11-02, sklearn 1.9.1; defterde 2.2 hücresi + `calisma/baris/kodlar/b2_2_agac_yorumu.py`)
Ölçeksiz hat (`agac_on_isleme`, aynı doldurma/kodlama; skor 0,604 = 2.1) + `min_samples_leaf=20`. Yorum ağacı bütün veride (yalnız yapı için).
- Kök: `log_min_gece` ≤ 20,5 gece (25/25 eğitim katında). Sağ dal (%32,8) neredeyse tamamen 100 gecelik teklifler; medyan fiyat 2.096 TL'ye karşı 4.483 TL → ağacın ilk sorusu evin niteliği değil, teklifin ölçülme biçimi (vekil hedef / 7464 sayılı Kanun etkisi).
- Kök eşiği katlar arasında 14,5–99,5 gece, ama 15–99 aralığında yalnız 298 ilan (%1,2) → ayrım pratikte aynı.
- Sol çocuk: `yatak_odasi` ≤ 1,5 (25/25); 2+ yatak odalı kısa dönem → %77 pahalı. Sağ çocuk kararsız: kapasite 14, oda_tipi Entire 8, Private room 3.
- Bütün veride ağaç: derinlik 22, 857 yaprak.

## 2.3 sonuçları (2026-11-02, sklearn 1.9.1; defterde 2.3 hücresi + `calisma/baris/kodlar/b2_3_optimist_yanlilik.py`)
Karar ağacı, 24 aday (max_depth 4–None × min_samples_leaf 1–50). İç ÇD de gruplu (`StratifiedGroupKFold`). Dış = `CV_GRUP[:5]`. Süre ≈ 4 dk.

| Kaynak | Rapor edilen | Gerçekçi | Şişme |
|---|---|---|---|
| best_score_ (gruplu iç ÇD) vs iç içe ÇD | 0,613 | 0,612 ± 0,015 | +0,001 |
| best_score_ satıra göre ÇD vs seçilen ayarın gruplu skoru | 0,666 | 0,598 | +0,067 |
| RF OOB (100 ağaç) vs gruplu test | 0,743 | 0,660 | +0,083 |

- Doğru bölmeyle ayar seçim yanlılığı çok küçük (24 aday birbirine yakın, 24 bin ilan). 1 adayda iç − dış = −0,003 (iç ÇD daha az veriyle eğitiyor → kötümser), 24 adayda +0,002 → net seçim etkisi ≈ +0,004.
- Asıl şişme yanlış bölmeden: satıra göre ayar derin ağacı seçiyor (depth 12, leaf 1), gruplu ayara göre −0,016 (22/25 bölmede kötü). Dış katlarda seçilen ayar da değişiyor (depth 6/8/12; leaf 5/50) → ayar kararsız.

## 2.4 sonuçları (2026-11-02, sklearn 1.9.1; defterde 2.4 + 2.4 devam hücreleri, `kodlar/b2_4_istatistiksel_test.py`, `b2_4b_duyarlilik.py`)
Gruplu 5×2 ÇD (`StratifiedGroupKFold(2)` × 5, tohum 42–46), McNemar için gruplu ayırma (2/3–1/3, tohum 42). Süre ≈ 5 dk.

| Çift (A − B) | Fark | Saf t (YANLIŞ) | Düzeltilmiş t | 5×2 t | 5×2 F | Ev sahibi bootstrap | McNemar | Sonuç |
|---|---|---|---|---|---|---|---|---|
| HistGB − RF | +0,005 | 0,004 | 0,25 | 0,038 (Holm 0,076) | 0,14 | 0,10 | 0,014 | **anlamlı değil** |
| RF − LR | +0,035 | <0,001 | <0,001 | <0,001 | <0,001 | <0,001 | <0,001 | anlamlı (pozitif kontrol) |
| k-NN − ağaç≥20 | +0,001 | 0,64 | 0,86 | 0,11 | 0,31 | 0,16 | 0,06 | anlamlı değil (negatif kontrol) |

- HistGB − RF: 5×2 t'nin payı keyfi seçilen ilk fark; 10 seçimden yalnız 1'inde p < 0,05 (p 0,038–0,76). Saf t ve ilan bazlı McNemar ev sahibi bağımlılığını yok saydığı için "anlamlı" diyor; düzeltilmiş/kümelenmiş testler demiyor.
- Duyarlılık hücresi (2.4 devam) ilk sonuçlar görüldükten sonra eklendi → raporda post hoc olarak belirtilmeli.

## 2.5 aktarım (2026-11-02, sklearn 1.9.1; defterde 2.5 hücresi + `kodlar/b2_5_aktarim.py`, `belgeler/p1_protokol.json`)
- **Seçilen model: RF** (300 ağaç, yaprak ≥ 2, ayarsız) — Barış kararı. Gerekçe: HistGB − RF anlamlı değil (2.4); RF ayarsız kararlı, açıklanabilirlik için uygun.
- Çıktılar (Colab'da defter üretir, GitHub'a konmaz): `p1_hat.joblib` (37 MB, bütün veride eğitilmiş Pipeline), `p1_temiz_veri.csv` (24.553 × 36: id, host_id, fiyat, minimum_nights, ilce_ham, 24 girdi, fiyat_dilimi, y, **p1_oof_*** kat dışı tahmin/olasılıklar).
- Kat dışı makro F1 (CV_GRUP ilk tur) 0,664; bütün veride eğitim içi 0,955 (raporlanmaz). Geri okuma kontrolü %100.
- Proje 2 uyarısı: hata/alt grup analizleri `p1_oof_*` ile yapılmalı. Ön bakış: Shared room 0,58 · Entire 0,66 · Private 0,69 · Hotel 0,72 doğruluk; ≥100 gece 0,67 / <100 gece 0,66.

## 2.5 hedef tartışması (2026-11-01, `calisma/baris/kodlar/b2_5_hedef_tartismasi.py`)
Barış'ın sorusu: "Keyfi sınırlarla üretilen etiket başarı ölçütü olabilir mi?" Ortak hat + LR, `CV_GRUP[:5]`.
- Hata yapısı: doğru %62,9 · komşu sınıf hatası %31,9 · uç hata (ucuz↔pahalı) %5,2.
- Sınıra uzaklığa göre hata: <%5 → %55,8 · >%40 → %17,4 (hatalar sınırların etrafında yığılıyor).
- Sınır duyarlılığı: makro F1 0,53 (4 sınıf) – 0,63 (ekip) – 0,77 (2 sınıf); Spearman (beklenen sınıf ~ fiyat) her sette ≈0,71; sınırsız log(fiyat) Ridge regresyonu da 0,711 (R² 0,49).
- Sonuç: F1 sınıra bağlı, öğrenilen sinyal değil → raporda F1 tek başına verilmemeli; şans düzeyi + sıralama gücü + bu tablo ile verilmeli. Yorum hücresini Barış yazacak.

## Bölüm 2 planı (sıradaki iş)
- Makale: Dietterich (1998), Neural Computation 10(7) 1895–1923. Özet verildi: 5 test; öneri 5×2 ÇD t-testi (10 kez çalıştırılabiliyorsa) ya da McNemar (bir kez). Türkçesi yok.
- 6. hafta (karar ağacı, NB, k-NN, ölçütler, bölme) Barış'a çalıştırılıyor; 7. hafta: RF, GB, iç içe ÇD, istatistiksel test.
- 2.1 iskeleti Kübra'nın fonksiyonları üzerine kurulacak: Dummy, karar ağacı, k-NN (ölçekli), RF, gradyan artırma; `CV_GRUP`; ort ± std + kutu grafiği.
- ✅ 2.1–2.5 kodları ve yorumları + Dietterich okuma notu tamam (2026-11-05). Barış Bölüm 3, Ek paket 1 ve Ek paket 2'den sorumlu değil (Barış beyanı).
- 2.4'te 5×2 ÇD **gruplu** yapılmalı (`StratifiedGroupKFold(n_splits=2)` × 5); McNemar için gruplu holdout.
- **Yorum hücreleri en sona bırakıldı (Barış kararı, 2026-11-02).** Önce 2.1–2.5 kodları, sonra bütün yorumlar.
