# Devir notu — BIL545 Proje 1 (Barış ile çalışma oturumu özeti)

Son güncelleme: 2026-11-01 (2.1 eklendi). Yeni bir sohbet açılınca Claude'un önce bu dosyayı okuması yeterli.

## Ders ve proje
- BIL545 Veri Madenciliğinde İleri Konular (Dr. Selma Güzel, Üsküdar Ü.), 2026–27 Güz. 5 kişilik grup → temel kapsam + Ek paket 1 + Ek paket 2.
- Konu 9.19 İstanbul Airbnb (Inside Airbnb, anlık görüntü **2026-09-25**, 27.232 × 90). Hedef: **gecelik fiyat dilimi** (ucuz/orta/pahalı).
- Puanlama: başarım değil kurulumun doğruluğu (%30), gerekçe (%25), dürüst raporlama (%25), yorum (%15), teknik düzen (%5). Sızıntı = kurulum puanı sıfır.
- Barış'ın sorumluluğu: **Bölüm 2 (sınıflandırma ve dürüst değerlendirme)**. Bölüm 1'i Kübra yazdı (aşağıda).

## Barış hakkında / çalışma biçimi
- Finans altyapılı analist; tablo sever; SQL/select alışkanlığı var. Türkçe, isimle hitap ("Barış").
- **Öğrenmek istiyor**: adım adım, önce kavram, sonra kod; Python + Weka paralel. Kodu verirken satır satır açıklama + yorum taslağı istiyor.
- Colab ortamı: Python 3.13, pandas 2.2.3, **scikit-learn 1.6.1** (sürüm farkı sonuçları değiştiriyor; test ederken bu sürümü kullan).

## Depolar ve dosyalar
- Grup deposu: github.com/barisaykent-spec/Veri_Muh_Proje1 (public). Barış'ın denemeleri `calisma/baris/`.
  - `calisma/baris/kodlar/` : adim2_3 (sütun rolleri), b1_3a–d (eksiklik), b1_4 (kodlama), b1_5 (sızıntı), b1_6 (baseline), w1 (Weka CSV), b2_1 (model karşılaştırması)
  - `calisma/baris/belgeler/sutun_sozlugu.md|csv` : 90 sütunun açıklaması, durumu, aksiyonu
  - `ORTAK_KARARLAR.md` : ilk kararlar (kapsam kısmı artık GEÇERSİZ, aşağıya bak)
- Kübra'nın defteri (Bölüm 1, ekip ana hattı): github.com/kubragiritkopuz/MachineLearningCourseDokuments → `BIL545_Proje1_20261007.ipynb`
  - İçinde hazır fonksiyonlar: `model_tablosu(d)`, `on_isleme(kodlama, add_indicator, ...)`, `lojistik()`, `grup_cv()`, `CV_GRUP`, `CV_SATIR`, `degerlendir()`, `tablo()`.

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

## 2.1 sonuçları (2026-11-01, sklearn 1.6.1, `calisma/baris/kodlar/b2_1_model_karsilastirma.py`)
Kübra'nın ortak hattı (`on_isleme(kodlama="nadir", add_indicator=True)`) + `CV_GRUP` 25 bölme. Hiperparametreler sabit (ayar 2.3'te).

| Model | Makro F1 | Eğitim − test |
|---|---|---|
| Dummy en sık / oranlarla | 0,163 / 0,334 | ≈0 |
| Lojistik regresyon | 0,630 ± 0,013 | 0,02 |
| Karar ağacı sınırsız / yaprak ≥ 20 | 0,565 / 0,599 | 0,44 / 0,15 |
| k-NN k=15 ölçekli | 0,602 ± 0,013 | 0,12 |
| Rastgele orman 300 ağaç | 0,665 ± 0,014 | 0,29 |
| HistGradientBoosting | 0,670 ± 0,014 | 0,19 |

- RF > LR 25/25 bölmede (+0,035). HistGB − RF = +0,005 (std 0,007, 19/25) → kazanan 2.4'teki testle belirlenecek.
- Yorum hücresi henüz yazılmadı: Barış yazacak (eğitim–test farkının ağaç vs. RF'de farklı okunması; makro F1 gerekçesi).

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
- 2.4'te 5×2 ÇD **gruplu** yapılmalı (`StratifiedGroupKFold(n_splits=2)` × 5); McNemar için gruplu holdout.
- Yorumlar ancak kararlar kesinleşince yazılacak.
