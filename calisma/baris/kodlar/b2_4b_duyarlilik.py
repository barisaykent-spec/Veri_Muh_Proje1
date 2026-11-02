# =====================================================================
# 2.4 (devam) Sonuç ne kadar sağlam? Önkoşul: önceki hücre (skor_5x2, dogru_h, tr_h, te_h, KARSILASTIRMALAR)
# (a) Dietterich'in 5×2 t testi payda 10 farkın hepsini, payda ise YALNIZ ilk farkı p_1^(1) kullanır.
#     Hangi tekrarın "ilk" sayılacağı keyfidir: payı 10 farkın her biriyle değiştirip p'nin oynamasına bakıyoruz.
# (b) McNemar ilanları bağımsız sayar; aynı ev sahibinin ilanları test kümesinde birlikte doğru/yanlış olur.
#     Ev sahibine göre bootstrap (ev sahiplerini yerine koyarak çek) farkın gerçek belirsizliğini verir.
# (c) Üç çift test edildi: ana test için Holm düzeltmesi.
# DÜRÜSTLÜK NOTU: Bu kontroller ilk sonuçlar görüldükten sonra, 5×2 t (p = 0,038) ile 5×2 F (p = 0,14)
# çeliştiği için eklendi. Sonradan yapılmış (post hoc) sağlamlık analizidir; raporda böyle belirtilmeli.
# =====================================================================
rng = np.random.default_rng(SEED)
grup_te = gruplar[te_h]
ev_sahipleri, ters = np.unique(grup_te, return_inverse=True)

duyarlilik = []
for a_, b_ in KARSILASTIRMALAR:
    p = skor_5x2[a_] - skor_5x2[b_]
    s2 = ((p - p.mean(1, keepdims=True)) ** 2).sum(1)
    p_degerleri = np.array([2 * stats.t.sf(abs(pay / np.sqrt(s2.mean())), df=5) for pay in p.ravel()])
    # ev sahibine göre bootstrap: her ev sahibinin ilanlarındaki "A doğru − B doğru" toplamı ve ilan sayısı
    fark_ilan = dogru_h[a_].astype(int) - dogru_h[b_].astype(int)
    toplam = np.bincount(ters, weights=fark_ilan); adet = np.bincount(ters)
    boot = []
    for _ in range(2000):
        sec = rng.integers(0, len(ev_sahipleri), len(ev_sahipleri))
        boot.append(toplam[sec].sum() / adet[sec].sum())
    boot = np.array(boot)
    gozlenen = fark_ilan.mean()
    duyarlilik.append({
        "karşılaştırma (A − B)": f"{a_}  −  {b_}",
        "5×2 t · p (Dietterich, ilk fark)": t_5x2(p)[1],
        "5×2 t · p en küçük": p_degerleri.min(), "5×2 t · p en büyük": p_degerleri.max(),
        "5×2 t · 10 seçimden p<0,05": int((p_degerleri < 0.05).sum()),
        "doğruluk farkı (ayırma)": gozlenen,
        "ev sahibi bootstrap %95 GA alt": np.percentile(boot, 2.5), "üst": np.percentile(boot, 97.5),
        "bootstrap p (iki yönlü)": min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())),
        "McNemar p (ilan bağımsız)": test_tablosu.loc[f"{a_}  −  {b_}", "p · McNemar"],
    })
duyarlilik = pd.DataFrame(duyarlilik).set_index("karşılaştırma (A − B)")

# (c) Holm: ana test (5×2 t) için üç çiftin p değerleri birlikte düzeltilir
p_ana = duyarlilik["5×2 t · p (Dietterich, ilk fark)"].to_numpy()
sira = np.argsort(p_ana); holm = np.empty(3)
holm[sira] = np.maximum.accumulate(np.minimum(1, (3 - np.arange(3)) * p_ana[sira]))
duyarlilik["5×2 t · p (Holm, 3 çift)"] = holm
display(duyarlilik.round(4))
print(f"Ev sahibi bootstrap: test kümesinde {len(ev_sahipleri):,} ev sahibi, {len(te_h):,} ilan; 2.000 tekrar.")

# ---- Karar özeti: bütün kanıtlar yan yana
ozet_24 = pd.DataFrame({
    "fark (makro F1, 25 bölme)": test_tablosu["fark (25 bölme)"],
    "düzeltilmiş t (25)": test_tablosu["p · düzeltilmiş t (25)"],
    "5×2 F": test_tablosu["p · 5×2 F (Alpaydın)"],
    "5×2 t (Holm)": duyarlilik["5×2 t · p (Holm, 3 çift)"],
    "ev sahibi bootstrap": duyarlilik["bootstrap p (iki yönlü)"],
})
ozet_24["sonuç"] = np.where((ozet_24.iloc[:, 1:] < 0.05).all(axis=1), "fark anlamlı (bütün sağlam testler)",
                     np.where((ozet_24.iloc[:, 1:] < 0.05).any(axis=1), "kanıt zayıf / tutarsız", "fark anlamlı değil"))
display(ozet_24.round(4))
