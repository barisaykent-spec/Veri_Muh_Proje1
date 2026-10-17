"""Inside Airbnb İstanbul listings.csv.gz → Weka'ya hazır ARFF.

ORTAK_KARARLAR.md'deki kararları uygular: minimum_nights < 30, fiyatı boş
satırlar dışarıda, hedef = üçte birlik fiyat dilimi, sızıntı / kimlik / metin /
boş sütunlar dışarıda. Eksik değerler ARFF'te '?' olarak yazılır.

Kullanım:  python araclar/weka_arff.py listings.csv.gz airbnb_ist_kisa.arff
"""
import sys
import numpy as np
import pandas as pd

SAYISAL = [
    "accommodates", "bedrooms", "beds", "banyo_sayisi", "olanak_sayisi",
    "latitude", "longitude", "minimum_nights", "maximum_nights",
    "availability_30", "availability_60", "availability_90", "availability_365",
    "number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d",
    "reviews_per_month", "review_scores_rating", "review_scores_accuracy",
    "review_scores_cleanliness", "review_scores_checkin",
    "review_scores_communication", "review_scores_location", "review_scores_value",
    "host_listings_count", "calculated_host_listings_count",
    "hosts_time_as_host_years",
]
KATEGORIK = [
    "room_type", "property_type", "neighbourhood_cleansed", "banyo_tipi",
    "host_is_superhost", "host_has_profile_pic", "host_identity_verified",
]
HEDEF = "fiyat_dilimi"
SINIRLAR = [3455.50, 5877.67]          # ORTAK_KARARLAR.md, karar 4
SINIFLAR = ["ucuz", "orta", "pahali"]


def hazirla(yol):
    df = pd.read_csv(yol, low_memory=False)
    df = df[(df["minimum_nights"] < 30) & df["price_quote_price_per_night"].notna()].copy()

    bt = df["bathrooms_text"].fillna("").str.lower()
    df["banyo_sayisi"] = pd.to_numeric(bt.str.extract(r"([\d.]+)")[0], errors="coerce")
    df.loc[bt.str.contains("half"), "banyo_sayisi"] = 0.5
    df["banyo_tipi"] = np.select([bt.str.contains("shared"), bt.str.contains("private")],
                                 ["paylasimli", "ozel"], "standart")
    df.loc[bt == "", "banyo_tipi"] = np.nan
    df["olanak_sayisi"] = df["amenities"].str.count('"') // 2

    fiyat = df["price_quote_price_per_night"]
    df[HEDEF] = pd.cut(fiyat, [-np.inf, *SINIRLAR, np.inf], right=False, labels=SINIFLAR).astype(str)
    return df[SAYISAL + KATEGORIK + [HEDEF]]


def tirnak(v):
    v = str(v).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{v}'"


def arff_yaz(df, yol):
    with open(yol, "w", encoding="utf-8") as f:
        f.write("% Inside Airbnb Istanbul 2026-09-25 | minimum_nights<30 | CC BY 4.0\n")
        f.write("@relation airbnb_istanbul_kisa_donem\n\n")
        for c in SAYISAL:
            f.write(f"@attribute {c} numeric\n")
        for c in KATEGORIK:
            degerler = sorted(df[c].dropna().astype(str).unique())
            f.write(f"@attribute {c} {{{','.join(tirnak(v) for v in degerler)}}}\n")
        f.write(f"@attribute {HEDEF} {{{','.join(SINIFLAR)}}}\n\n@data\n")
        for satir in df.itertuples(index=False):
            parca = []
            for c, v in zip(df.columns, satir):
                if pd.isna(v):
                    parca.append("?")
                elif c in SAYISAL:
                    parca.append(repr(float(v)) if not float(v).is_integer() else str(int(v)))
                elif c == HEDEF:
                    parca.append(v)
                else:
                    parca.append(tirnak(v))
            f.write(",".join(parca) + "\n")


if __name__ == "__main__":
    veri = hazirla(sys.argv[1])
    arff_yaz(veri, sys.argv[2])
    print(veri.shape, veri[HEDEF].value_counts().to_dict())
