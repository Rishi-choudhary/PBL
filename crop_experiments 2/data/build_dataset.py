"""
Build a district-level, real-world crop recommendation dataset for India.

Layer 1 (this script, always runs): crop outcomes from the ICRISAT-TCI District
Level Database (apportioned, 1966 district boundaries), 1978-2017, 311 districts,
20 states. Produces area shares, yields, relative yields and multi-label
"major crop" targets per district-year.

Layer 2 (optional, runs if files are supplied): district soil features from the
Soil Health Card nutrient data, and district climate (rainfall, temperature,
humidity) from any district-year table. Joined on a harmonised district key.

Usage:
  python3 build_dataset.py --icrisat "ICRISAT-District Level Data.csv" \
      [--soil soil_district.csv] [--climate climate_district_year.csv] \
      --out ./out --share-threshold 0.05

Soil CSV expected columns (any extra columns are kept):
  state, district, <feature columns...>      (one row per district)
Climate CSV expected columns:
  state, district, year, <feature columns...> (one row per district-year)
District names are matched after normalisation; unmatched names are written
to out/unmatched_*.csv so they can be fixed in a mapping file.
"""
import argparse, re, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

# Crops kept as recommendation labels. Aggregates (SORGHUM = kharif+rabi,
# OILSEEDS, FRUITS AND VEGETABLES, FODDER) are excluded to avoid double counting.
CROPS = [
    "RICE", "WHEAT", "KHARIF SORGHUM", "RABI SORGHUM", "PEARL MILLET", "MAIZE",
    "FINGER MILLET", "BARLEY", "CHICKPEA", "PIGEONPEA", "MINOR PULSES",
    "GROUNDNUT", "SESAMUM", "RAPESEED AND MUSTARD", "SAFFLOWER", "CASTOR",
    "LINSEED", "SUNFLOWER", "SOYABEAN", "SUGARCANE", "COTTON", "POTATOES",
    "ONION", "FRUITS", "VEGETABLES",
]
AREA_ONLY = {"POTATOES", "ONION", "FRUITS", "VEGETABLES"}  # no yield column in DLD


def norm_name(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def load_icrisat(path):
    df = pd.read_csv(path)
    df = df.replace(-1, np.nan)  # DLD uses -1 as missing sentinel
    df = df.rename(columns={"Dist Code": "dist_code", "Year": "year",
                            "State Code": "state_code", "State Name": "state",
                            "Dist Name": "district"})
    return df


def long_table(df):
    rows = []
    for c in CROPS:
        a = f"{c} AREA (1000 ha)"
        if a not in df.columns:
            continue
        p = f"{c} PRODUCTION (1000 tons)"
        y = f"{c} YIELD (Kg per ha)"
        sub = df[["dist_code", "year", "state", "district", a]].copy()
        sub["crop"] = c
        sub = sub.rename(columns={a: "area_kha"})
        sub["prod_kt"] = df[p] if p in df.columns else np.nan
        sub["yield_kg_ha"] = df[y] if y in df.columns else np.nan
        rows.append(sub)
    lt = pd.concat(rows, ignore_index=True)
    lt = lt[lt["area_kha"].notna()]
    # gross cropped area proxy = sum of listed crop areas in that district-year
    gca = lt.groupby(["dist_code", "year"])["area_kha"].sum().rename("gca_proxy_kha")
    lt = lt.join(gca, on=["dist_code", "year"])
    lt["area_share"] = lt["area_kha"] / lt["gca_proxy_kha"].replace(0, np.nan)
    # relative yield vs. state-year median for that crop (suitability signal)
    med = lt.groupby(["state", "year", "crop"])["yield_kg_ha"].transform("median")
    lt["rel_yield"] = lt["yield_kg_ha"] / med.replace(0, np.nan)
    return lt


def wide_targets(lt, thr):
    share = lt.pivot_table(index=["dist_code", "year", "state", "district"],
                           columns="crop", values="area_share", aggfunc="first")
    share.columns = [f"share_{c.lower().replace(' ', '_')}" for c in share.columns]
    share = share.fillna(0.0)
    labels = (share.values >= thr).astype(int)
    lab = pd.DataFrame(labels, index=share.index,
                       columns=[c.replace("share_", "grown_") for c in share.columns])
    top1 = share.idxmax(axis=1).str.replace("share_", "", regex=False).rename("top1_crop")
    n_major = lab.sum(axis=1).rename("n_major_crops")
    wide = pd.concat([share, lab, top1, n_major], axis=1).reset_index()
    return wide


def attach_soil(wide, soil_path, out):
    soil = pd.read_csv(soil_path)
    soil["_k"] = soil["state"].map(norm_name) + "|" + soil["district"].map(norm_name)
    wide["_k"] = wide["state"].map(norm_name) + "|" + wide["district"].map(norm_name)
    feats = [c for c in soil.columns if c not in ("state", "district", "_k")]
    merged = wide.merge(soil[["_k"] + feats].rename(columns={c: f"soil_{c}" for c in feats}),
                        on="_k", how="left")
    um = wide.loc[~wide["_k"].isin(soil["_k"]), ["state", "district"]].drop_duplicates()
    um.to_csv(out / "unmatched_soil_districts.csv", index=False)
    print(f"soil: matched {wide['_k'].isin(soil['_k']).mean():.1%} of district-years; "
          f"{len(um)} districts unmatched -> unmatched_soil_districts.csv")
    return merged.drop(columns="_k")


def attach_climate(wide, clim_path, out):
    cl = pd.read_csv(clim_path)
    cl["_k"] = cl["state"].map(norm_name) + "|" + cl["district"].map(norm_name)
    wide["_k"] = wide["state"].map(norm_name) + "|" + wide["district"].map(norm_name)
    feats = [c for c in cl.columns if c not in ("state", "district", "year", "_k")]
    merged = wide.merge(cl[["_k", "year"] + feats].rename(columns={c: f"clim_{c}" for c in feats}),
                        on=["_k", "year"], how="left")
    um = wide.loc[~wide["_k"].isin(cl["_k"]), ["state", "district"]].drop_duplicates()
    um.to_csv(out / "unmatched_climate_districts.csv", index=False)
    print(f"climate: matched {merged[[f'clim_{feats[0]}']].notna().mean().iloc[0]:.1%} of district-years")
    return merged.drop(columns="_k")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icrisat", required=True)
    ap.add_argument("--soil")
    ap.add_argument("--climate")
    ap.add_argument("--out", default="out")
    ap.add_argument("--share-threshold", type=float, default=0.05)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df = load_icrisat(a.icrisat)
    lt = long_table(df)
    wide = wide_targets(lt, a.share_threshold)
    if a.soil:
        wide = attach_soil(wide, a.soil, out)
    if a.climate:
        wide = attach_climate(wide, a.climate, out)

    lt.to_csv(out / "crop_long_district_year_crop.csv", index=False)
    wide.to_csv(out / "crop_wide_district_year.csv", index=False)
    df[["dist_code", "state_code", "state", "district"]].drop_duplicates() \
        .sort_values(["state", "district"]).to_csv(out / "district_key.csv", index=False)

    meta = {
        "icrisat_rows": int(len(df)), "districts": int(df["dist_code"].nunique()),
        "states": int(df["state"].nunique()), "years": [int(df.year.min()), int(df.year.max())],
        "crops": CROPS, "share_threshold": a.share_threshold,
        "long_rows": int(len(lt)), "wide_rows": int(len(wide)),
        "soil_attached": bool(a.soil), "climate_attached": bool(a.climate),
    }
    (out / "build_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
