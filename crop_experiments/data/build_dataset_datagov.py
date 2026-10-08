"""
Build a district x year x season crop-outcome table from the data.gov.in
"District-wise, season-wise crop production statistics" (1997-2015,
646 districts, 33 states/UTs, 124 crop names).

Compared with the ICRISAT table (build_dataset.py) this adds seasons, more
districts and more crops, and uses post-2000 district names that are closer to
the LGD names used by Soil Health Card data.

Cleaning
  * strip whitespace; merge synonyms (Paddy->Rice, Blackgram->Urad, Lentil->Masoor,
    Kapas->Cotton(lint)); drop aggregate rows (totals, Jute & mesta)
  * seasons: Kharif, Rabi, Summer, Whole Year; Autumn and Winter are kept as
    their own seasons (they occur in a few eastern states)
  * keep crops that are a major crop (share >= threshold) in at least
    --min-occurrences district-season-years
Targets (per district-year-season, from cropped AREA, not production, because
production units differ across crops, e.g. coconut in nuts)
  share_<crop>, grown_<crop> (share >= threshold), top1_crop, n_major_crops
Context feature: ctx_season_<season> one-hot (known to the farmer at decision time)

Usage
  python3 build_dataset_datagov.py --raw datagov_district_season_crop_production_1997_2015.csv \
      --out build_datagov [--soil soil_district.csv] [--climate climate_district_year.csv]
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

SYNONYMS = {"Paddy": "Rice", "Blackgram": "Urad", "Lentil": "Masoor", "Kapas": "Cotton(lint)",
            "Other  Rabi pulses": "Other Rabi pulses", "Moong(Green Gram)": "Moong"}
AGGREGATES = {"Total foodgrain", "Pulses total", "Oilseeds total", "Jute & mesta"}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.raw)
    for c in ["State_Name", "District_Name", "Season", "Crop"]:
        d[c] = d[c].astype(str).str.strip()
    d["Crop"] = d["Crop"].replace(SYNONYMS)
    d = d[~d["Crop"].isin(AGGREGATES)]
    d = d[d["Area"].notna() & (d["Area"] > 0)]
    d = d.groupby(["State_Name", "District_Name", "Crop_Year", "Season", "Crop"], as_index=False)["Area"].sum()
    tot = d.groupby(["State_Name", "District_Name", "Crop_Year", "Season"])["Area"].transform("sum")
    d["share"] = d["Area"] / tot
    major = d[d["share"] >= a.share_threshold].groupby("Crop").size()
    keep = major[major >= a.min_occurrences].index
    print(f"crops kept as labels: {len(keep)} of {d.Crop.nunique()} (>= {a.min_occurrences} major occurrences)")
    d = d[d["Crop"].isin(keep)]

    share = d.pivot_table(index=["State_Name", "District_Name", "Crop_Year", "Season"], columns="Crop",
                          values="share", aggfunc="sum", fill_value=0.0)
    share.columns = [f"share_{slug(c)}" for c in share.columns]
    grown = (share >= a.share_threshold).astype(int)
    grown.columns = [c.replace("share_", "grown_") for c in grown.columns]
    wide = pd.concat([share, grown], axis=1).reset_index()
    wide = wide.rename(columns={"State_Name": "state", "District_Name": "district", "Crop_Year": "year",
                                "Season": "season"})
    wide["top1_crop"] = share.values.argmax(1)
    wide["top1_crop"] = [share.columns[i].replace("share_", "") for i in wide["top1_crop"]]
    wide["n_major_crops"] = grown.values.sum(1)
    wide = wide[wide["n_major_crops"] >= 1].reset_index(drop=True)
    wide["dist_code"] = wide["state"].map(norm) + "|" + wide["district"].map(norm)
    seasons = pd.get_dummies(wide["season"], prefix="ctx_season").astype(int)
    seasons.columns = [slug(c).replace("ctx_season_", "ctx_season_") for c in seasons.columns]
    wide = pd.concat([wide, seasons], axis=1)

    if a.soil:
        soil = pd.read_csv(a.soil)
        soil["dist_code"] = soil["state"].map(norm) + "|" + soil["district"].map(norm)
        feats = [c for c in soil.columns if c not in ("state", "district", "dist_code")]
        wide = wide.merge(soil[["dist_code"] + feats].rename(columns={c: f"soil_{c}" for c in feats}),
                          on="dist_code", how="left")
        print(f"soil matched for {wide[f'soil_{feats[0]}'].notna().mean():.1%} of rows")
    if a.climate:
        cl = pd.read_csv(a.climate)
        cl["dist_code"] = cl["state"].map(norm) + "|" + cl["district"].map(norm)
        feats = [c for c in cl.columns if c not in ("state", "district", "year", "dist_code")]
        wide = wide.merge(cl[["dist_code", "year"] + feats].rename(columns={c: f"clim_{c}" for c in feats}),
                          on=["dist_code", "year"], how="left")
        print(f"climate matched for {wide[f'clim_{feats[0]}'].notna().mean():.1%} of rows")

    wide.to_csv(out / "crop_wide_district_season_year.csv", index=False)
    wide[["state", "district", "dist_code"]].drop_duplicates().sort_values(["state", "district"]) \
        .to_csv(out / "district_key_datagov.csv", index=False)
    meta = {"rows": int(len(wide)), "districts": int(wide.dist_code.nunique()), "states": int(wide.state.nunique()),
            "years": [int(wide.year.min()), int(wide.year.max())], "seasons": sorted(wide.season.unique().tolist()),
            "labels": sorted(c.replace("share_", "") for c in share.columns), "share_threshold": a.share_threshold}
    (out / "build_meta_datagov.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "labels"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="datagov_district_season_crop_production_1997_2015.csv")
    ap.add_argument("--out", default="build_datagov")
    ap.add_argument("--soil"); ap.add_argument("--climate")
    ap.add_argument("--share-threshold", type=float, default=0.05)
    ap.add_argument("--min-occurrences", type=int, default=200)
    main(ap.parse_args())
