# District-level crop recommendation dataset (India) — data card, v0.2

Built 2026-10-08. Status: crop-outcome layer complete; soil and climate layers pending input files.

## 1. What this is

A real-world replacement for the Kaggle "Crop Recommendation" benchmark (2,200 rows, 100 per class, 7 features). Instead of one synthetic-looking row per crop sample, each row is a **district-year** with the crops actually grown there, their area shares and yields, to be joined with district soil chemistry and climate. This makes three things possible that the Kaggle benchmark cannot support: spatial hold-out (leave-state-out), temporal hold-out (train ≤2010, test 2011–2017), and multi-label / top-k recommendation on an imbalanced label distribution.

## 2. Sources and provenance

| Layer | Source | Coverage | Licence / access | Status |
|---|---|---|---|---|
| Crop area, production, yield | ICRISAT–TCI District Level Database (DLD), apportioned to 1966 district boundaries. Citation: ICRISAT, Meso level data for India, doi:10.21421/D2/XFB1BZ | 311 districts, 20 states, 1978–2017, 29 crop columns (25 used) | Open access (DLD terms of use) | **Done** — file `ICRISAT_district_level_data.csv` (12,418 rows). Obtained from a public GitHub mirror of the DLD export; must be re-verified against the DLD portal before publication. |
| Soil chemistry | Soil Health Card scheme, MoAFW. District-level: India Data Portal resource `024cf507-4281-4c89-a40e-37b5add3a4df` (village-level, 2023–25, 1.2 GiB, ODC-BY) or soilhealth.dac.gov.in nutrient dashboard district exports. Point-level with lat/lon per state: dataful.in (e.g. dataset 24083 Punjab, 199,881 points, N/P/K/S/OC/pH/EC/B/Fe/Zn/Cu/Mn) | 738 districts (2023–25) | ODC-BY (IDP); dataful terms vary | **Pending** — supply `soil_district.csv` |
| Climate | Monthly rainfall from DLD rainfall table (same portal), or IMD gridded rainfall; temperature/humidity from NASA POWER (district centroid, free API) | district-year | Open | **Pending** — supply `climate_district_year.csv` |

## 3. Files produced

- `out/crop_long_district_year_crop.csv` — 309,647 rows: dist_code, year, state, district, crop, area_kha, prod_kt, yield_kg_ha, gca_proxy_kha, area_share, rel_yield.
- `out/crop_wide_district_year.csv` — 12,392 rows: one per district-year; `share_<crop>` (25 cols), `grown_<crop>` binary labels (25 cols), `top1_crop`, `n_major_crops`.
- `out/district_key.csv` — 311 districts with codes and state, for matching soil/climate names.
- `build_meta.json` — build parameters.

## 4. Target construction

- `gca_proxy_kha` = sum of area across the 25 kept crops in that district-year (aggregate columns SORGHUM, OILSEEDS, FRUITS AND VEGETABLES, FODDER excluded to avoid double counting). This is a proxy for gross cropped area, not the official GCA.
- `area_share` = crop area / gca_proxy.
- `grown_<crop>` = 1 if area_share ≥ 0.05 (threshold is a parameter; report sensitivity at 0.03 and 0.10).
- `top1_crop` = crop with the largest share (single-label variant for comparison with the literature).
- `rel_yield` = yield / median yield of that crop in the same state and year. Intended as a suitability signal; heavy right tail (max 170) from tiny-area crops — winsorise or require area ≥ 1 kha before use.

## 5. Label distribution (threshold 0.05)

rice 71.3%, wheat 57.0%, minor pulses 48.2%, maize 28.0%, chickpea 23.9%, kharif sorghum 19.3%, pearl millet 19.1%, cotton 18.0%, vegetables 17.1%, groundnut 15.9%, rapeseed-mustard 15.3%, fruits 12.9%, sugarcane 12.0%, soyabean 11.6%, pigeonpea 9.6%, rabi sorghum 8.8%, finger millet 8.4%, sesamum 5.0%, barley 4.1%, potatoes 3.8%, castor 2.6%, sunflower 2.5%, linseed 1.4%, safflower 0.8%, onion 0.3%. Mean 4.17 major crops per district-year (min 1, max 10).

## 6. Known limitations (state these in the paper)

1. **Soil is a snapshot.** SHC data are 2015+ (bulk 2023–25); joining them to 1978–2017 crop records assumes district soil chemistry is approximately stationary. Mitigation: restrict the modelling window to 2010–2017, and report results for the full window only as a robustness check.
2. **Crop choice ≠ crop suitability.** Area shares reflect markets, irrigation, policy (MSP) and habit, not only agronomy. The label is "crops farmers in this district actually grow at scale", which is what a recommender is trained to reproduce anyway, but the paper must say so.
3. **1966 boundaries.** DLD apportions new districts back to parent districts; SHC uses current (LGD) districts. Many-to-one mapping is required; `district_key.csv` + an explicit mapping file handles this.
4. **Missing values.** DLD uses −1 for missing; converted to NaN. Minor crops are frequently missing in early years.
5. **Mirror provenance.** The ICRISAT file was taken from a GitHub mirror; re-download from data.icrisat.org/dld and diff before submission.
6. Fruits, vegetables, potatoes, onion have area but no yield in the DLD export.

## 7. Comparison with the Kaggle benchmark

| | Kaggle (Ingle) | This dataset |
|---|---|---|
| Rows | 2,200 | 12,392 district-years (309,647 crop rows) |
| Unit | synthetic-looking sample | real district-year |
| Labels | 22, exactly 100 each | 25, 245:1 imbalance, multi-label |
| Features | 7 | soil (up to 12 SHC parameters) + climate + spatial/temporal keys |
| Spatial/temporal CV possible | No | Yes |
| Provenance documented | No | Yes (this card) |

## 8. Second outcome source: data.gov.in district-season production (added v0.2)

- Source: "District-wise, season-wise crop production statistics", data.gov.in (Directorate of Economics & Statistics). 246,091 rows, 1997–2015, 33 states/UTs, 646 districts, 124 crop names. File obtained from the Harvestify repository (commit aac427c, Data-raw/raw_districtwise_yield_data.csv); re-download from data.gov.in before publication. The same 246,091-record table is used by Senapaty et al. (Agriculture 14:1256, 2024).
- Built by `build_dataset_datagov.py`: synonyms merged (Paddy→Rice, Blackgram→Urad, Lentil→Masoor, Kapas→Cotton(lint)); aggregates dropped (Total foodgrain, Pulses total, Oilseeds total, Jute & mesta); labels from cropped area share within a district-season-year; crops kept if a major crop (share ≥ 0.05) in ≥ 200 district-season-years → 48 labels.
- Output: 34,940 district-season-year rows, 652 state|district keys, seasons Kharif 9,127 / Rabi 8,678 / Whole Year 7,810 / Summer 4,755 / Winter 2,302 / Autumn 2,268. Season enters the models as a context one-hot (`ctx_season_*`).
- Limitations: state coverage varies by year (22–33 states; 2015 only 2 states, excluded). "Whole Year" is used for different crops in different states. Area, not production, is used for labels because production units differ (coconut in nuts).

## 9. Provenance of the Kaggle benchmark (added v0.2)

The Kaggle crop-recommendation dataset was built by its creator from a 22-row crop → N, P, K table plus bounded random integer offsets (N ±20, P −5..+20, K ±5), with temperature/humidity/pH/rainfall copied from an undocumented file. Verified for 2,200/2,200 rows. Evidence and licence in `provenance/harvestify_aac427c/`. Use the benchmark only to diagnose evaluation practice, not as agronomic data.
