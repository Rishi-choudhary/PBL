# Mentor requests: 5-fold CV, combining the datasets, multi-model

## Decisions

| Request | Worth doing? | What we did |
|---|---|---|
| 1. Evaluate with 5-fold cross-validation | Yes | `14_five_fold_cv.py`: stratified 5-fold CV on **all rows** for the 10 crop and 8 fertilizer algorithms. Experiment 12 only ran 5-fold CV inside the 1,760 training rows (crop) and 5×10 repeated CV (fertilizer). |
| 2. Combine the two datasets, or keep two models | Combine: **no**. Two models: already the case | `15_multimodel_experiment.py` section A shows the datasets cannot be merged honestly (below). We keep two models. |
| 3. Multi-model that suggests crop and fertilizer | Yes, as a **chained pipeline**, not a jointly trained model | `multimodel.py`: `CropFertilizerSystem.recommend()` returns the top-3 crops and fertilizer advice in one call. |

## Why the datasets are not merged

- No record is shared and neither file has a row identifier, so a row-level join would have to invent the pairing.
- Only 5 inputs mean the same thing in both (N, P, K, temperature, humidity). The crop data also has pH and rainfall; the fertilizer data also has moisture, soil type and crop type.
- The value ranges differ a lot (N 0–140 vs 4–42, K 5–205 vs 0–19, humidity 14–100 vs 50–72). A Random Forest tells the two datasets apart from those 5 inputs alone with balanced accuracy 1.0000.
- Only 10 of the 22 crops have a counterpart in the fertilizer data's crop types (maize, cotton, rice = Paddy, 7 pulses = Pulses). 7 of its 11 crop types (wheat, sugarcane, tobacco, …) are not in the crop data.
- Result that matters most: of the 440 held-out crop farmers, **0** lie inside the fertilizer model's training range for all five shared inputs (`tables/tabS14_ml_fertilizer_coverage.csv`, `figures/figS8_dataset_overlap.png`).

## How the multi-model works (`multimodel.py`)

1. **Crop model.** Random Forest (experiment 12's system model) → top-k crops with probabilities.
2. **Fertilizer, route A: nutrient-gap rule** (works for all 22 crops). `gap = nutrient the crop needs − nutrient in the soil`; the largest gap decides: soil short → add urea (N), DAP/SSP (P) or MOP (K); soil has extra → skip it. Same rule as `tabS9`.
3. **Fertilizer, route B: ML product model** (XGBoost on the 99-row dataset). The crop model's top-1 crop becomes the "Crop Type" input. It runs only if (a) soil moisture and soil type are given, (b) the top-1 crop has a fertilizer crop type, and (c) every shared input is inside the model's training range. Otherwise the answer lists which check failed. This guard is there because the model has never seen inputs outside that range.

Run it:

```
cd crop_experiments
python3 multimodel.py --N 90 --P 42 --K 43 --temperature 20.9 --humidity 82 --ph 6.5 --rainfall 203
python3 multimodel.py --N 21 --P 42 --K 19 --temperature 28 --humidity 55 --ph 6.8 --rainfall 50 --moisture 40 --soil-type Sandy
```

The second command is a hand-entered input chosen so that every check passes; it shows route B once. It is not data.

## Run order and outputs

```
python3 14_five_fold_cv.py           # about 30 s  -> tabS11, tabS12, figS7, summary_cv5.txt
python3 15_multimodel_experiment.py  # about 6 s   -> tabS13-S15, figS8, summary_multimodel.txt
```

Experiments 12 and 13 were not changed, and none of their outputs changed. The new scripts import `crop_models()` and `fert_models()` from `12_crop_system.py`. Every number is listed in `results/CLAIMS.md` under "Added after the mentor's review".

## What this does not show

- **No end-to-end accuracy for crop → fertilizer.** No dataset has both labels for the same record, so the chain is checked stage by stage. The multi-model reproduces experiment 12's top-3 crops for 6 of 6 farmers and its nutrient-gap rows for 12 of 12.
- Both datasets are benchmark datasets (see `CLAUDE.MD`); these accuracies compare algorithms and are not evidence for real farms.
- The paper, deck and report have **not** been updated with these results.
