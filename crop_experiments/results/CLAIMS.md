# CLAIMS ledger

This file lists every number in `paper/main.tex` (Crop Recommendation System) and the file it comes from.
All of them are produced by `12_crop_system.py`. Paths are relative to `crop_experiments/results/system/`, and T = `tables/`.

## Crop dataset and analysis

| Claim | Value | Source |
|---|---|---|
| Rows, crops, per crop, missing, duplicates | 2,200 / 22 / 100 / 0 / 0 | `summary.txt` |
| Train / test split | 1,760 / 440, stratified, seed 42 | `summary.txt` |
| Feature mean, std, min, max (Table II) | as printed | T `tabS1_feature_summary.csv` |
| Correlation P–K; other pairs | 0.736; −0.231 to 0.205 | T `tabS1_correlation.csv` |
| Apple, grapes mean P / K; cotton mean N | 134.2/199.9, 132.5/200.1; 117.8 | computed from `data/Crop_recommendation.csv` (group means) |
| Rice vs jute mean N, P, K, humidity, rainfall | 79.9/47.6/39.9, 82.3, 236.2 vs 78.4/46.9/40.0, 79.6, 174.8 | same |
| Blackgram vs maize humidity | 65.1 vs 65.1 | same |

## Algorithm comparison (Table IV, Fig. 3)

| Claim | Value | Source |
|---|---|---|
| Every row of Table IV (test accuracy, precision, recall, F1, errors, CV mean ± sd, training time) | as printed | T `tabS2_crop_model_comparison.csv` |
| Best test accuracy | Random Forest and Naive Bayes, 99.55 % (2 errors each) | T `tabS2_crop_model_comparison.csv` |
| AdaBoost default / depth-3 | 17.50 % / 99.32 % | T `tabS2_crop_model_comparison.csv` |
| Top-5 CV range | 98.75–99.49 % | T `tabS2_crop_model_comparison.csv` |
| RF errors | rice→jute 1, blackgram→maize 1 | `summary.txt` |
| RF per-crop F1 < 1 | blackgram 0.974, jute 0.976, maize 0.976, rice 0.974 | T `tabS3_per_crop_best.csv` |
| Six farmers' top-3 crops and probabilities (Table V) | as printed | T `tabS4_farmer_examples.csv` |

## Explainable AI (Figs. 5, 6)

| Claim | Value | Source |
|---|---|---|
| Global SHAP shares | humidity 21.8, rainfall 18.1, K 17.3, N 17.0, P 16.1, temperature 6.6, pH 3.1 % | T `tabS5_shap_global.csv` |
| Top feature per crop counts | rainfall 8, humidity 7, P 3, N 2, K 2 | `summary.txt`, T `tabS5_shap_per_crop.csv` |
| Farmer F1: mango p = 0.98; SHAP humidity 0.23, rainfall 0.20; LIME rule humidity ≤ 60.2 | as printed | T `tabS6_farmer_explanation.csv`, `summary.txt` |
| SHAP vs LIME, 100 test rows | same top feature 0.66, top-3 overlap 0.78 | T `tabS6_shap_lime_agreement.csv` |

## Fertilizer (Table VI, Fig. 7, Table VII)

| Claim | Value | Source |
|---|---|---|
| Rows, class counts | 99; Urea 22, DAP 18, 28-28 17, 14-35-14 14, 20-20 14, 17-17-17 7, 10-26-26 7 | `summary.txt` |
| Every row of Table VI | as printed | T `tabS7_fertilizer_model_comparison.csv` |
| Permutation importance P, N, K; others | 0.438, 0.314, 0.261; ≤ 0.006 in absolute value | T `tabS8_fertilizer_importance.csv` |
| XGBoost one 5-fold run | 97/99 correct (0.9798); errors 10-26-26→14-35-14, 17-17-17→20-20 | `summary.txt`, `figures/figS6_fertilizer.png` |
| Nutrient-gap rows (Table VII) | as printed | T `tabS9_nutrient_gap_examples.csv` |

## Not in this paper

The real district data experiments (07/08, soil and climate) have not been run. They are mentioned only as future work.
The ledger for the earlier draft (benchmark diagnosis) is in git history, commit 2b93679.
