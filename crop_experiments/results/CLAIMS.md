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

## Algorithm comparison (Table IV held-out columns)

| Claim | Value | Source |
|---|---|---|
| Every row of Table IV (test accuracy, precision, recall, F1, errors, CV mean ± sd, training time) | as printed | T `tabS2_crop_model_comparison.csv` |
| Best test accuracy | Random Forest and Naive Bayes, 99.55 % (2 errors each) | T `tabS2_crop_model_comparison.csv` |
| AdaBoost default / depth-3 | 17.50 % / 99.32 % | T `tabS2_crop_model_comparison.csv` |
| Top-5 CV range on the training set (earlier draft; the paper now uses 5-fold CV on all rows, 99.14–99.55 %) | 98.75–99.49 % | T `tabS2_crop_model_comparison.csv` |
| RF errors | rice→jute 1, blackgram→maize 1 | `summary.txt` |
| RF per-crop F1 < 1 | blackgram 0.974, jute 0.976, maize 0.976, rice 0.974 | T `tabS3_per_crop_best.csv` |
| Six farmers' top-3 crops and probabilities (Table V) | as printed | T `tabS4_farmer_examples.csv` |

## Overfitting check (PBL-3 deck slide 19; not in the paper; `13_overfitting_check.py`)

| Claim | Value | Source |
|---|---|---|
| Train / test / gap: RF, NB, XGBoost, LR, DT | 100.00/99.55/0.45, 99.49/99.55/−0.06, 100.00/99.09/0.91, 97.39/97.27/0.11, 100.00/97.95/2.05 | T `tabS10_train_vs_test.csv` |
| Tree depth curve, train vs test | e.g. depth 8: 89.49 / 87.05; unlimited: 100.00 / 97.95 | T `tabS10_tree_depth.csv` |
| RF shuffled labels (100 permutations) | mean 4.61 %, max 5.85 %, chance 4.55 %, p = 0.0099 | T `tabS10_permutation_test.csv` |

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

## Added after the mentor's review (in the paper: Table IV, Table VI, Fig. 3, Fig. 8, Sections III-E, III-H, IV-F; not yet in the deck or report)

Produced by `14_five_fold_cv.py`, `15_multimodel_experiment.py` and `multimodel.py`. Paths are relative to `results/system/`, T = `tables/`.

| Claim | Value | Source |
|---|---|---|
| 5-fold protocol | stratified, shuffled, seed 42, all 2,200 crop rows and all 99 fertilizer rows; every row tested once; population sd | `summary_cv5.txt` |
| Crop, 5-fold accuracy mean ± sd | Random Forest 99.55 ± 0.32 (best), Naive Bayes 99.45 ± 0.18, LightGBM 99.32 ± 0.32, XGBoost 99.32 ± 0.35, AdaBoost depth-3 99.14 ± 0.42, Decision Tree 98.77 ± 0.68, SVM 98.36 ± 0.44, KNN 97.14 ± 0.68, Logistic Regression 97.09 ± 0.22, AdaBoost default 25.41 ± 9.19 | T `tabS11_crop_5fold_summary.csv`, per fold in T `tabS11_crop_5fold_per_fold.csv` |
| Random Forest out of fold | 10 of 2,200 rows wrong: rice→jute 7, mothbeans→lentil 1, blackgram→maize 1, jute→rice 1 | `summary_cv5.txt` |
| Fertilizer, 5-fold accuracy mean ± sd | XGBoost 98.00 ± 2.45, Decision Tree 98.00 ± 2.45, Logistic Regression 97.00 ± 4.00, Random Forest 96.00 ± 4.90, SVM 86.84 ± 4.09, AdaBoost depth-3 86.74 ± 12.35, KNN 73.74 ± 4.87, Naive Bayes 64.63 ± 10.03 | T `tabS12_fert_5fold_summary.csv` |
| Shared features, ranges (crop / fertilizer) | N 0–140 / 4–42; P 5–145 / 0–42; K 5–205 / 0–19; temperature 8.8–43.7 / 25–38; humidity 14.3–100 / 50–72 | T `tabS13_shared_feature_ranges.csv` |
| Crop names with a fertilizer crop type | 10 of 22 crops (maize, cotton, rice = Paddy, 7 pulses = Pulses); 4 of 11 fertilizer crop types have a crop-dataset counterpart | T `tabS13_crop_name_map.csv`, `summary_multimodel.txt` |
| Which dataset did a row come from? | Random Forest on the 5 shared features, 5-fold CV: balanced accuracy 1.0000, ROC AUC 1.0000 | T `tabS13_source_classifier.csv` |
| Held-out crop farmers inside the fertilizer data's range | N 235, P 170, K 95, temperature 248, humidity 107, all five 0, of 440; top-1 crop has a fertilizer crop type 199 of 440 | T `tabS14_ml_fertilizer_coverage.csv` |
| Multimodel reproduces experiment 12 | top-3 crops identical for 6 of 6 farmers (tabS4); nutrient-gap rows identical for 12 of 12 (tabS9) | `summary_multimodel.txt`, T `tabS15_multimodel_six_farmers.csv` |
| End-to-end accuracy of crop → fertilizer | NOT AVAILABLE: no dataset has both a crop label and a fertilizer label for the same record | – |
| Two hand-entered inputs in `summary_multimodel.txt` (section D) | typed by us to show the code path; not data, not a result | – |

## Not in this paper

The real district data experiments (07/08, soil and climate) have not been run. They are mentioned only as future work.
The ledger for the earlier draft (benchmark diagnosis) is in git history, commit 2b93679.
