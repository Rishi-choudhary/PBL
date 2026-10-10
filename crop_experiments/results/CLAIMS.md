# CLAIMS ledger

Every number in `paper/main.tex` and the file it comes from. Paths are relative to `crop_experiments/results/`.
K = `kaggle/tables/`, F = `fertilizer/tables/`.

## Kaggle crop dataset

| Claim in paper | Value | Source |
|---|---|---|
| Rows, classes, per class | 2,200 / 22 / 100 | `kaggle/eda_summary.txt` |
| Duplicates, missing values | 0 / 0 | `kaggle/eda_summary.txt` |
| Crop–feature cells consistent with normal | 0 (0.0 for every feature) | `kaggle/eda_summary.txt` |
| Share consistent with uniform, per feature | 0.818–1.000 | `kaggle/eda_summary.txt` |
| Best model, accuracy | Stacking 99.57 ± 0.26 % | K `tab02_benchmark_paper.csv` |
| Every row of Table II (accuracy, macro-F1, top-3, log-loss, ECE, fit time, Holm p) | as printed | K `tab02_benchmark_paper.csv` |
| Models not significantly worse than best | 11 of the other 15 (4 are significantly worse) | K `tab02_significance_vs_best.csv` (`significantly_worse_than_best`) |
| Significantly worse | SVM (p=0.003), LR, KNN, MLP (p<0.001) | K `tab02_significance_vs_best.csv` |
| Gaussian NB | 99.47 %, log-loss 0.015, ECE 0.007, fit 0.0 s | K `tab02_benchmark_paper.csv` |
| AdaBoost d=3 | 99.19 %, log-loss 3.079, ECE 0.946 | K `tab02_benchmark_paper.csv` |
| Stump alone | 0.0909 | K `tab03_adaboost_diagnosis.csv` |
| AdaBoost + stump, 50 / 500 rounds | 0.2541 / 0.1982 | K `tab03_adaboost_diagnosis.csv` |
| AdaBoost + depth-3 | 0.9914 | K `tab03_adaboost_diagnosis.csv` |
| SVM alone (unscaled) / boosted | 0.9782 / 0.8845 | K `tab03_adaboost_diagnosis.csv` |
| Gaussian NB alone / boosted | 0.9945 / 0.9782 | K `tab03_adaboost_diagnosis.csv` |
| Train–test gap: NB, RF, CatBoost, KNN | 0.06, 0.50, 0.50, 2.45 pp | K `tab09_train_vs_test_gap.csv` |
| Permutation test: real, mean, max, p | 0.9959, 0.0463, 0.0605, 0.0099 (100 perms) | K `tab09_permutation_test.csv` |
| Rows consistent with generator | 2200 / 2200 | K `tab09_generator_check.csv` |
| Distinct N, P, K offsets | 41, 26, 11 | K `tab09_generator_check.csv` |
| Lookup accuracy (N, P, K only) | 0.6582 | K `tab09_generator_check.csv` |
| RF on N, P, K only | 0.6500 | K `tab09_generator_check.csv` |
| RF macro-F1 at 8 rows per crop | 0.9741 | K `tab04_learning_curve.csv` |
| Noise 0.2 sd: LightGBM, RF, SVM | 0.7727, 0.8722, 0.9468 | K `tab04_noise_robustness.csv` |
| RF climate-only / soil-only | 0.9185 / 0.7880 | K `tab04_feature_ablation.csv` |
| RF all features / without rainfall | 0.9950 / 0.9673 | K `tab04_feature_ablation.csv` |
| SHAP Kendall's W across splits | 0.9657 | K `tab05_explanation_stability.csv` |
| Top feature humidity in every split | top1_agreement_share 1.0 | K `tab05_explanation_stability.csv` |
| SHAP vs LIME: top-3 overlap, same top, Spearman, sign | 0.7633, 0.8000, 0.6489, 0.9767 | K `tab10_shap_lime_disagreement.csv` |
| Deletion, P(pred) after removing top-1: none/SHAP/LIME/random | 0.9871 / 0.0801 / 0.2004 / 0.7799 | K `tab10_deletion_faithfulness.csv` |
| Noise feature rank on crop data (all 3 methods) | 8 of 8 (last) | K `tab10_noise_feature_sanity.csv` |
| Counterfactual: reachable by soil change / inside range | 0.3762 / 0.3970 (110 instances) | K `tab10_counterfactual_summary.csv` |
| ECE raw → isotonic: SVM, RF | 0.1101 → 0.0070, 0.0398 → 0.0038 | K `tab06_calibration_ece.csv` |
| LightGBM LAC α=0.10: coverage, worst crop | 0.8967, 0.6050 | K `tab06_conformal_summary.csv` |
| LightGBM APS α=0.10: coverage, set size | 0.9940, 1.1408 | K `tab06_conformal_summary.csv` |

## Fertilizer dataset (99 rows)

| Claim in paper | Value | Source |
|---|---|---|
| Class counts | Urea 22, DAP 18, 28-28 17, 14-35-14 14, 20-20 14, 17-17-17 7, 10-26-26 7 | `data/Fertilizer_Prediction_99rows.csv` (counted) |
| Every cell of Table III (LOO accuracy) | as printed | F `tab11_fertilizer_loo_by_feature_set.csv` |
| Majority class | 0.2222 | F `tab11_fertilizer_loo_by_feature_set.csv` |
| Every cell of Table IV (cosine) | as printed | F `tab11_npk_vs_grade_cosine.csv` |
| Classes matching own grade | 100 % (20-20 / 28-28 tie) | F `tab11_grade_match_summary.csv` |
| Tree rules (P ≤ 4 → Urea, P > 34 → DAP) | as printed | F `tab11_npk_rules.txt` |
| Uninformative-feature share: SHAP, LIME, permutation | 0.2349, 0.3588, 0.0232 | K `tab10_fertilizer_attribution_shares.csv` |
| STCR rice dose → DAP, urea, MOP | 113.0, 173.2, 41.7 kg/ha | F `tab11_product_conversion_example.csv` |
| ICAR-IISS and TNAU thresholds | as printed | `11_fertilizer_recommendation.py` docstring (from the cited standards) |

## Real-data experiments

| Claim | Value |
|---|---|
| District validation (07), ranking (08), SHC rating comparison (11 Part B), label-threshold sensitivity | NOT AVAILABLE: soil and climate files not yet obtained (network blocked here; see chat). |
| District table size used in Future Work | 34,940 rows, 48 labels: `data/DATA_CARD.md` §8 and `data/build_datagov/` |
