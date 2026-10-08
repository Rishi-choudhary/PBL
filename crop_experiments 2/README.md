# Crop and fertilizer recommendation experiments (PBL)

Code, data and figures for the paper. Everything runs from this folder.

```
data/
  Crop_recommendation.csv              Kaggle benchmark (2,200 rows)
  Fertilizer_Prediction_99rows.csv     common fertilizer benchmark (99 rows)
  ICRISAT_district_level_data.csv      ICRISAT-TCI DLD crop area/yield, 311 districts, 1978-2017
  datagov_district_season_crop_production_1997_2015.csv   data.gov.in, 646 districts, seasons
  build_dataset.py                     ICRISAT -> district-year table (+ soil/climate joins)
  build_dataset_datagov.py             data.gov.in -> district-season-year table (+ joins)
  build_datagov/                       built season-wise table (34,940 rows, 48 crop labels)
  provenance/harvestify_aac427c/       creator's notebooks + base table for the Kaggle dataset
  DATA_CARD.md
common.py                    model zoo, metrics, statistics, conformal, figure style
01_eda_kaggle.py             why the Kaggle benchmark is saturated
02_benchmark_kaggle.py       16 models x 25 CV splits + significance tests
03_adaboost_diagnosis.py     why AdaBoost collapses on 22 classes
04_robustness_kaggle.py      learning curve, measurement noise, feature ablation
05_explainability_kaggle.py  SHAP / permutation / gain + stability (Kendall's W)
06_uncertainty_kaggle.py     calibration (ECE) + conformal prediction sets
07_district_experiments.py   real data: random vs district vs state vs temporal validation
08_district_ranking.py       real data: crops ranked per district (LambdaRank, NDCG@3)
09_overfitting_diagnosis.py  train/test gap, complexity curves, permutation test, generator check
10_xai_extended.py           SHAP vs LIME disagreement, deletion test, noise feature, counterfactuals
11_fertilizer_recommendation.py  fertilizer benchmark diagnosis, SHC rating standards, dose -> products
results/kaggle/  results/fertilizer/   figures (PNG 300 dpi + PDF) and tables (CSV + LaTeX)
(--smoke-test on 07/08 checks the pipeline with FAKE features; those figures are watermarked)
```

## Running

```bash
pip install -r requirements.txt
./run_all.sh                 # ~1.5 h on 2 CPU cores
```
If `pip install lime` fails with `AttributeError: install_layout`, download the source (`pip download --no-deps --no-binary :all: lime==0.2.0.1`), extract it and copy the `lime/` folder into site-packages; it is pure Python.

TabPFN (needs internet for weights, so Colab): `pip install tabpfn "tabpfn-extensions[many_class]"`, then `python 02_benchmark_kaggle.py --tabpfn --out results/kaggle_tabpfn` and `python 07_district_experiments.py --tabpfn`. TabPFN-2.5/2.6 support at most 10 classes natively; the code wraps it in `ManyClassClassifier` for more.

Real district experiments once soil/climate files exist (season-wise table recommended):
```bash
cd data
python build_dataset_datagov.py --soil soil_district.csv --climate climate_district_year.csv --out build_datagov
cd ..
D=data/build_datagov/crop_wide_district_season_year.csv
python 07_district_experiments.py --data $D --feature-prefixes soil_,clim_,ctx_ --year-min 2010 --year-max 2014 --cutoff 2012 --out results/district_datagov
python 08_district_ranking.py     --data $D --feature-prefixes soil_,clim_,ctx_ --year-min 2010 --year-max 2014 --cutoff 2012 --out results/district_datagov
# ablations: --feature-prefixes soil_,ctx_  and  clim_,ctx_
# robustness: the ICRISAT table (data/build_dataset.py, default 07/08 settings)
python 11_fertilizer_recommendation.py --shc shc_samples.csv --col-n <N col> --col-p <P col> --col-k <K col>
```
The 2015 rows of the data.gov.in table cover only 2 states; the commands above exclude them.

## Which figure goes where

| Paper section | Figure / table |
|---|---|
| Dataset provenance (Kaggle) | data/provenance, tab09_generator_check, fig14(d) |
| Dataset: limits of the benchmark | fig01 feature ranges, fig02 separability, fig03 correlation, tab01_distribution_summary |
| Is it overfitting? | fig14, tab09_* |
| Results: model comparison | fig04, tab02_benchmark_paper (.tex), fig05 cost vs F1 |
| Error analysis | fig06 confusion matrix, fig07 per-crop F1 |
| AdaBoost anomaly | fig08, tab03 |
| Robustness | fig09 learning curve + noise, fig10 ablation |
| Explainability | fig11 importance + stability, fig12 per-crop SHAP, fig15 disagreement/deletion/noise, tab10_* |
| Actionable XAI / link to fertilizer | fig16 counterfactual feasibility, tab10_counterfactual_* |
| Uncertainty | fig13 calibration + conformal, tab06 |
| Fertilizer recommendation | fig17, tab11_* (results/fertilizer) |
| Main contribution (real data) | figD1–figD7, tabD1–tabD6 from 07/08 (pending soil data) |

## Results so far (all numbers produced by these scripts)

Kaggle crop dataset
- Provenance: the creator's repository (Harvestify, commit aac427c) calls it a "custom built dataset" and warns not to use it for farming decisions. Its notebook sets N = base[crop] + randint(-20, 20), P = base[crop] + randint(-5, 20), K = base[crop] + randint(-5, 5). All 2,200 rows satisfy this; the offsets take exactly 41, 26 and 11 distinct values. Temperature, humidity, pH and rainfall are copied from cpdata.csv, whose origin is undocumented.
- Our earlier statistical test detected this independently: 0/154 crop-feature cells normal, 82–100% consistent with uniform.
- Not overfitting in the usual sense: train-test gaps are 0.5 pp (Random Forest) to 2.5 pp (KNN); the tree-depth curve shows no divergence; with shuffled labels Random Forest scores 4.6% (chance 4.5%; p = 0.0099, 100 permutations). The high score comes from features generated from the label: a lookup of the creator's 22-row table identifies the crop from N, P, K with 65.8% expected accuracy and no learning; a Random Forest trained on N, P, K reaches 65.0%.
- Benchmark (5x5 repeated stratified CV): best is Stacking at 99.57% ± 0.26. 11 of 16 models are not significantly worse (corrected resampled t-test, Holm), including Gaussian NB (99.47%, ~0 s fit) and a single decision tree (98.80%).
- Learning curve: RF 0.974 macro-F1 with 8 samples per crop. Noise 0.2 x std: LightGBM 0.773, RF 0.872, SVM 0.947. Ablation: climate-only 0.918 vs soil-only 0.788.
- AdaBoost with default stumps: 20–25%; with depth-3 trees 99.1%; boosting SVM (88.5%) or NB (97.8%) is worse than the base learner. The report's Fig 7.2 values (0.886, 0.975) are close to these; Table 8.3's 21.36% falls in the stump range (suggestive, not proof).
- Explanations on random folds are stable (SHAP Kendall's W = 0.97). SHAP vs LIME (200 instances): 76% top-3 overlap, same top feature 80%, Spearman 0.65, sign agreement 0.98. Deletion test: removing the top SHAP feature drops P(predicted crop) from 0.99 to 0.08 (LIME 0.20, random 0.78). A pure-noise feature is ranked last by all three methods.
- Counterfactuals with climate fixed and only N, P, K, pH changeable: soil change alone reaches the target crop in 38% of crop pairs; only 40% of those counterfactuals fall inside the target crop's range in the data, the rest rely on extrapolation.
- Calibration: isotonic lowers ECE for SVM 0.110 → 0.007 and RF 0.040 → 0.004. Conformal (LightGBM, alpha 0.10): coverage 0.897, worst crop 0.605.

Fertilizer dataset (99 rows)
- N, P, K alone: Random Forest and SVM 100% leave-one-out accuracy. Temperature/humidity/moisture: 16–25%; soil + crop type: 9–14%; majority class: 22%. Adding the uninformative features drops SVM to 67%.
- Every class's median N:P:K is most similar (cosine) to the grade of the fertilizer in its label (20-20 and 28-28 tie, identical ratio). The inputs mirror the label's composition; the construction is undocumented, so treat this as strong evidence, not proof.
- SHAP gives 23% and LIME 36% of attribution to the uninformative features; held-out permutation importance 2%. Attribution methods describe what the model uses, not what is predictive.

## Research gaps this project addresses

| Gap in the literature | Evidence here | What we do |
|---|---|---|
| Benchmark provenance unchecked; the Kaggle dataset's N/P/K are generated from the label | creator's notebook + 2200/2200 rows verified | report it; move to government data |
| Accuracy reported without significance tests or train/test gap | 11/16 models tied; gaps < 2.5 pp | corrected t-test, permutation test, complexity curves |
| Random-split validation only; no spatial/temporal testing | review of 12 studies (Raut & Chimate 2026) | random vs leave-district/state-out vs temporal (07, 08) |
| XAI applied but never evaluated | SHAP/LIME disagree; 23–36% attribution to uninformative features | disagreement, deletion, noise-feature and stability tests (05, 10) |
| Explanations not actionable | 38% of switches reachable by soil change; 60% of counterfactuals extrapolate | constrained counterfactuals over mutable nutrients (10) |
| Fertilizer ML predicts a product name from inputs that mirror it, without doses | fertilizer benchmark results | soil-test rating (ICAR-IISS vs TNAU) + dose-to-product conversion (11) |
| No uncertainty | — | calibration + conformal sets, also under spatial shift (06, 07) |
| Recommendation framed as one label | — | multi-label and learning-to-rank (07, 08) |

## Datasets

| Role | Dataset | Status |
|---|---|---|
| Crop outcomes (primary) | data.gov.in district-season crop production, 1997–2015, 646 districts, 48 labels kept | built (data/build_datagov) |
| Crop outcomes (robustness) | ICRISAT-TCI DLD, 311 districts, 1978–2017 | built |
| Soil | Soil Health Card: district nutrient dashboard, or dataful.in point data with lat/lon (N, P, K, S, OC, pH, EC, micronutrients) | needs your download |
| Climate | DLD monthly rainfall; IMD gridded rainfall; NASA POWER temperature/humidity | needs your download |
| Fertilizer (real) | Mendeley "Soil Macro Nutrient Dataset for Fertilizer Combinations Advice, Pune" (DOI 10.17632/tfvgnm8rxt.1, SHC 2016–18); SHC cards include issued dose recommendations | needs your download |
| Benchmarks (diagnosis only) | Kaggle crop (2,200), Fertilizer Prediction (99) | done |

## Algorithms

Included: Logistic Regression, Gaussian NB, KNN, RBF-SVM, Decision Tree, Random Forest, Extra Trees, Hist. Gradient Boosting, XGBoost, LightGBM, CatBoost, MLP, AdaBoost, Bagging, Soft Voting, Stacking; class-balanced RF/LightGBM and a prior baseline on district data; LambdaRank; TabPFN v2 via flag.
Worth adding: classifier chains (crop co-occurrence), FT-Transformer as deep-tabular baseline (expect GBDTs to stay competitive, Grinsztajn et al. NeurIPS 2022).
Not worth it: more ensembles on Kaggle, CTGAN/SMOTE augmentation of Kaggle rows, CNN/LSTM on static tabular features.

## Still to run

1. Soil Health Card + rainfall downloads → rebuild → 07/08 on the season-wise table (headline result), ICRISAT as robustness.
2. 11 Part B on SHC sample data (rating-standard sensitivity).
3. Label-threshold sensitivity (`--share-threshold 0.03`, `0.10`).
4. TabPFN on Colab.
5. Hyperparameters are defaults: say so, or add nested tuning for the top 3 models.
6. Re-download ICRISAT and data.gov.in tables from the official portals and diff against the mirror copies.

## Method references

Nadeau & Bengio 2003; Bouckaert & Frank 2004 (corrected resampled t-test). Ojala & Garriga 2010, JMLR 11 (permutation test for classifiers). Sadinle, Lei & Wasserman 2019 (LAC); Romano, Sesia & Candès 2020 (APS); Angelopoulos & Bates 2021. Guo et al. 2017 (calibration). Roberts et al. 2017; Ploton et al. 2020 (spatial CV). Lundberg & Lee 2017 (SHAP); Ribeiro, Singh & Guestrin 2016 (LIME); Krishna et al. 2024, TMLR (disagreement problem); Wachter, Mittelstadt & Russell 2018 (counterfactual explanations); Ustun, Spangher & Liu 2019 (actionable recourse). Hollmann et al. 2025 (TabPFN). ICAR-IISS, Harit Dhara 4(1) 2021 (SHC ratings; STCR targeted yield). TNAU Agritech soil rating chart.
