# Paper: Crop Recommendation System (IEEE conference format)

- `main.tex`: LaTeX source, built on the IEEE conference template (version 6/27/2024, `IEEEtran` class, letter size). Upload this folder to Overleaf, or build it with `pdflatex main.tex` (run it twice).
- `main.pdf`: compiled paper, 9 pages.
- `figures/`: the 8 graphs, copied from `crop_experiments/results/system/figures/` (PDF versions).
- `DATASETS.md`: dataset links for your teacher.

Authors: Rishi Choudhary and Anant Srivastav (Dept. of Data Science and Engineering), Jaswant Kumar (Dept. of Computer Science and Engineering), Manipal University Jaipur.

## Where each figure and table comes from

Paths are in `crop_experiments/results/system/`.

| In paper | Section | Source file | Made by |
|---|---|---|---|
| Fig. 1: system architecture | III-A | drawn in LaTeX | – |
| Table I: datasets | III-B | `DATASETS.md`, `summary.txt` | `12_crop_system.py` |
| Table II: feature summary | III-C | `tables/tabS1_feature_summary.csv` | `12_crop_system.py` |
| Fig. 2: correlation + mean N, P, K per crop | III-C | `figures/figS1_data_analysis.pdf` | `12_crop_system.py` |
| Table III: algorithms and settings | III-D | `crop_models()` in `12_crop_system.py`, `common.py` | – |
| Table IV: 5-fold CV + held-out test | IV-B | `tables/tabS11_crop_5fold_summary.csv`, `tables/tabS2_crop_model_comparison.csv` | `14_five_fold_cv.py`, `12_crop_system.py` |
| Fig. 3: 5-fold CV, crop and fertilizer | IV-B, IV-E | `figures/figS7_five_fold_cv.pdf` | `14_five_fold_cv.py` |
| Fig. 4: confusion matrix (Random Forest) | IV-C | `figures/figS3_confusion_best.pdf`, `tables/tabS3_per_crop_best.csv` | `12_crop_system.py` |
| Table V: top-3 crops for six farmers | IV-C | `tables/tabS4_farmer_examples.csv` | `12_crop_system.py` |
| Fig. 5: SHAP global + per crop | IV-D | `figures/figS4_shap_global_per_crop.pdf`, `tables/tabS5_*` | `12_crop_system.py` |
| Fig. 6: SHAP vs LIME for one farmer | IV-D | `figures/figS5_farmer_explanation.pdf`, `tables/tabS6_*` | `12_crop_system.py` |
| Table VI: fertilizer algorithms (5-fold and 5×10) | IV-E | `tables/tabS12_fert_5fold_summary.csv`, `tables/tabS7_fertilizer_model_comparison.csv` | `14_five_fold_cv.py`, `12_crop_system.py` |
| Fig. 7: fertilizer accuracy, importance, confusion | IV-E | `figures/figS6_fertilizer.pdf`, `tables/tabS8_fertilizer_importance.csv` | `12_crop_system.py` |
| Table VII: nutrient-gap advice | IV-E | `tables/tabS9_nutrient_gap_examples.csv` | `12_crop_system.py` |
| Fig. 8: dataset overlap and coverage | IV-F | `figures/figS8_dataset_overlap.pdf`, `tables/tabS13_*`, `tables/tabS14_ml_fertilizer_coverage.csv` | `15_multimodel_experiment.py` |
| Multi-model check (6 of 6, 12 of 12) | IV-F | `summary_multimodel.txt`, `tables/tabS15_multimodel_six_farmers.csv` | `15_multimodel_experiment.py`, `multimodel.py` |

All numbers: `results/system/summary.txt`, `summary_cv5.txt`, `summary_multimodel.txt` and `results/CLAIMS.md`.

## Before submission

1. Check the conference's page limit. The paper is 9 pages; many IEEE conferences allow 6, or 8 with an extra-page fee.
2. Add Jaswant Kumar's e-mail address to the third author block if the conference needs one for every author (it was not provided).
3. Open and check every reference. None could be opened from this session because journal and DOI sites are blocked. Check in particular Kumar et al. 2015, Pudumalar et al. 2017, Doshi et al. 2018 and Ryo 2022 (titles, pages, and that they say what Section II-B says), and the Kaggle fertilizer link.
4. Rewrite the text in your own words, section by section, so you can explain every part in the viva.
