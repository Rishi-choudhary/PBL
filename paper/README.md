# Paper: Crop Recommendation System (IEEE conference format)

- `main.tex`: LaTeX source, same IEEEtran two-column format as the sample paper. You can upload this folder to Overleaf, or build it with `pdflatex main.tex` (run it twice).
- `main.pdf`: compiled paper, 8 pages.
- `figures/`: graphs made by `crop_experiments/12_crop_system.py`.
- `DATASETS.md`: dataset links for your teacher.

## Where each figure and table goes

| In paper | Section | Source file (in `crop_experiments/results/system/`) |
|---|---|---|
| Fig. 1: system architecture | III-A | drawn in LaTeX |
| Table I: datasets | III-B | `DATASETS.md` |
| Table II: feature summary | III-C | `tables/tabS1_feature_summary.csv` |
| Fig. 2: correlation + mean N, P, K per crop | III-C | `figures/figS1_data_analysis.pdf` |
| Table III: algorithms and settings | III-D | `12_crop_system.py` (`crop_models`) |
| Table IV: algorithm comparison (test + CV) | IV-B | `tables/tabS2_crop_model_comparison.csv` |
| Fig. 3: algorithm comparison graph | IV-B | `figures/figS2_model_comparison.pdf` |
| Fig. 4: confusion matrix (Random Forest) | IV-C | `figures/figS3_confusion_best.pdf`, `tables/tabS3_per_crop_best.csv` |
| Table V: top-3 crops for six farmers | IV-C | `tables/tabS4_farmer_examples.csv` |
| Fig. 5: SHAP global + per crop | IV-D | `figures/figS4_shap_global_per_crop.pdf`, `tables/tabS5_*` |
| Fig. 6: SHAP vs LIME for one farmer | IV-D | `figures/figS5_farmer_explanation.pdf`, `tables/tabS6_*` |
| Table VI: fertilizer algorithm comparison | IV-E | `tables/tabS7_fertilizer_model_comparison.csv` |
| Fig. 7: fertilizer accuracy, importance, confusion | IV-E | `figures/figS6_fertilizer.pdf`, `tables/tabS8_fertilizer_importance.csv` |
| Table VII: nutrient-gap fertilizer advice | IV-E | `tables/tabS9_nutrient_gap_examples.csv` |

All numbers: `results/system/summary.txt` and `results/CLAIMS.md`.

## Before submission

1. Replace the placeholder authors (Author One, Author Two, Guide Name).
2. Open and check every reference. None could be opened from this session because journal and DOI sites are blocked. The method papers are standard. Check these in particular: Kumar et al. 2015, Pudumalar et al. 2017, Doshi et al. 2018 and Ryo 2022 (titles, pages, and that they say what Section II-B says), and the Kaggle fertilizer link.
3. Rewrite the text in your own words, section by section, so you can explain every part in the viva.
