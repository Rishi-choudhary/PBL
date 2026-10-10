# Paper (IEEE conference format)

- `main.tex`: the paper, in the same IEEEtran two-column format as the sample you sent.
- `main.pdf`: compiled output, 8 pages.
- `figures/`: copies of the figures from `crop_experiments/results/`.

To edit, upload this folder to Overleaf, or run `pdflatex main.tex` twice.

## Where each figure and table goes

| In paper | Section | Source file |
|---|---|---|
| Table I: models and settings | II-A Background | `common.py` (`get_models`) |
| Fig. 1: pipeline | III-C | drawn in LaTeX (TikZ) |
| Table II: 16-model comparison | IV-B Model comparison | `results/kaggle/tables/tab02_benchmark_paper.csv` |
| Fig. 2: model comparison | IV-B | `results/kaggle/figures/fig04_model_comparison.pdf` |
| Fig. 3: AdaBoost diagnosis | IV-B | `fig08_adaboost_diagnosis.pdf` |
| Fig. 4: overfitting vs leakage | IV-C | `fig14_overfitting_vs_leakage.pdf` |
| Fig. 5: learning curve and noise | IV-D Robustness | `fig09_learning_curve_and_noise.pdf` |
| Fig. 6: SHAP vs LIME, deletion, noise feature | IV-E Explanations | `fig15_xai_disagreement_faithfulness.pdf` |
| Fig. 7: counterfactuals | IV-E | `fig16_counterfactual_feasibility.pdf` |
| Fig. 8: calibration and conformal | IV-F Uncertainty | `fig13_calibration_conformal.pdf` |
| Table III: fertilizer LOO by feature group × model | IV-G Fertilizer | `results/fertilizer/tables/tab11_fertilizer_loo_by_feature_set.csv` |
| Table IV: fertilizer N:P:K vs grade (cosine) | IV-G | `tab11_npk_vs_grade_cosine.csv` |
| Fig. 9: fertilizer benchmark | IV-G | `results/fertilizer/figures/fig17_fertilizer_benchmark.pdf` |
| Table V: dose → DAP/urea/MOP and rating standards | IV-G | `tab11_product_conversion_example.csv`, `11_fertilizer_recommendation.py` |

Not used: fig01–03, fig05–07, fig10–12 (in the supplement or the viva if asked). We also left out the `research/` folder plots, because the newer `crop_experiments/` results replace them.

## Still to do before submission

1. Replace the placeholder authors in `main.tex`.
2. References: none could be opened from this session (DOI and publisher sites are blocked), so open each one and check it. Fill the two `% TODO` entries: ICAR-IISS Harit Dhara (title, authors, pages) and the TNAU page URL.
3. Real district results (07/08) are not in the paper. They go in once the soil and climate data are attached.
