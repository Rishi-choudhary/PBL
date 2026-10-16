# Executable files behind the PBL-3 deck

Deck: `ppt/Crop_Recommendation_System_PBL3.pptx` (30 slides). Run the steps in this order. Paths are relative to the repository root.

| Step | File | Run from | Output the deck uses | Slides |
|---|---|---|---|---|
| 1 | `crop_experiments/12_crop_system.py` | `crop_experiments/` | `results/system/figures/figS1`–`figS6`, `results/system/tables/tabS1`–`tabS9`, `results/system/summary.txt` | 3, 10–18, 20–25, 27 |
| 2 | `crop_experiments/13_overfitting_check.py` | `crop_experiments/` | `tabS10_train_vs_test`, `tabS10_tree_depth`, `tabS10_permutation_test` | 19, 27 |
| 3 | `paper/main.tex` (`pdflatex main.tex`, twice) | `paper/` | `paper/main.pdf` | 26 |
| 4 | `report/make_report_figures.py` | `report/` | `report/figures/fig1_1_architecture.png`, `fig2_5_overfitting.png` (drawn from the `tabS10` tables), `paper_page-1.png` | 4, 19, 26 |
| 5 | `ppt/build_ppt.py` | `ppt/` | the deck itself, built from `DSE3170_PPT_Templates_for_PBL-3.pptx` | all |
| – | `crop_experiments/common.py` | not run directly | shared helpers: data loading, XGBoost and LightGBM settings, figure style, saving tables (imported by steps 1, 2 and 4) | – |

Inputs read:

- `crop_experiments/data/Crop_recommendation.csv`
- `crop_experiments/data/Fertilizer_Prediction_99rows.csv`
- `crop_experiments/data/provenance/harvestify_aac427c/FertilizerData.csv`
- `ppt/DSE3170_PPT_Templates_for_PBL-3.pptx`

The numbers on the slides are typed into `build_ppt.py`. `crop_experiments/results/CLAIMS.md` records which file each one comes from.

Not used by the deck: `crop_experiments/01_*.py` to `11_*.py`, `run_all.sh`, the `crop_experiments 2/` folder, the root zip files, and `report/build_report.py` (it builds the report docx).

Re-run check: all five steps were re-run from a clean copy. Step 1 took about 124 s and step 2 about 160 s. Every results table matched the committed files, except training times, which depend on the machine. The rebuilt deck's text was identical to the committed deck.
