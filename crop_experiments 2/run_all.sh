#!/usr/bin/env bash
# Runs every experiment in order. Kaggle + fertilizer part ~1.5 h on 2 CPU cores.
set -e
cd "$(dirname "$0")"
python3 01_eda_kaggle.py
python3 02_benchmark_kaggle.py --folds 5 --repeats 5
python3 03_adaboost_diagnosis.py
python3 04_robustness_kaggle.py
python3 05_explainability_kaggle.py
python3 06_uncertainty_kaggle.py
python3 09_overfitting_diagnosis.py
python3 10_xai_extended.py
python3 11_fertilizer_recommendation.py
# Real district data: needs data/crop_wide_district_year.csv with soil_* / clim_* columns
if python3 -c "import pandas as pd,sys; c=pd.read_csv('data/crop_wide_district_year.csv',nrows=1).columns; sys.exit(0 if any(x.startswith(('soil_','clim_')) for x in c) else 1)" 2>/dev/null; then
  python3 07_district_experiments.py
  python3 08_district_ranking.py
else
  echo "District features not attached yet - skipping 07/08 (run with --smoke-test to check the pipeline)."
fi
