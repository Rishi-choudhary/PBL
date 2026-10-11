# Datasets used in the paper

| # | Dataset | What it is | Link | Copy in this repo |
|---|---|---|---|---|
| 1 | Crop Recommendation Dataset (Atharva Ingle, Kaggle) | 2,200 rows, 22 crops (100 each). Inputs: N, P, K, temperature, humidity, pH, rainfall | https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset | `crop_experiments/data/Crop_recommendation.csv` |
| 2 | Fertilizer Prediction (Kaggle) | 99 rows, 7 fertilizers. Inputs: temperature, humidity, moisture, soil type, crop type, N, P, K | https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction | `crop_experiments/data/Fertilizer_Prediction_99rows.csv` |
| 3 | Crop nutrient requirement table (Harvestify) | 22 crops: required N, P, K, pH | https://github.com/Gladiator07/Harvestify/blob/master/Data-processed/fertilizer.csv | `crop_experiments/data/provenance/harvestify_aac427c/FertilizerData.csv` |

Checked from this session: link 1 is the dataset link given in the creator's own Harvestify README, and the data in link 3 is the same as our copy (N, P, K, pH values match).
Not checked: link 2. Kaggle is blocked from this session. Before you send it, open it and make sure it is the 99-row "Fertilizer Prediction.csv" file with a column named `Temparature` (the misspelling is in the original).

Code that produces every number and graph in the paper: `crop_experiments/12_crop_system.py`. Run it from `crop_experiments/` with `python3 12_crop_system.py`. It takes about 2 minutes on a CPU.
