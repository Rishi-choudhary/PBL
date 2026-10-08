# Provenance of the Kaggle "Crop Recommendation" dataset

Source repository: https://github.com/Gladiator07/Harvestify, commit aac427c1dc6adbadd4d8f16e77e74d6e323d39a1 (16 May 2023), GPL-3.0 (LICENSE copied here).
The repository README links https://www.kaggle.com/atharvaingle/crop-recommendation-dataset and calls it a "custom built dataset", with the disclaimer:
"The data used here comes up with no guarantee from the creator. So, don't use it for making farming decisions."

Files copied verbatim:
- Final_recommendationdata_creation.ipynb  - builds crop_recommendation.csv. N, P, K for each row are
  base[crop] + random.randint(-20,20) / (-5,20) / (-5,5) respectively.
- Crop_data_prep.ipynb                       - selects the 22 crops from cpdata.csv and FertilizerData.csv.
- FertilizerData.csv                         - the 22-row base table (crop -> N, P, K, pH).
- cpdata.csv                                 - source of temperature, humidity, ph, rainfall (origin undocumented).

Verified in this project (09_overfitting_diagnosis.py): 2200/2200 rows of the Kaggle CSV satisfy the rule;
the observed offsets take exactly 41, 26 and 11 distinct values, the number the randint ranges allow.

The district production table (../datagov_district_season_crop_production_1997_2015.csv) was taken from the same
repository's Data-raw/raw_districtwise_yield_data.csv; it is the data.gov.in "District-wise, season-wise crop
production statistics" table (246,091 rows). Re-download it from data.gov.in before publication.
