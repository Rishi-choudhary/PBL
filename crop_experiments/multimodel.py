"""Multi-model crop + fertilizer recommender (mentor request).

One call, `CropFertilizerSystem.recommend(farmer)`, returns both answers:
  1. Crop model (Random Forest, the system model of experiment 12): top-k crops with probabilities.
  2. Fertilizer, two routes:
     a. nutrient-gap rule (all 22 crops): gap = nutrient the crop needs - nutrient in the soil; the
        largest gap decides the advice. Same rule as experiment 12 (tabS9).
     b. ML product model (XGBoost, trained on the 99-row fertilizer dataset): predicts a fertilizer
        product. It is used ONLY when its inputs are inside what it was trained on (see `_product`);
        otherwise the answer says why it was skipped.

The two models are trained on two different datasets that share no records, so they are chained, not
trained jointly: the crop model's top-1 crop becomes the "Crop Type" input of the product model.

Command line (trains on all rows, about 3 s):
  python3 multimodel.py --N 90 --P 42 --K 43 --temperature 20.9 --humidity 82 --ph 6.5 --rainfall 203
  python3 multimodel.py ... --moisture 40 --soil-type Loamy      # adds the ML product route
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from common import load_kaggle, KAGGLE_FEATURES, SEED

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("crop_system", HERE / "12_crop_system.py")
crop_system = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crop_system)

# Crop names of the crop dataset -> "Crop Type" of the fertilizer dataset. Only these 10 of the 22 crops
# have a counterpart (Paddy = rice; Pulses = the seven pulse crops). The other 12 have none.
CROP_TO_FERT_TYPE = {"maize": "Maize", "cotton": "Cotton", "rice": "Paddy",
                     **{c: "Pulses" for c in ["chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
                                              "mungbean", "blackgram", "lentil"]}}
ADVICE = {("N", "low"): "add nitrogen (e.g. urea)", ("N", "high"): "skip nitrogen this season",
          ("P", "low"): "add phosphate (e.g. DAP or SSP)", ("P", "high"): "skip phosphate this season",
          ("K", "low"): "add potash (e.g. MOP)", ("K", "high"): "skip potash this season"}
FERT_NUM = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Phosphorous", "Potassium"]   # spelling as in the data
FERT_CAT = ["Soil Type", "Crop Type"]
FARMER_TO_FERT = {"temperature": "Temparature", "humidity": "Humidity", "moisture": "Moisture",
                  "N": "Nitrogen", "P": "Phosphorous", "K": "Potassium"}


def _fert_pre(scale):
    return ColumnTransformer([("num", StandardScaler() if scale else "passthrough", FERT_NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FERT_CAT)])


class CropFertilizerSystem:
    def __init__(self, crop_csv=HERE / "data/Crop_recommendation.csv",
                 fert_csv=HERE / "data/Fertilizer_Prediction_99rows.csv",
                 req_csv=HERE / "data/provenance/harvestify_aac427c/FertilizerData.csv",
                 crop_model="Random Forest", fert_model="XGBoost"):
        self.crop_csv, self.fert_csv, self.req_csv = crop_csv, fert_csv, req_csv
        self.crop_model_name, self.fert_model_name = crop_model, fert_model

    def fit(self, train_rows=None):
        """Train both models. `train_rows` restricts the crop model to some rows of the crop CSV (experiments
        pass the 1,760 training rows of the 80/20 split); the default is all 2,200 rows."""
        _, X, y, le = load_kaggle(self.crop_csv)
        rows = np.arange(len(y)) if train_rows is None else np.asarray(train_rows)
        self.crop_names = list(le.classes_)
        self.crop_model = crop_system.crop_models()[self.crop_model_name].fit(X[rows], y[rows])

        fd = pd.read_csv(self.fert_csv); fd.columns = [c.strip() for c in fd.columns]
        enc = LabelEncoder()
        yf = enc.fit_transform(fd["Fertilizer Name"])
        self.fert_classes = list(enc.classes_)
        self.fert_model = crop_system.fert_models(_fert_pre)[self.fert_model_name].fit(fd[FERT_NUM + FERT_CAT], yf)
        self.fert_range = {c: (float(fd[c].min()), float(fd[c].max())) for c in FERT_NUM}   # training range
        self.soil_types = sorted(fd["Soil Type"].unique())
        self.requirement = pd.read_csv(self.req_csv, index_col=0).set_index("Crop")
        return self

    # ------------------------------------------------------------------ stage 1
    def crop_probabilities(self, X):
        return self.crop_model.predict_proba(np.atleast_2d(X))

    # ------------------------------------------------------------------ stage 2a
    def nutrient_gap(self, crop, soil_npk):
        need = self.requirement.loc[crop, ["N", "P", "K"]].astype(float).values
        gap = need - np.asarray(soil_npk, float)
        j = int(np.argmax(np.abs(gap)))
        nut, status = "NPK"[j], "low" if gap[j] > 0 else "high"
        return dict(required_NPK=tuple(int(v) for v in need), gap_NPK=tuple(float(v) for v in gap),
                    largest_gap=f"{nut} {status} by {abs(gap[j]):.0f}", advice=ADVICE[(nut, status)])

    # ------------------------------------------------------------------ stage 2b
    def product_checks(self, farmer, top_crop):
        """Reasons why the ML product model cannot be trusted for this input (empty list = it can)."""
        why = []
        if farmer.get("moisture") is None:
            why.append("soil moisture not given")
        if farmer.get("soil_type") is None:
            why.append("soil type not given")
        elif farmer["soil_type"] not in self.soil_types:
            why.append(f"soil type '{farmer['soil_type']}' is not one of {', '.join(self.soil_types)}")
        if top_crop not in CROP_TO_FERT_TYPE:
            why.append(f"{top_crop} is not a crop type in the fertilizer data")
        for key, col in FARMER_TO_FERT.items():
            v = farmer.get(key)
            lo, hi = self.fert_range[col]
            if v is not None and not lo <= v <= hi:
                why.append(f"{key} = {v:.4g} is outside {lo:g}-{hi:g}, the range of the fertilizer training data")
        return why

    def _product(self, farmer, top_crop):
        why = self.product_checks(farmer, top_crop)
        if why:
            return dict(applicable=False, product=None, probability=None, reasons=why)
        row = {col: farmer[key] for key, col in FARMER_TO_FERT.items()}
        row.update({"Soil Type": farmer["soil_type"], "Crop Type": CROP_TO_FERT_TYPE[top_crop]})
        p = self.fert_model.predict_proba(pd.DataFrame([row]))[0]
        i = int(np.argmax(p))
        return dict(applicable=True, product=self.fert_classes[i], probability=float(p[i]), reasons=[])

    # ------------------------------------------------------------------ both stages
    def recommend(self, farmer, top_k=3):
        x = np.array([farmer[f] for f in KAGGLE_FEATURES], float)
        p = self.crop_probabilities(x)[0]
        crops = [dict(crop=self.crop_names[i], probability=float(p[i]),
                      nutrient_gap=self.nutrient_gap(self.crop_names[i], x[:3])) for i in np.argsort(-p)[:top_k]]
        return dict(crops=crops, fertilizer_product=self._product(farmer, crops[0]["crop"]))


def format_result(r):
    lines = ["Recommended crops:"]
    for k, c in enumerate(r["crops"], 1):
        g = c["nutrient_gap"]
        lines.append(f"  {k}. {c['crop']:12s} probability {c['probability']:.2f}   needs N-P-K "
                     f"{'-'.join(map(str, g['required_NPK']))}; largest gap: {g['largest_gap']} -> {g['advice']}")
    f = r["fertilizer_product"]
    if f["applicable"]:
        lines.append(f"Fertilizer product (ML model, crop = {r['crops'][0]['crop']}): {f['product']} "
                     f"(probability {f['probability']:.2f})")
    else:
        lines.append("Fertilizer product (ML model): not used because " + "; ".join(f["reasons"]))
    lines.append("Nutrient-gap advice above covers every crop; the ML product model only covers inputs inside "
                 "its 99-row training range.")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in KAGGLE_FEATURES:
        ap.add_argument(f"--{f}", type=float, required=True)
    ap.add_argument("--moisture", type=float, default=None, help="soil moisture (needed for the ML product route)")
    ap.add_argument("--soil-type", default=None, help="Sandy, Loamy, Black, Red or Clayey (needed for the ML product route)")
    ap.add_argument("--top-k", type=int, default=3)
    a = ap.parse_args()
    farmer = {f: getattr(a, f) for f in KAGGLE_FEATURES}
    farmer.update(moisture=a.moisture, soil_type=a.soil_type)
    print(format_result(CropFertilizerSystem().fit().recommend(farmer, a.top_k)))
