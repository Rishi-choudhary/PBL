"""Experiment 11 - Fertilizer recommendation: what the public benchmark measures,
and a soil-test-based alternative.

Part A (runs now): the 99-row "Fertilizer Prediction" dataset used by many ML
fertilizer papers. Leave-one-out accuracy by feature group, majority baseline,
and a test of whether each class's N:P:K profile mirrors the nutrient grade of
the fertilizer named in its label (cosine similarity of class-median N, P, K
with each product's N-P2O5-K2O grade).

Part B (runs when real Soil Health Card data are supplied with --shc): soil
nutrient status under two published rating standards, and how many samples
change category between them.
    ICAR-IISS (Harit Dhara 4(1), 2021): N <280 | 280-560 | >560 kg/ha;
                                         P <10 | 10-25 | >25;  K <120 | 120-280 | >280
    TNAU Agritech rating chart:          N <240 | 240-480 | >480 kg/ha;
                                         P <11 | 11-22 | >22;  K <110 | 110-280 | >280

Part C (function, no default agronomy): convert a nutrient dose
(N, P2O5, K2O kg/ha, taken from your state's package of practices or an STCR
equation) into DAP (18-46-0), urea (46-0-0) and MOP (0-0-60) quantities.

Outputs -> results/fertilizer/{tables,figures}
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

from common import set_style, savefig, save_table, CAT, HILITE, DIM, INK2, MUTED, SEQ, COL1, COL2, SEED

GRADES = {"Urea": (46, 0, 0), "DAP": (18, 46, 0), "14-35-14": (14, 35, 14), "17-17-17": (17, 17, 17),
          "20-20": (20, 20, 0), "28-28": (28, 28, 0), "10-26-26": (10, 26, 26)}
STANDARDS = {"ICAR-IISS": {"N": (280, 560), "P": (10, 25), "K": (120, 280)},
             "TNAU": {"N": (240, 480), "P": (11, 22), "K": (110, 280)}}


def rate(v, lo_hi):
    lo, hi = lo_hi
    return np.where(v < lo, "Low", np.where(v <= hi, "Medium", "High"))


def nutrient_to_products(N, P2O5, K2O):
    """Products for a nutrient dose in kg/ha: P2O5 from DAP first (which also
    supplies 18% N), remaining N from urea, K2O from MOP."""
    dap = P2O5 / 0.46
    urea = max(N - 0.18 * dap, 0) / 0.46
    mop = K2O / 0.60
    return {"DAP_kg_ha": round(dap, 1), "Urea_kg_ha": round(urea, 1), "MOP_kg_ha": round(mop, 1)}


def part_a(a, out):
    f = pd.read_csv(a.fert); f.columns = [c.strip() for c in f.columns]
    y_names = f["Fertilizer Name"]; cls = sorted(y_names.unique())
    y = np.array([cls.index(v) for v in y_names])
    num = {"N, P, K only": ["Nitrogen", "Phosphorous", "Potassium"],
           "Temperature, humidity, moisture": ["Temparature", "Humidity", "Moisture"]}
    Xall = pd.get_dummies(f.drop(columns="Fertilizer Name"), columns=["Soil Type", "Crop Type"]).astype(float)
    sets = {"All features": Xall.values, **{k: f[v].values.astype(float) for k, v in num.items()},
            "Soil type + crop type": pd.get_dummies(f[["Soil Type", "Crop Type"]]).astype(float).values}
    models = {"Decision Tree": DecisionTreeClassifier(random_state=SEED),
              "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
              "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
              "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10))}
    rows = []
    loo = LeaveOneOut()
    for sname, X in sets.items():
        for mname, m in models.items():
            pred = cross_val_predict(clone(m), X, y, cv=loo)
            rows.append({"feature_set": sname, "model": mname, "loo_accuracy": accuracy_score(y, pred),
                         "loo_macro_f1": f1_score(y, pred, average="macro")})
        print(f"[A] {sname} done", flush=True)
    res = pd.DataFrame(rows)
    majority = np.bincount(y).max() / len(y)
    res.loc[len(res)] = {"feature_set": "Majority class", "model": "-", "loo_accuracy": majority,
                         "loo_macro_f1": np.nan}
    save_table(res, out / "tables", "tab11_fertilizer_loo_by_feature_set")
    print(res.pivot(index="feature_set", columns="model", values="loo_accuracy").round(3))

    # grade mirroring: cosine similarity of class-median (N, P, K) with each grade
    med = f.groupby("Fertilizer Name")[["Nitrogen", "Phosphorous", "Potassium"]].median().loc[cls]
    G = np.array([GRADES[c] for c in cls], float)
    A = med.values / np.linalg.norm(med.values, axis=1, keepdims=True)
    B = G / np.linalg.norm(G, axis=1, keepdims=True)
    cos = A @ B.T
    cos_df = pd.DataFrame(cos, index=[f"rows labelled {c}" for c in cls], columns=[f"grade {c}" for c in cls])
    cos_df.to_csv(out / "tables" / "tab11_npk_vs_grade_cosine.csv")
    # ties count as matches: 20-20 and 28-28 share the 1:1:0 ratio, so cosine cannot separate them
    diag_best = np.mean([cos[i, i] >= cos[i].max() - 1e-9 for i in range(len(cls))])
    print(f"[A] classes whose N:P:K is (jointly) most similar to their own fertilizer grade: {diag_best:.0%}")
    pd.DataFrame({"share_of_classes_matching_own_grade": [diag_best],
                  "note": ["20-20 and 28-28 have identical N:P:K ratios; ties counted as matches"]}
                 ).to_csv(out / "tables" / "tab11_grade_match_summary.csv", index=False)
    tree = DecisionTreeClassifier(max_depth=4, random_state=SEED).fit(sets["N, P, K only"], y)
    (out / "tables" / "tab11_npk_rules.txt").write_text(
        export_text(tree, feature_names=["N", "P", "K"], class_names=cls))

    # figure
    set_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    order = ["All features", "N, P, K only", "Soil type + crop type", "Temperature, humidity, moisture"]
    xs = np.arange(len(order)); w = 0.2
    for i, (mname, c) in enumerate(zip(models, CAT)):
        v = res[res.model == mname].set_index("feature_set").reindex(order).loo_accuracy
        a1.bar(xs + (i - 1.5) * w, v, width=w * 0.92, color=c, label=mname)
    a1.axhline(majority, color=INK2, ls="--", lw=1, label=f"majority class ({majority:.2f})")
    a1.set_xticks(xs, ["All", "N, P, K", "Soil + crop\ntype", "Temp., humidity,\nmoisture"])
    a1.set_ylim(0, 1.05); a1.set_ylabel("Leave-one-out accuracy"); a1.grid(axis="x", visible=False)
    a1.legend(fontsize=6, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    a1.set_title("(a) What predicts the fertilizer label", loc="left", pad=30)
    im = a2.imshow(cos, cmap=SEQ, vmin=0, vmax=1)
    a2.set_xticks(range(len(cls)), cls, rotation=45, ha="right"); a2.set_yticks(range(len(cls)), cls)
    a2.set_xlabel("Product grade (N-P₂O₅-K₂O)"); a2.set_ylabel("Rows labelled"); a2.grid(False)
    for i in range(len(cls)):
        for j in range(len(cls)):
            a2.text(j, i, f"{cos[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if cos[i, j] > 0.6 else "#0b0b0b")
    a2.set_title("(b) Class N:P:K vs fertilizer grade (cosine)", loc="left", pad=30)
    fig.tight_layout()
    savefig(fig, out / "figures", "fig17_fertilizer_benchmark")


def part_b(a, out):
    s = pd.read_csv(a.shc)
    cols = {"N": a.col_n, "P": a.col_p, "K": a.col_k}
    rows = []
    for nut, col in cols.items():
        v = pd.to_numeric(s[col], errors="coerce").dropna().values
        r1, r2 = rate(v, STANDARDS["ICAR-IISS"][nut]), rate(v, STANDARDS["TNAU"][nut])
        for cat in ["Low", "Medium", "High"]:
            rows.append({"nutrient": nut, "category": cat, "share_ICAR_IISS": np.mean(r1 == cat),
                         "share_TNAU": np.mean(r2 == cat)})
        rows.append({"nutrient": nut, "category": "changes category", "share_ICAR_IISS": np.mean(r1 != r2),
                     "share_TNAU": np.mean(r1 != r2)})
    res = pd.DataFrame(rows)
    save_table(res, out / "tables", "tab11_shc_rating_standard_sensitivity")
    print(res.to_string(index=False))


def main(a):
    out = Path(a.out); (out / "tables").mkdir(parents=True, exist_ok=True)
    part_a(a, out)
    if a.shc:
        part_b(a, out)
    else:
        print("[B] skipped: pass --shc <Soil Health Card CSV> --col-n ... --col-p ... --col-k ... (kg/ha)")
    ex = nutrient_to_products(100, 52, 25)
    print(f"[C] example, STCR rice dose 100-52-25 kg/ha (ICAR-IISS Harit Dhara 4(1), 2021, Table 1): {ex}")
    pd.DataFrame([{"N": 100, "P2O5": 52, "K2O": 25, **ex,
                   "source_of_dose": "STCR-IPNS rice, target 6 t/ha, Coimbatore; ICAR-IISS Harit Dhara 4(1) 2021"}]
                 ).to_csv(out / "tables" / "tab11_product_conversion_example.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fert", default="data/Fertilizer_Prediction_99rows.csv")
    ap.add_argument("--shc", help="Soil Health Card sample-level CSV with numeric N, P, K in kg/ha")
    ap.add_argument("--col-n", default="nitrogen"); ap.add_argument("--col-p", default="phosphorous")
    ap.add_argument("--col-k", default="potassium")
    ap.add_argument("--out", default="results/fertilizer")
    main(ap.parse_args())
