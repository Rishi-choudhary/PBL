"""Experiment 4 - Robustness of the headline accuracy.

(a) Learning curve: macro-F1 vs number of training samples per crop. If a
    model reaches ~99% with a handful of samples per crop, the benchmark is
    saturated and cannot distinguish models.
(b) Measurement noise: Gaussian noise added to *test* features only, with
    standard deviation = k x (feature standard deviation in the training fold).
    Mimics soil-test / sensor error at deployment time.
(c) Feature ablation: drop one feature at a time, and soil-only vs climate-only.

Outputs -> results/kaggle/{tables,figures}
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from common import (load_kaggle, get_models, KAGGLE_FEATURES, set_style, savefig, save_table,
                    CAT, HILITE, DIM, INK2, MUTED, COL1, COL2, SEED)

CURVE_MODELS = ["Random Forest", "LightGBM", "SVM (RBF)", "Gaussian NB", "Logistic Regression"]
PER_CLASS = [2, 4, 8, 16, 32, 48, 64, 80]          # training samples per crop
NOISE = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]  # x feature std
SOIL = ["N", "P", "K", "ph"]; CLIMATE = ["temperature", "humidity", "rainfall"]


def main(a):
    set_style()
    df, X, y, le = load_kaggle(a.data)
    K = len(le.classes_)
    models = get_models(include=CURVE_MODELS, n_classes=K)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rng = np.random.default_rng(SEED)
    lc, nz = [], []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        # (a) learning curve: stratified subsample of the training fold
        for n in PER_CLASS:
            idx = np.concatenate([rng.choice(tr[y[tr] == c], size=min(n, (y[tr] == c).sum()), replace=False)
                                  for c in range(K)])
            for name, proto in models.items():
                m = clone(proto).fit(X[idx], y[idx])
                lc.append({"fold": fold, "per_class": n, "model": name,
                           "macro_f1": f1_score(y[te], m.predict(X[te]), average="macro")})
        # (b) noise: fit on clean training fold, perturb the test fold
        sd = X[tr].std(0)
        for name, proto in models.items():
            m = clone(proto).fit(X[tr], y[tr])
            for k in NOISE:
                for rep in range(3):
                    Xn = X[te] + rng.normal(0, 1, X[te].shape) * (k * sd)
                    nz.append({"fold": fold, "noise_k": k, "rep": rep, "model": name,
                               "macro_f1": f1_score(y[te], m.predict(Xn), average="macro")})
        print(f"fold {fold + 1}/5 done", flush=True)
    lc, nz = pd.DataFrame(lc), pd.DataFrame(nz)
    save_table(lc.groupby(["model", "per_class"]).macro_f1.agg(["mean", "std"]).reset_index(),
               f"{a.out}/tables", "tab04_learning_curve")
    save_table(nz.groupby(["model", "noise_k"]).macro_f1.agg(["mean", "std"]).reset_index(),
               f"{a.out}/tables", "tab04_noise_robustness")

    # (c) ablation with Random Forest and LightGBM -----------------------------
    sets = {"All 7 features": KAGGLE_FEATURES}
    sets.update({f"without {f}": [g for g in KAGGLE_FEATURES if g != f] for f in KAGGLE_FEATURES})
    sets["Soil only (N, P, K, pH)"] = SOIL
    sets["Climate only (temp., humidity, rain)"] = CLIMATE
    ab = []
    for name in ["Random Forest", "LightGBM"]:
        for sname, feats in sets.items():
            cols = [KAGGLE_FEATURES.index(f) for f in feats]
            sc = []
            for tr, te in cv.split(X, y):
                m = clone(models[name]).fit(X[tr][:, cols], y[tr])
                sc.append(f1_score(y[te], m.predict(X[te][:, cols]), average="macro"))
            ab.append({"model": name, "feature_set": sname, "macro_f1_mean": np.mean(sc),
                       "macro_f1_std": np.std(sc, ddof=1)})
    ab = pd.DataFrame(ab)
    save_table(ab, f"{a.out}/tables", "tab04_feature_ablation")

    # figure: learning curve + noise -------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.7))
    for c, name in zip(CAT, CURVE_MODELS):
        g = lc[lc.model == name].groupby("per_class").macro_f1.agg(["mean", "std"])
        a1.plot(g.index, g["mean"], color=c, marker="o", ms=3, label=name)
        h = nz[nz.model == name].groupby("noise_k").macro_f1.agg(["mean", "std"])
        a2.plot(h.index, h["mean"], color=c, marker="o", ms=3, label=name)
    a1.set_xscale("log", base=2); a1.set_xticks(PER_CLASS, PER_CLASS)
    a1.set_xlabel("Training samples per crop"); a1.set_ylabel("Macro-F1 (5-fold mean)")
    a1.set_title("(a) Learning curve", loc="left"); a1.set_ylim(0, 1.02)
    a2.set_xlabel("Test-time noise σ (× feature std. dev.)"); a2.set_ylabel("Macro-F1")
    a2.set_title("(b) Sensitivity to measurement noise", loc="left"); a2.set_ylim(0, 1.02)
    a2.legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    savefig(fig, f"{a.out}/figures", "fig09_learning_curve_and_noise")

    # figure: ablation (Random Forest) -------------------------------------------
    r = ab[ab.model == "Random Forest"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(COL1, 2.9))
    col = [HILITE if s == "All 7 features" else DIM for s in r.feature_set]
    ax.barh(r.feature_set, r.macro_f1_mean, xerr=r.macro_f1_std, color=col, height=0.65,
            error_kw={"lw": 0.8, "ecolor": INK2})
    for i, (v, fs) in enumerate(zip(r.macro_f1_mean, r.feature_set)):
        ax.text(0.02, i, f"{v:.3f}", va="center", fontsize=6.3,
                color="white" if fs == "All 7 features" else "#0b0b0b")
    ax.set_xlim(0, 1.02); ax.set_xlabel("Macro-F1, Random Forest (5-fold)")
    ax.set_title("Feature ablation", loc="left"); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=6.5)
    fig.tight_layout()
    savefig(fig, f"{a.out}/figures", "fig10_feature_ablation")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Crop_recommendation.csv")
    ap.add_argument("--out", default="results/kaggle")
    main(ap.parse_args())
