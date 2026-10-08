"""Experiment 6 - Calibration and conformal prediction sets.

(a) Calibration: expected calibration error (ECE) of out-of-fold probabilities
    for several models, before and after isotonic recalibration.
(b) Split conformal prediction (LAC and APS scores) on 20 random
    train/calibration/test splits (60/20/20, stratified): empirical coverage,
    mean set size and worst per-crop coverage across target error rates alpha.
    The marginal guarantee is P(true crop in set) >= 1 - alpha.

Outputs -> results/kaggle/{tables,figures}
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV

from common import (load_kaggle, get_models, aligned_proba, ece_score, conformal_quantile, lac_scores,
                    lac_sets, aps_scores, aps_sets, set_style, savefig, save_table,
                    CAT, HILITE, DIM, INK2, MUTED, COL1, COL2, SEED)

CAL_MODELS = ["Random Forest", "LightGBM", "SVM (RBF)", "Gaussian NB", "MLP (128-64)"]
ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]


def main(a):
    set_style()
    df, X, y, le = load_kaggle(a.data)
    K = len(le.classes_)
    models = get_models(include=CAL_MODELS, n_classes=K)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # (a) calibration -------------------------------------------------------------
    rows, oof = [], {}
    for name, proto in models.items():
        P_raw, P_iso = np.zeros((len(y), K)), np.zeros((len(y), K))
        for tr, te in cv.split(X, y):
            P_raw[te] = aligned_proba(clone(proto).fit(X[tr], y[tr]), X[te], K)
            iso = CalibratedClassifierCV(clone(proto), method="isotonic", cv=3).fit(X[tr], y[tr])
            P_iso[te] = aligned_proba(iso, X[te], K)
        oof[name] = P_raw
        rows.append({"model": name, "ece_raw": ece_score(y, P_raw), "ece_isotonic": ece_score(y, P_iso),
                     "mean_confidence": P_raw.max(1).mean(), "accuracy": (P_raw.argmax(1) == y).mean()})
        print(f"calibration {name} done", flush=True)
    cal = pd.DataFrame(rows)
    save_table(cal, f"{a.out}/tables", "tab06_calibration_ece")

    # (b) conformal ----------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    conf_rows = []
    for name in ["LightGBM", "Random Forest", "Gaussian NB"]:
        for rep in range(20):
            Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=rep)
            Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=rep)
            m = clone(models[name]).fit(Xtr, ytr)
            Pc, Pt = aligned_proba(m, Xcal, K), aligned_proba(m, Xte, K)
            for score in ["LAC", "APS"]:
                for al in ALPHAS:
                    if score == "LAC":
                        q = conformal_quantile(lac_scores(Pc, ycal), al); S = lac_sets(Pt, q)
                    else:
                        q = conformal_quantile(aps_scores(Pc, ycal, rng), al); S = aps_sets(Pt, q, rng)
                    covered = S[np.arange(len(yte)), yte]; size = S.sum(1)
                    per_class = [covered[yte == c].mean() for c in range(K)]
                    conf_rows.append({"model": name, "score": score, "alpha": al, "rep": rep,
                                      "coverage": covered.mean(), "mean_set_size": size.mean(),
                                      "singleton_share": (size == 1).mean(), "empty_share": (size == 0).mean(),
                                      "worst_crop_coverage": np.min(per_class)})
        print(f"conformal {name} done", flush=True)
    cf = pd.DataFrame(conf_rows)
    summ = cf.groupby(["model", "score", "alpha"]).agg(
        coverage=("coverage", "mean"), coverage_sd=("coverage", "std"),
        mean_set_size=("mean_set_size", "mean"), singleton_share=("singleton_share", "mean"),
        empty_share=("empty_share", "mean"), worst_crop_coverage=("worst_crop_coverage", "mean")).reset_index()
    save_table(summ, f"{a.out}/tables", "tab06_conformal_summary")
    print(summ[summ.model == "LightGBM"].to_string(index=False))

    # figure: ECE bars + conformal coverage / set size -----------------------------
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(COL2, 2.5), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    c = cal.sort_values("ece_raw")
    yy = np.arange(len(c))
    a1.barh(yy + 0.18, c.ece_raw, height=0.34, color=CAT[0], label="raw")
    a1.barh(yy - 0.18, c.ece_isotonic, height=0.34, color=CAT[1], label="isotonic")
    a1.set_yticks(yy, c.model); a1.set_xlabel("ECE (lower is better)")
    a1.set_title("(a) Calibration error", loc="left"); a1.legend(loc="lower right"); a1.grid(axis="y", visible=False)
    s = summ[(summ.model == "LightGBM")]
    for col, sc in zip(CAT[:2], ["LAC", "APS"]):
        t = s[s.score == sc]
        a2.plot(1 - t.alpha, t.coverage, color=col, marker="o", ms=3, label=sc)
        a3.plot(t.alpha, t.mean_set_size, color=col, marker="o", ms=3, label=sc)
    lim = [0.78, 1.0]
    a2.plot(lim, lim, color=MUTED, ls="--", lw=1, label="target")
    a2.set_xlim(lim); a2.set_ylim(lim)
    a2.set_xlabel("Target coverage 1 − α"); a2.set_ylabel("Empirical coverage")
    a2.set_title("(b) Conformal coverage", loc="left"); a2.legend(loc="lower right")
    a3.set_xlabel("α"); a3.set_ylabel("Mean prediction-set size")
    a3.set_title("(c) Set size (LightGBM)", loc="left"); a3.set_ylim(0, max(2.0, s.mean_set_size.max() * 1.1))
    a3.legend(loc="upper right")
    fig.tight_layout()
    savefig(fig, f"{a.out}/figures", "fig13_calibration_conformal")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Crop_recommendation.csv")
    ap.add_argument("--out", default="results/kaggle")
    main(ap.parse_args())
