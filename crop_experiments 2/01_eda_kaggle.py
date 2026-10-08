"""Experiment 1 - Diagnostics of the Kaggle crop recommendation benchmark.

Questions answered (each maps to a paragraph in the Dataset section):
  1. Is the dataset balanced / are there duplicates?
  2. How separable are the classes? (feature ranges per crop, accuracy of
     very shallow trees and of single-feature trees)
  3. Do per-class feature values look like field measurements or like samples
     drawn from simple distributions? (KS tests against uniform and normal)
  4. Correlation between features.

Outputs -> results/kaggle/figures, results/kaggle/tables
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from common import (load_kaggle, KAGGLE_FEATURES, set_style, savefig, save_table,
                    HILITE, DIM, INK2, MUTED, DIV, COL1, COL2, SEED)

UNITS = {"N": "N (ratio)", "P": "P (ratio)", "K": "K (ratio)", "temperature": "Temp. (°C)",
         "humidity": "Humidity (%)", "ph": "Soil pH", "rainfall": "Rainfall (mm)"}


def main(a):
    set_style()
    fig_dir, tab_dir = f"{a.out}/figures", f"{a.out}/tables"
    df, X, y, le = load_kaggle(a.data)
    crops = list(le.classes_)
    lines = []

    # 1. balance and duplicates -------------------------------------------------
    counts = df["label"].value_counts()
    lines.append(f"rows={len(df)}, classes={len(crops)}, per-class min={counts.min()} max={counts.max()}")
    lines.append(f"exact duplicate rows={df.duplicated().sum()}, duplicate feature vectors="
                 f"{df.duplicated(subset=KAGGLE_FEATURES).sum()}, missing values={int(df.isna().sum().sum())}")

    # 2a. feature range per crop (median dot, 5-95 percentile bar) ----------------
    order = df.groupby("label")["rainfall"].median().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 7, figsize=(COL2, 4.6), sharey=True)
    ypos = np.arange(len(order))
    for ax, f in zip(axes, KAGGLE_FEATURES):
        g = df.groupby("label")[f]
        lo, med, hi = g.quantile(0.05)[order], g.median()[order], g.quantile(0.95)[order]
        mn, mx = g.min()[order], g.max()[order]
        ax.hlines(ypos, mn, mx, color=DIM, lw=1.0)
        ax.hlines(ypos, lo, hi, color=HILITE, lw=2.4)
        ax.plot(med, ypos, "o", ms=3.2, color="#0d366b")
        ax.set_title(UNITS[f], fontsize=7)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=6)
    axes[0].set_yticks(ypos, order)
    axes[0].invert_yaxis()
    fig.suptitle("Per-crop feature ranges in the Kaggle benchmark (grey: min–max, blue: 5–95th pct., dot: median)",
                 fontsize=8, y=1.0)
    fig.tight_layout()
    savefig(fig, fig_dir, "fig01_feature_ranges_by_crop")

    # 2b. separability: CV accuracy of trees by depth, and of single-feature trees
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    depths = list(range(1, 16))
    dacc = [cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=SEED), X, y, cv=cv).mean()
            for d in depths]
    single = {f: cross_val_score(DecisionTreeClassifier(random_state=SEED), X[:, [i]], y, cv=cv).mean()
              for i, f in enumerate(KAGGLE_FEATURES)}
    pair_best = None
    for i in range(7):
        for j in range(i + 1, 7):
            s = cross_val_score(DecisionTreeClassifier(random_state=SEED), X[:, [i, j]], y, cv=cv).mean()
            if pair_best is None or s > pair_best[2]:
                pair_best = (KAGGLE_FEATURES[i], KAGGLE_FEATURES[j], s)
    sep = pd.DataFrame({"max_depth": depths, "cv_accuracy": dacc})
    save_table(sep, tab_dir, "tab01_tree_depth_vs_accuracy")
    save_table(pd.DataFrame({"feature": list(single), "cv_accuracy_single_feature_tree": list(single.values())}),
               tab_dir, "tab01_single_feature_accuracy")
    lines.append(f"best two-feature tree: {pair_best[0]}+{pair_best[1]} -> CV accuracy {pair_best[2]:.4f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.3), gridspec_kw={"width_ratios": [1.3, 1]})
    a1.plot(depths, dacc, color=HILITE, marker="o", ms=3.5)
    a1.axhline(1 / len(crops), color=MUTED, lw=1, ls="--")
    a1.text(15, 1 / len(crops) + 0.02, "chance (1/22)", ha="right", color=MUTED, fontsize=6.5)
    for d in (5, 9, 15):
        a1.annotate(f"{dacc[d-1]:.3f}", (d, dacc[d-1]), textcoords="offset points", xytext=(4, -10),
                    fontsize=6.5, color=INK2)
    a1.set_xlabel("Maximum tree depth"); a1.set_ylabel("5-fold CV accuracy")
    a1.set_ylim(0, 1.02); a1.set_title("(a) One decision tree, accuracy vs depth")
    s = pd.Series(single).sort_values()
    a2.barh(s.index, s.values, color=HILITE, height=0.6)
    for i, v in enumerate(s.values):
        a2.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=6.5, color=INK2)
    a2.set_xlim(0, 1); a2.set_xlabel("5-fold CV accuracy (one feature only)")
    a2.set_title("(b) Single-feature decision trees"); a2.grid(axis="y", visible=False)
    fig.tight_layout()
    savefig(fig, fig_dir, "fig02_separability")

    # 3. distribution tests per crop x feature --------------------------------
    rows = []
    for c in crops:
        sub = df[df.label == c]
        for f in KAGGLE_FEATURES:
            v = np.sort(sub[f].values.astype(float))
            lo, hi = v[0], v[-1]
            # Conditional on the sample min and max, the remaining n-2 points of a
            # uniform sample are iid U(min, max); testing only the interior points
            # avoids the bias of estimating the range from the same data.
            inner = v[1:-1]
            p_unif = stats.kstest((inner - lo) / (hi - lo), "uniform").pvalue if hi > lo else np.nan
            p_norm = stats.shapiro(v).pvalue
            rows.append({"crop": c, "feature": f, "n_unique": len(np.unique(v)),
                         "ks_uniform_p": p_unif, "shapiro_normal_p": p_norm})
    dist = pd.DataFrame(rows)
    save_table(dist, tab_dir, "tab01_distribution_tests")
    summ = dist.groupby("feature").agg(
        frac_consistent_uniform=("ks_uniform_p", lambda p: (p > 0.05).mean()),
        frac_consistent_normal=("shapiro_normal_p", lambda p: (p > 0.05).mean()),
        median_unique_values=("n_unique", "median")).reset_index()
    save_table(summ, tab_dir, "tab01_distribution_summary")
    lines.append("share of crop x feature cells NOT rejected (p>0.05):\n" + summ.to_string(index=False))

    # 4. correlation ------------------------------------------------------------
    corr = df[KAGGLE_FEATURES].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(COL1, 3.0))
    im = ax.imshow(corr.values, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(7), KAGGLE_FEATURES, rotation=45, ha="right"); ax.set_yticks(range(7), KAGGLE_FEATURES)
    ax.grid(False)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(corr.values[i, j]) > 0.6 else "#0b0b0b")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")
    ax.set_title("Feature correlation (Kaggle benchmark)")
    fig.tight_layout()
    savefig(fig, fig_dir, "fig03_feature_correlation")

    rep = "\n".join(lines)
    open(f"{a.out}/eda_summary.txt", "w").write(rep)
    print(rep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Crop_recommendation.csv")
    ap.add_argument("--out", default="results/kaggle")
    main(ap.parse_args())
