"""Experiment 5 - Explanations and their stability.

For each of 10 train/test splits (5-fold x 2 repeats) a LightGBM model is
fitted and three feature-importance estimates are computed on the test fold:
  * SHAP (TreeExplainer), mean |SHAP| over samples and classes
  * permutation importance (drop in macro-F1, 5 permutations)
  * split-gain importance from the booster
Reported:
  * global importance with spread across splits
  * per-crop SHAP profile (which features drive each crop's prediction)
  * stability: Kendall's W (agreement of feature rankings across splits) for
    each method, and Kendall's tau between the methods' mean rankings.

Outputs -> results/kaggle/{tables,figures}
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.inspection import permutation_importance

from common import (load_kaggle, get_models, KAGGLE_FEATURES, set_style, savefig, save_table,
                    CAT, HILITE, DIM, INK2, SEQ, COL1, COL2, SEED)


def kendalls_w(rank_matrix):
    """rank_matrix: m raters (splits) x n items (features), ranks 1..n."""
    R = np.asarray(rank_matrix, float)
    m, n = R.shape
    S = ((R.sum(0) - m * (n + 1) / 2) ** 2).sum()
    return 12 * S / (m ** 2 * (n ** 3 - n))


def shap_values_3d(explainer, X, K):
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    if sv.ndim == 2:            # binary case
        sv = sv[..., None]
    return sv                   # (n, features, classes)


def main(a):
    import shap
    set_style()
    df, X, y, le = load_kaggle(a.data)
    K, F = len(le.classes_), len(KAGGLE_FEATURES)
    proto = get_models(include=["LightGBM"], n_classes=K)["LightGBM"]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    imp = {"SHAP": [], "Permutation": [], "Gain": []}
    per_crop = np.zeros((K, F)); per_crop_n = np.zeros(K)
    for split, (tr, te) in enumerate(cv.split(X, y)):
        m = clone(proto).fit(X[tr], y[tr])
        sv = shap_values_3d(shap.TreeExplainer(m), X[te], K)
        imp["SHAP"].append(np.abs(sv).mean(axis=(0, 2)))
        for c in range(K):                        # samples of crop c, output for crop c
            idx = y[te] == c
            per_crop[c] += np.abs(sv[idx, :, c]).sum(0); per_crop_n[c] += idx.sum()
        pi = permutation_importance(m, X[te], y[te], scoring="f1_macro", n_repeats=5, random_state=SEED)
        imp["Permutation"].append(pi.importances_mean)
        g = m.booster_.feature_importance(importance_type="gain")
        imp["Gain"].append(g / g.sum())
        print(f"split {split + 1}/10 done", flush=True)
    per_crop /= per_crop_n[:, None]

    # stability ---------------------------------------------------------------
    stab, mean_rank = [], {}
    for meth, vals in imp.items():
        V = np.array(vals)
        ranks = np.array([stats.rankdata(-v) for v in V])       # 1 = most important
        mean_rank[meth] = ranks.mean(0)
        pair_tau = [stats.kendalltau(ranks[i], ranks[j])[0] for i in range(len(V)) for j in range(i + 1, len(V))]
        stab.append({"method": meth, "kendalls_W_across_splits": kendalls_w(ranks),
                     "mean_pairwise_tau": np.mean(pair_tau), "min_pairwise_tau": np.min(pair_tau),
                     "top1_feature_modal": KAGGLE_FEATURES[np.bincount(ranks.argmin(1)).argmax()],
                     "top1_agreement_share": np.bincount(ranks.argmin(1)).max() / len(V)})
    stab = pd.DataFrame(stab)
    meths = list(imp)
    agree = pd.DataFrame([[stats.kendalltau(mean_rank[p], mean_rank[q])[0] for q in meths] for p in meths],
                         index=meths, columns=meths)
    save_table(stab, f"{a.out}/tables", "tab05_explanation_stability")
    save_table(agree.reset_index().rename(columns={"index": "method"}), f"{a.out}/tables",
               "tab05_method_agreement_tau")
    glob = pd.DataFrame({m: np.array(v).mean(0) for m, v in imp.items()}, index=KAGGLE_FEATURES)
    save_table(glob.reset_index().rename(columns={"index": "feature"}), f"{a.out}/tables",
               "tab05_global_importance")
    pd.DataFrame(per_crop, index=le.classes_, columns=KAGGLE_FEATURES).to_csv(
        f"{a.out}/tables/tab05_shap_per_crop.csv")
    print(stab.to_string(index=False)); print(agree.round(3))

    # figure: global importance per method (normalised), with split range -----
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.3), sharey=True)
    order = np.argsort(np.array(imp["SHAP"]).mean(0))
    for ax, meth in zip(axes, meths):
        V = np.array(imp[meth]); V = V / V.sum(1, keepdims=True)
        mu = V.mean(0)[order]
        ax.barh(np.arange(F), mu, color=HILITE, height=0.6)
        ax.hlines(np.arange(F), V.min(0)[order], V.max(0)[order], color=INK2, lw=0.9)
        ax.set_yticks(np.arange(F), np.array(KAGGLE_FEATURES)[order])
        ax.set_title(f"{meth} (W = {stab.set_index('method').loc[meth, 'kendalls_W_across_splits']:.2f})",
                     loc="left", fontsize=8)
        ax.set_xlabel("Share of total importance"); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    savefig(fig, f"{a.out}/figures", "fig11_global_importance_methods")

    # figure: per-crop SHAP profile (row-normalised) ---------------------------
    Pn = per_crop / per_crop.sum(1, keepdims=True)
    row_order = np.lexsort((-Pn.max(1), Pn.argmax(1)))
    fig, ax = plt.subplots(figsize=(COL1, 4.4))
    im = ax.imshow(Pn[row_order], cmap=SEQ, vmin=0, vmax=Pn.max(), aspect="auto")
    ax.set_xticks(range(F), KAGGLE_FEATURES, rotation=45, ha="right")
    ax.set_yticks(range(K), le.classes_[row_order]); ax.grid(False)
    for i, r in enumerate(row_order):
        j = Pn[r].argmax()
        ax.text(j, i, f"{Pn[r, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, label="Share of |SHAP| for that crop")
    ax.set_title("What drives each crop's prediction\n(LightGBM, mean |SHAP|, row-normalised)", fontsize=8)
    fig.tight_layout()
    savefig(fig, f"{a.out}/figures", "fig12_shap_per_crop")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Crop_recommendation.csv")
    ap.add_argument("--out", default="results/kaggle")
    main(ap.parse_args())
