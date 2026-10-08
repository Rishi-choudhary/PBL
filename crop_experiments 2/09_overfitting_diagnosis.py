"""Experiment 9 - Is >95% accuracy on the Kaggle crop dataset overfitting?

Overfitting means the model fits the training data much better than unseen
data. This script measures that directly, then tests the alternative
explanation: that the features were generated from the label.

  (a) train vs held-out accuracy for each model (5-fold stratified CV)
  (b) complexity curves: decision-tree depth and KNN k, train vs held-out
  (c) label-permutation test: accuracy when labels are shuffled
  (d) generator check: the dataset creator's notebook sets
        N = base_N[crop] + randint(-20, 20)
        P = base_P[crop] + randint(-5, 20)
        K = base_K[crop] + randint(-5, 5)
      (data/provenance/harvestify_aac427c). We verify every row against this
      rule and measure how well a zero-learning lookup of the base table
      identifies the crop from N, P, K alone.

Outputs -> results/kaggle/{tables,figures}
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from common import (load_kaggle, get_models, set_style, savefig, save_table,
                    CAT, HILITE, DIM, INK2, MUTED, COL2, SEED)

GAP_MODELS = ["Logistic Regression", "Gaussian NB", "KNN (k=5)", "SVM (RBF)", "Decision Tree", "Random Forest",
              "Extra Trees", "XGBoost", "LightGBM", "CatBoost", "MLP (128-64)"]
RULE = {"N": (-20, 20), "P": (-5, 20), "K": (-5, 5)}


def train_test_curve(make, params, X, y, cv):
    tr_s, te_s = [], []
    for p in params:
        a, b = [], []
        for tr, te in cv.split(X, y):
            m = make(p).fit(X[tr], y[tr])
            a.append(accuracy_score(y[tr], m.predict(X[tr]))); b.append(accuracy_score(y[te], m.predict(X[te])))
        tr_s.append(np.mean(a)); te_s.append(np.mean(b))
    return np.array(tr_s), np.array(te_s)


def main(a):
    set_style()
    df, X, y, le = load_kaggle(a.data)
    K = len(le.classes_)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out_t, out_f = f"{a.out}/tables", f"{a.out}/figures"

    # (a) train vs held-out ---------------------------------------------------------
    rows = []
    for name, proto in get_models(include=GAP_MODELS, n_classes=K).items():
        tr_acc, te_acc = [], []
        for tr, te in cv.split(X, y):
            m = clone(proto).fit(X[tr], y[tr])
            tr_acc.append(accuracy_score(y[tr], m.predict(X[tr]))); te_acc.append(accuracy_score(y[te], m.predict(X[te])))
        rows.append({"model": name, "train_accuracy": np.mean(tr_acc), "test_accuracy": np.mean(te_acc),
                     "gap_pp": 100 * (np.mean(tr_acc) - np.mean(te_acc))})
        print(f"{name:22s} train {np.mean(tr_acc):.4f}  test {np.mean(te_acc):.4f}", flush=True)
    gap = pd.DataFrame(rows).sort_values("test_accuracy")
    save_table(gap, out_t, "tab09_train_vs_test_gap")

    # (b) complexity curves --------------------------------------------------------------
    depths = list(range(1, 21))
    dt_tr, dt_te = train_test_curve(lambda d: DecisionTreeClassifier(max_depth=d, random_state=SEED), depths, X, y, cv)
    ks = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80]
    kn_tr, kn_te = train_test_curve(
        lambda k: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)), ks, X, y, cv)
    save_table(pd.DataFrame({"max_depth": depths, "train": dt_tr, "test": dt_te}), out_t, "tab09_tree_depth_curve")
    save_table(pd.DataFrame({"k": ks, "train": kn_tr, "test": kn_te}), out_t, "tab09_knn_k_curve")

    # (c) permutation test -----------------------------------------------------------------
    score, perm, p = permutation_test_score(
        RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED), X, y, cv=cv,
        n_permutations=a.permutations, random_state=SEED, n_jobs=1)
    save_table(pd.DataFrame({"real_score": [score], "perm_mean": [perm.mean()], "perm_max": [perm.max()],
                             "p_value": [p], "n_permutations": [a.permutations]}), out_t, "tab09_permutation_test")
    print(f"permutation test: real {score:.4f}, shuffled mean {perm.mean():.4f}, p = {p:.4f}")
    pd.DataFrame({"shuffled_label_accuracy": perm}).to_csv(f"{out_t}/tab09_permutation_scores.csv", index=False)

    # (d) generator check and zero-learning lookup ------------------------------------------
    base = pd.read_csv(a.base).drop(columns=["Unnamed: 0"], errors="ignore")
    base = base.set_index("Crop").loc[le.classes_]
    off = {f: df[f].values - base[f].reindex(df["label"]).values for f in RULE}
    ok = np.all([(off[f] >= lo) & (off[f] <= hi) for f, (lo, hi) in RULE.items()], axis=0)
    cand = np.zeros((len(df), K), bool)
    for j, c in enumerate(le.classes_):
        cand[:, j] = np.all([(df[f].values - base.loc[c, f] >= lo) & (df[f].values - base.loc[c, f] <= hi)
                             for f, (lo, hi) in RULE.items()], axis=0)
    n_cand = cand.sum(1)
    lookup_acc = np.mean(cand[np.arange(len(df)), y] / np.maximum(n_cand, 1))   # expected acc, random tie-break
    npk = [0, 1, 2]
    rf_npk = np.mean([accuracy_score(y[te], RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
                                     .fit(X[tr][:, npk], y[tr]).predict(X[te][:, npk])) for tr, te in cv.split(X, y)])
    gen = pd.DataFrame({
        "rows_consistent_with_generator": [int(ok.sum())], "rows_total": [len(df)],
        "distinct_N_offsets": [len(np.unique(off["N"]))], "distinct_P_offsets": [len(np.unique(off["P"]))],
        "distinct_K_offsets": [len(np.unique(off["K"]))],
        "rows_with_unique_NPK_box": [int((n_cand == 1).sum())], "median_candidates": [float(np.median(n_cand))],
        "lookup_expected_accuracy_NPK_only": [lookup_acc], "random_forest_cv_accuracy_NPK_only": [rf_npk]})
    save_table(gen, out_t, "tab09_generator_check")
    print(gen.T.to_string())

    # figure --------------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(COL2, 5.4))
    ax = axs[0, 0]
    yy = np.arange(len(gap))
    ax.hlines(yy, gap.test_accuracy, gap.train_accuracy, color=DIM, lw=2)
    ax.scatter(gap.train_accuracy, yy, color=CAT[1], s=18, label="training folds", zorder=3)
    ax.scatter(gap.test_accuracy, yy, color=CAT[0], s=18, label="held-out fold", zorder=3)
    ax.set_yticks(yy, gap.model); ax.set_xlabel("Accuracy (5-fold mean)"); ax.grid(axis="y", visible=False)
    ax.set_title("(a) Train vs held-out accuracy", loc="left"); ax.legend(loc="upper left", fontsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5)

    ax = axs[0, 1]
    ax.plot(depths, dt_tr, color=CAT[1], marker="o", ms=2.5, label="train")
    ax.plot(depths, dt_te, color=CAT[0], marker="o", ms=2.5, label="held-out")
    ax.set_xlabel("Decision-tree max depth"); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.02)
    ax.set_title("(b) Complexity curve: no train/test divergence", loc="left"); ax.legend(loc="lower right")

    ax = axs[1, 0]
    ax.hist(perm, bins=15, color=DIM, label=f"shuffled labels ({a.permutations} runs)")
    ax.axvline(score, color=CAT[0], lw=2, label=f"real labels ({score:.3f})")
    ax.axvline(1 / K, color=MUTED, ls="--", lw=1)
    ax.set_xlim(0, 1.02); ax.set_xlabel("Random Forest 5-fold accuracy"); ax.set_ylabel("Count")
    ax.set_title("(c) Label-permutation test", loc="left"); ax.legend(loc="upper center", fontsize=6.5)

    ax = axs[1, 1]
    vals, cnts = np.unique(n_cand, return_counts=True)
    ax.bar(vals, cnts / len(df), color=HILITE, width=0.6)
    for v, c in zip(vals, cnts):
        ax.text(v, c / len(df) + 0.01, f"{100 * c / len(df):.0f}%", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(vals); ax.set_xlabel("Crops whose generator range contains the row's N, P, K")
    ax.set_ylabel("Share of rows"); ax.set_ylim(0, 1)
    ax.set_title("(d) Lookup of the creator's base table", loc="left")
    ax.text(0.98, 0.95, f"expected accuracy of lookup: {100 * lookup_acc:.1f}%\n"
            f"Random Forest trained on N, P, K: {100 * rf_npk:.1f}%", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.8, color=INK2)
    fig.tight_layout()
    savefig(fig, out_f, "fig14_overfitting_vs_leakage")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Crop_recommendation.csv")
    ap.add_argument("--base", default="data/provenance/harvestify_aac427c/FertilizerData.csv")
    ap.add_argument("--out", default="results/kaggle")
    ap.add_argument("--permutations", type=int, default=100)
    main(ap.parse_args())
