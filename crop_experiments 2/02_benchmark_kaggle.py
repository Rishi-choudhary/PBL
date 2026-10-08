"""Experiment 2 - Model benchmark on the Kaggle dataset with proper statistics.

Repeated stratified k-fold (default 5 x 5 = 25 train/test splits). For every
model and split: accuracy, balanced accuracy, macro P/R/F1, top-3 accuracy,
log-loss, Brier score, expected calibration error, fit time, predict time.
Differences against the best model are tested with the corrected resampled
t-test (Nadeau & Bengio 2003; Bouckaert & Frank 2004) with Holm correction,
because ordinary paired t-tests over overlapping CV folds are anti-conservative.

Outputs -> results/kaggle/{tables,figures}
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score

from common import (load_kaggle, get_models, all_metrics, timed_fit_predict,
                    corrected_repeated_cv_ttest, holm, mean_ci, set_style, savefig,
                    save_table, HILITE, DIM, INK2, MUTED, SEQ, COL1, COL2, SEED)

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall",
           "top3_accuracy", "log_loss", "brier", "ece"]


def run(a):
    df, X, y, le = load_kaggle(a.data)
    K = len(le.classes_)
    models = get_models(include=a.models.split(",") if a.models else None, n_classes=K,
                        with_tabpfn=a.tabpfn)
    cv = RepeatedStratifiedKFold(n_splits=a.folds, n_repeats=a.repeats, random_state=SEED)
    out = Path(a.out); (out / "raw").mkdir(parents=True, exist_ok=True)
    rows, oof = [], {name: np.zeros((len(y), K)) for name in models}
    t_start = time.time()
    for split, (tr, te) in enumerate(cv.split(X, y)):
        rep = split // a.folds
        for name, proto in models.items():
            P, fit_s, pred_s = timed_fit_predict(clone(proto), X[tr], y[tr], X[te], K)
            r = {"model": name, "split": split, "repeat": rep, "n_train": len(tr), "n_test": len(te),
                 "fit_s": fit_s, "predict_ms_per_sample": 1000 * pred_s / len(te)}
            r.update(all_metrics(y[te], P, K))
            rows.append(r)
            if rep == 0:
                oof[name][te] = P
        pd.DataFrame(rows).to_csv(out / "raw" / "benchmark_per_split.csv", index=False)
        print(f"split {split + 1}/{a.folds * a.repeats} done ({time.time() - t_start:.0f}s)", flush=True)
    np.savez_compressed(out / "raw" / "oof_repeat0.npz", y=y, classes=le.classes_,
                        **{n.replace(" ", "_"): p for n, p in oof.items()})
    return pd.DataFrame(rows), oof, y, le


def summarise(res, oof, y, le, a):
    set_style()
    out = Path(a.out); fig_dir, tab_dir = out / "figures", out / "tables"
    K = len(le.classes_)

    # summary table -------------------------------------------------------------
    agg = []
    for name, g in res.groupby("model", sort=False):
        r = {"model": name}
        for m in METRICS + ["fit_s", "predict_ms_per_sample"]:
            mean, lo, hi = mean_ci(g[m])
            r[f"{m}_mean"], r[f"{m}_std"], r[f"{m}_ci_lo"], r[f"{m}_ci_hi"] = mean, g[m].std(ddof=1), lo, hi
        agg.append(r)
    summ = pd.DataFrame(agg).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)

    # significance vs best ------------------------------------------------------
    best = summ.loc[0, "model"]
    n_tr, n_te = res["n_train"].iloc[0], res["n_test"].iloc[0]
    b = res[res.model == best].sort_values("split")["macro_f1"].values
    tests = []
    for name in summ.model:
        if name == best:
            continue
        o = res[res.model == name].sort_values("split")["macro_f1"].values
        t, p = corrected_repeated_cv_ttest(b, o, n_tr, n_te)
        tests.append({"model": name, "delta_macro_f1_vs_best": b.mean() - o.mean(), "t": t, "p_raw": p})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests["p_raw"].values)
    tests["significantly_worse_than_best"] = tests["p_holm"] < 0.05
    summ = summ.merge(tests[["model", "p_holm", "significantly_worse_than_best"]], on="model", how="left")
    save_table(summ, tab_dir, "tab02_benchmark_summary_full")
    save_table(tests, tab_dir, "tab02_significance_vs_best")

    # compact paper table
    def pm(m, s=100, d=2):
        return summ[f"{m}_mean"].mul(s).round(d).astype(str) + " ± " + summ[f"{m}_std"].mul(s).round(d).astype(str)
    paper = pd.DataFrame({
        "Model": summ.model, "Accuracy (%)": pm("accuracy"), "Macro-F1 (%)": pm("macro_f1"),
        "Top-3 (%)": pm("top3_accuracy"), "Log-loss": pm("log_loss", 1, 3), "ECE": pm("ece", 1, 3),
        "Fit time (s)": summ["fit_s_mean"].round(2),
        "Holm p vs best": summ["p_holm"].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}")})
    paper.to_csv(tab_dir / "tab02_benchmark_paper.csv", index=False)
    (tab_dir / "tab02_benchmark_paper.tex").write_text(paper.to_latex(index=False, escape=True))
    print(paper.to_string(index=False))

    # fig: macro-F1 with CI and log-loss --------------------------------------
    s = summ.iloc[::-1]
    color = [HILITE if (m == best or not w) else DIM
             for m, w in zip(s.model, s.significantly_worse_than_best.fillna(False))]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 3.6), sharey=True)
    ypos = np.arange(len(s))
    a1.hlines(ypos, s.macro_f1_ci_lo, s.macro_f1_ci_hi, color=color, lw=2)
    a1.scatter(s.macro_f1_mean, ypos, color=color, s=18, zorder=3)
    a1.set_yticks(ypos, s.model)
    a1.set_xlabel("Macro-F1 (mean, 95% CI over 25 splits)")
    a1.set_title("(a) Macro-F1 (axis zoomed)")
    a1.grid(axis="y", visible=False)
    a2.hlines(ypos, s.log_loss_ci_lo, s.log_loss_ci_hi, color=color, lw=2)
    a2.scatter(s.log_loss_mean, ypos, color=color, s=18, zorder=3)
    a2.set_xscale("log"); a2.set_xlabel("Log-loss (lower is better, log scale)")
    a2.set_title("(b) Probabilistic quality"); a2.grid(axis="y", visible=False)
    fig.text(0.01, -0.02, f"Blue: best model ({best}) and models not significantly worse "
             f"(corrected resampled t-test, Holm-adjusted p ≥ 0.05). Grey: significantly worse.",
             fontsize=6.5, color=INK2)
    fig.tight_layout()
    savefig(fig, fig_dir, "fig04_model_comparison")

    # fig: fit time vs macro-F1 -------------------------------------------------
    fig, ax = plt.subplots(figsize=(COL1, 2.8))
    ax.scatter(summ.fit_s_mean, summ.macro_f1_mean, s=16, color=HILITE)
    srt = summ.sort_values("fit_s_mean")
    front, best_so_far = [], -1
    for _, r in srt.iterrows():
        if r.macro_f1_mean > best_so_far:
            front.append(r); best_so_far = r.macro_f1_mean
    front = pd.DataFrame(front)
    ax.plot(front.fit_s_mean, front.macro_f1_mean, color=MUTED, lw=1, ls="--", zorder=1)
    short = lambda s: s.split(" (")[0]
    xmax = summ.fit_s_mean.max()
    for _, r in front.iterrows():
        right_edge = r.fit_s_mean >= xmax / 3
        left_edge = r.fit_s_mean == summ.fit_s_mean.min()
        ax.annotate(short(r.model), (r.fit_s_mean, r.macro_f1_mean), textcoords="offset points",
                    xytext=(0, 6), ha="right" if right_edge else ("left" if left_edge else "center"),
                    fontsize=6, color=INK2)
    ax.set_ylim(top=summ.macro_f1_mean.max() + 0.003)
    ax.set_xscale("log"); ax.set_xlabel("Mean fit time per split (s, log scale)")
    ax.set_ylabel("Macro-F1"); ax.set_title("Accuracy–cost trade-off (dashed: Pareto front)")
    fig.tight_layout()
    savefig(fig, fig_dir, "fig05_cost_vs_f1")

    # fig: confusion matrix of best model (repeat 0 out-of-fold) ---------------
    pred = oof[best].argmax(1)
    cm = confusion_matrix(y, pred, normalize="true")
    fig, ax = plt.subplots(figsize=(COL2 * 0.62, COL2 * 0.55))
    ax.imshow(cm, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(K), le.classes_, rotation=90); ax.set_yticks(range(K), le.classes_)
    ax.grid(False)
    for i in range(K):
        for j in range(K):
            if cm[i, j] >= 0.005 and i != j:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=5, color="#0b0b0b")
    ax.set_xlabel("Predicted crop"); ax.set_ylabel("True crop")
    ax.set_title(f"Row-normalised confusion matrix, {best} (out-of-fold)\noff-diagonal errors labelled")
    fig.tight_layout()
    savefig(fig, fig_dir, "fig06_confusion_best")

    # fig: per-class F1 heatmap, models x crops -------------------------------
    names = list(summ.model)
    F = np.array([f1_score(y, oof[n].argmax(1), average=None, labels=range(K)) for n in names])
    col_order = np.argsort(F.mean(0))
    fig, ax = plt.subplots(figsize=(COL2, 3.8))
    im = ax.imshow(F[:, col_order], cmap=SEQ, vmin=max(0.5, F.min()), vmax=1, aspect="auto")
    ax.set_xticks(range(K), le.classes_[col_order], rotation=90); ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="F1 (out-of-fold, repeat 0)")
    ax.set_title("Per-crop F1 by model (crops ordered hardest → easiest)")
    fig.tight_layout()
    savefig(fig, fig_dir, "fig07_per_crop_f1")
    pd.DataFrame(F, index=names, columns=le.classes_).to_csv(tab_dir / "tab02_per_crop_f1.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Crop_recommendation.csv")
    ap.add_argument("--out", default="results/kaggle")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--models", default="", help="comma-separated subset of model names")
    ap.add_argument("--tabpfn", action="store_true", help="add TabPFN (needs tabpfn + internet)")
    ap.add_argument("--summarise-only", action="store_true", help="re-plot from saved raw results")
    a = ap.parse_args()
    if a.summarise_only:
        res = pd.read_csv(Path(a.out) / "raw" / "benchmark_per_split.csv")
        z = np.load(Path(a.out) / "raw" / "oof_repeat0.npz", allow_pickle=True)
        df, X, y, le = load_kaggle(a.data)
        oof = {n: z[n.replace(" ", "_")] for n in res.model.unique()}
        summarise(res, oof, y, le, a)
    else:
        res, oof, y, le = run(a)
        summarise(res, oof, y, le, a)
