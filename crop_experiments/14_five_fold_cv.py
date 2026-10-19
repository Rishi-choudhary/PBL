"""Experiment 14 - 5-fold cross-validation of the crop and the fertilizer algorithms (mentor request).

Experiment 12 ran 5-fold CV on the 1,760 training rows only (crop) and 5x10 repeated CV (fertilizer).
Here every algorithm is evaluated by plain stratified 5-fold CV on ALL rows of its dataset: the data is
split into 5 folds, each fold is the test set once, and the other 4 folds train the model. Every row is
tested exactly once. Scalers and one-hot encoders sit inside the pipeline, so they are fitted on the 4
training folds only. The standard deviation is the population sd (np.std), as in experiment 12.

Outputs -> results/system/tables/tabS11_*, tabS12_*, results/system/figures/figS7_*, results/system/summary_cv5.txt
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from common import load_kaggle, set_style, savefig, save_table, CAT, HILITE, DIM, INK2, MUTED, COL2, SEED

spec = importlib.util.spec_from_file_location("crop_system", Path(__file__).with_name("12_crop_system.py"))
crop_system = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crop_system)

N_FOLDS = 5


def sub(a, idx):
    return a.iloc[idx] if hasattr(a, "iloc") else a[idx]


def five_fold(models, X, y):
    """Per-fold metrics for every model, and the out-of-fold prediction of every row."""
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    rows, oof = [], {}
    for name, model in models.items():
        pred = np.empty_like(y)
        for k, (tr, te) in enumerate(cv.split(X, y), 1):
            m = clone(model)
            t0 = time.perf_counter(); m.fit(sub(X, tr), y[tr]); fit_t = time.perf_counter() - t0
            p = m.predict(sub(X, te)); pred[te] = p
            rows.append(dict(model=name, fold=k, n_test=len(te), accuracy=accuracy_score(y[te], p),
                             precision=precision_score(y[te], p, average="macro", zero_division=0),
                             recall=recall_score(y[te], p, average="macro", zero_division=0),
                             f1=f1_score(y[te], p, average="macro", zero_division=0), fit_time_s=fit_t))
        print(f"  {name:26s} mean accuracy {np.mean([r['accuracy'] for r in rows if r['model'] == name]):.4f}")
        oof[name] = pred
    return pd.DataFrame(rows), oof


def summarise(per_fold):
    g = per_fold.groupby("model", sort=False)
    out = pd.DataFrame(dict(
        accuracy_mean=g.accuracy.mean(), accuracy_std=g.accuracy.apply(np.std),
        precision_mean=g.precision.mean(), recall_mean=g.recall.mean(),
        f1_mean=g.f1.mean(), f1_std=g.f1.apply(np.std),
        accuracy_min_fold=g.accuracy.min(), accuracy_max_fold=g.accuracy.max(),
        fit_time_s_mean=g.fit_time_s.mean())).reset_index()
    return out.sort_values("accuracy_mean", ascending=False).reset_index(drop=True)


def dots(ax, per_fold, summ, xlim, title, label_off_scale=True):
    """Dot plot: one row per algorithm, one dot per fold, diamond = mean over the 5 folds."""
    order = summ.model.tolist()[::-1]
    best = summ.accuracy_mean.max()
    for yy, name in enumerate(order):
        v = per_fold[per_fold.model == name].accuracy.values * 100
        mean = v.mean()
        ax.plot([v.min(), v.max()], [yy, yy], color=DIM, lw=1.2, zorder=1, solid_capstyle="round")
        ax.scatter(v, [yy] * len(v), s=14, color=CAT[1], zorder=2, edgecolor="white", linewidth=0.6)
        ax.scatter([mean], [yy], s=34, marker="D", color=HILITE if mean / 100 == best else INK2, zorder=3,
                   edgecolor="white", linewidth=0.8)
        if mean < xlim[0] and label_off_scale:
            ax.text(xlim[0] + 0.4, yy, f"{mean:.1f} (off scale)", va="center", fontsize=6, color=INK2)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    ax.set_xlim(*xlim); ax.set_xlabel("Accuracy (%)"); ax.set_title(title)
    ax.grid(axis="y", visible=False)


def main(a):
    set_style()
    out = Path(a.out); fig_dir, tab_dir = out / "figures", out / "tables"
    log = [f"{N_FOLDS}-fold stratified CV, shuffle, seed {SEED}; each row is tested exactly once"]

    # ----------------------------------------------------------------- crop
    print("[1] crop, 5-fold CV on all rows")
    df, X, y, le = load_kaggle(a.crop)
    crop_pf, oof = five_fold(crop_system.crop_models(), X, y)
    crop_sum = summarise(crop_pf)
    save_table(crop_pf, tab_dir, "tabS11_crop_5fold_per_fold")
    save_table(crop_sum, tab_dir, "tabS11_crop_5fold_summary")
    holdout = pd.read_csv(tab_dir / "tabS2_crop_model_comparison.csv").set_index("model")
    log.append(f"crop dataset: {len(y)} rows, folds of {sorted(int(v) for v in crop_pf.n_test.unique())} rows; models: {len(crop_sum)}")
    log.append("crop, 5-fold accuracy mean+-sd (min..max fold) | macro F1 | hold-out test accuracy (tabS2):")
    for _, r in crop_sum.iterrows():
        log.append(f"  {r.model:26s} {r.accuracy_mean:.4f}+-{r.accuracy_std:.4f} ({r.accuracy_min_fold:.4f}..{r.accuracy_max_fold:.4f})"
                   f" | F1 {r.f1_mean:.4f} | hold-out {holdout.loc[r.model, 'test_accuracy']:.4f}")
    top = crop_sum[crop_sum.accuracy_mean == crop_sum.accuracy_mean.max()].model.tolist()
    log.append(f"highest 5-fold accuracy: {top}")
    rf = oof["Random Forest"]
    wrong = [(le.classes_[t], le.classes_[p]) for t, p in zip(y, rf) if t != p]
    log.append(f"Random Forest out-of-fold: {len(wrong)} of {len(y)} rows wrong; (actual -> predicted): "
               + ", ".join(f"{t}->{p}" for t, p in wrong))

    # ----------------------------------------------------------------- fertilizer
    print("[2] fertilizer, 5-fold CV on all rows")
    fd = pd.read_csv(a.fert); fd.columns = [c.strip() for c in fd.columns]
    num = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Phosphorous", "Potassium"]
    cat = ["Soil Type", "Crop Type"]
    Xf, yf = fd[num + cat], LabelEncoder().fit_transform(fd["Fertilizer Name"])

    def pre(scale):
        return ColumnTransformer([("num", StandardScaler() if scale else "passthrough", num),
                                  ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    fert_pf, _ = five_fold(crop_system.fert_models(pre), Xf, yf)
    fert_sum = summarise(fert_pf)
    rep = pd.read_csv(tab_dir / "tabS7_fertilizer_model_comparison.csv").set_index("model")
    fert_sum["repeated_5x10_accuracy_mean"] = fert_sum.model.map(rep.accuracy_mean)
    save_table(fert_pf, tab_dir, "tabS12_fert_5fold_per_fold")
    save_table(fert_sum, tab_dir, "tabS12_fert_5fold_summary")
    log.append(f"fertilizer dataset: {len(yf)} rows, folds of {sorted(int(v) for v in fert_pf.n_test.unique())} rows; models: {len(fert_sum)}")
    log.append("fertilizer, 5-fold accuracy mean+-sd (min..max fold) | macro F1 | 5x10 repeated CV accuracy (tabS7):")
    for _, r in fert_sum.iterrows():
        log.append(f"  {r.model:26s} {r.accuracy_mean:.4f}+-{r.accuracy_std:.4f} ({r.accuracy_min_fold:.4f}..{r.accuracy_max_fold:.4f})"
                   f" | F1 {r.f1_mean:.4f} | 5x10 {r.repeated_5x10_accuracy_mean:.4f}")

    # ----------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.9), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.62))
    dots(axes[0], crop_pf, crop_sum, (95, 100.4), "(a) Crop: 10 algorithms, 2,200 rows")
    dots(axes[1], fert_pf, fert_sum, (55, 101), "(b) Fertilizer: 8 algorithms, 99 rows")
    h = [plt.Line2D([], [], marker="o", ls="", color=CAT[1], ms=4, label="One fold (5 per algorithm)"),
         plt.Line2D([], [], marker="D", ls="", color=INK2, ms=5, label="Mean of the 5 folds"),
         plt.Line2D([], [], marker="D", ls="", color=HILITE, ms=5, label="Best mean")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.115), fontsize=6.5)
    savefig(fig, fig_dir, "figS7_five_fold_cv")

    (out / "summary_cv5.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", default="data/Crop_recommendation.csv")
    ap.add_argument("--fert", default="data/Fertilizer_Prediction_99rows.csv")
    ap.add_argument("--out", default="results/system")
    main(ap.parse_args())
