"""Experiment 7 - Does crop recommendation generalise across space and time?

Runs on the district-level dataset built by data/build_dataset.py
(ICRISAT-TCI District Level Database crop outcomes + Soil Health Card soil
features + climate). Feature columns are every column starting with the given
prefixes (default soil_ and clim_).

Tasks
  top1        multiclass: the crop with the largest area share in a district-year
  multilabel  every crop holding >= threshold share (grown_* columns)

Validation regimes (same models, same metrics)
  random      5-fold KFold over district-years (what most papers report)
  district    5-fold GroupKFold, all years of a district held out together
  state       leave-one-state-out (20 folds)
  temporal    train on years <= cutoff, test on later years

Extras
  * split-conformal prediction sets (LAC, alpha=0.10) whose calibration data
    come from the training side of each regime -> coverage under shift
  * SHAP stability across held-out states vs across random folds (Kendall's W)

--smoke-test fabricates soil/climate features only to check that the pipeline
runs end to end. Its numbers are meaningless and every figure is stamped.

Outputs -> results/district/{tables,figures}
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import (f1_score, accuracy_score, top_k_accuracy_score, average_precision_score,
                             hamming_loss)

from common import (get_models, conformal_quantile, lac_scores, set_style, savefig, save_table,
                    CAT, HILITE, DIM, INK2, MUTED, SEQ, COL1, COL2, SEED)

TOP1_MODELS = ["Logistic Regression", "Gaussian NB", "KNN (k=5)", "Random Forest", "Extra Trees",
               "XGBoost", "LightGBM", "CatBoost", "MLP (128-64)"]
ML_MODELS = ["Logistic Regression", "KNN (k=5)", "Random Forest", "Extra Trees", "LightGBM"]
REGIMES = ["random", "district", "state", "temporal"]
REGIME_LABEL = {"random": "Random k-fold", "district": "Leave-district-out",
                "state": "Leave-state-out", "temporal": "Temporal hold-out"}
STAMP = ""


class Encoded(BaseEstimator, ClassifierMixin):
    """Imputes, label-encodes per fold (XGBoost needs contiguous labels when a
    class is absent from a training fold) and exposes global classes_."""
    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.le_ = LabelEncoder().fit(y)
        self.model_ = make_pipeline(SimpleImputer(strategy="median"), clone(self.est))
        self.model_.fit(X, self.le_.transform(y))
        self.classes_ = self.le_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class PerLabel(BaseEstimator):
    """Multi-label wrapper. Tree ensembles and KNN are fitted once in native
    multi-output mode (for RF/ET this means splits use the joint impurity over
    all crops - say so in the paper); other models get one binary model per
    label, with a constant probability when a label has a single class in the
    training fold. predict_proba -> (n, L)."""
    def __init__(self, est):
        self.est = est

    NATIVE = ("RandomForestClassifier", "ExtraTreesClassifier", "KNeighborsClassifier", "DecisionTreeClassifier")

    def _is_native(self):
        e = self.est
        while hasattr(e, "steps"):
            e = e.steps[-1][1]
        return type(e).__name__ in self.NATIVE

    def fit(self, X, Y):
        self.native_ = None
        if self._is_native():   # one multi-output fit instead of one fit per label
            self.native_ = make_pipeline(SimpleImputer(strategy="median"), clone(self.est)).fit(X, Y)
            return self
        self.models_ = []
        for j in range(Y.shape[1]):
            yj = Y[:, j]
            if yj.min() == yj.max():
                self.models_.append(float(yj[0]))
            else:
                self.models_.append(make_pipeline(SimpleImputer(strategy="median"), clone(self.est)).fit(X, yj))
        return self

    def predict_proba(self, X):
        if self.native_ is not None:
            P, cls = self.native_.predict_proba(X), self.native_.classes_
            out = np.zeros((X.shape[0], len(P)))
            for j, (p, c) in enumerate(zip(P, cls)):
                c = list(c)
                out[:, j] = p[:, c.index(1)] if 1 in c else 0.0
            return out
        out = np.zeros((X.shape[0], len(self.models_)))
        for j, m in enumerate(self.models_):
            out[:, j] = m if isinstance(m, float) else m.predict_proba(X)[:, 1]
        return out


def full_proba(model, X, K):
    P = model.predict_proba(X)
    out = np.zeros((X.shape[0], K))
    out[:, model.classes_.astype(int)] = P
    return out


def make_splits(df, regime, cutoff):
    idx = np.arange(len(df))
    if regime == "random":
        return list(KFold(5, shuffle=True, random_state=SEED).split(idx))
    if regime == "district":
        return list(GroupKFold(5).split(idx, groups=df["dist_code"]))
    if regime == "state":
        return list(LeaveOneGroupOut().split(idx, groups=df["state"]))
    if regime == "temporal":
        tr, te = idx[df["year"].values <= cutoff], idx[df["year"].values > cutoff]
        return [(tr, te)]
    raise ValueError(regime)


def smoke_features(df, rng):
    """FAKE features for pipeline testing only."""
    shares = df[[c for c in df.columns if c.startswith("share_")]].values
    d_ids = df["dist_code"].values
    uniq = np.unique(d_ids)
    W = rng.normal(size=(shares.shape[1], 8))
    dist_mean = {d: shares[d_ids == d].mean(0) @ W + rng.normal(0, 0.5, 8) for d in uniq}
    soil = np.array([dist_mean[d] for d in d_ids])
    for j in range(8):
        df[f"soil_fake{j}"] = soil[:, j]
    for j in range(4):
        df[f"clim_fake{j}"] = shares @ rng.normal(size=shares.shape[1]) + rng.normal(0, 1.0, len(df))
    return df


def run_top1(df, X, feats, a, out):
    le = LabelEncoder().fit(df["top1_crop"]); y = le.transform(df["top1_crop"]); K = len(le.classes_)
    protos = {"Prior (class frequency)": DummyClassifier(strategy="prior")}
    protos.update(get_models(include=TOP1_MODELS, n_classes=K))
    # class-weighted variants: real labels are heavily imbalanced
    from sklearn.ensemble import RandomForestClassifier
    from lightgbm import LGBMClassifier
    protos["Random Forest (balanced)"] = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                                n_jobs=-1, random_state=SEED)
    protos["LightGBM (balanced)"] = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                                   class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=2)
    if a.tabpfn:
        protos.update(get_models(include=["TabPFN v2"], n_classes=K, with_tabpfn=True))
    rows, per_state, conf = [], [], []
    for regime in REGIMES:
        splits = make_splits(df, regime, a.cutoff)
        for name, proto in protos.items():
            for f, (tr, te) in enumerate(splits):
                if len(te) == 0 or len(np.unique(y[tr])) < 2:
                    continue
                m = Encoded(proto).fit(X[tr], y[tr])
                P = full_proba(m, X[te], K); pred = P.argmax(1)
                r = {"regime": regime, "model": name, "fold": f, "n_test": len(te),
                     "accuracy": accuracy_score(y[te], pred),
                     "macro_f1": f1_score(y[te], pred, average="macro", labels=np.unique(y[te]), zero_division=0),
                     "top3_accuracy": top_k_accuracy_score(y[te], P, k=3, labels=np.arange(K)),
                     "unseen_class_share": np.mean(~np.isin(y[te], np.unique(y[tr])))}
                rows.append(r)
                if regime == "state":
                    per_state.append({"model": name, "state": df["state"].values[te][0], **r})
            print(f"[top1] {regime:9s} {name} done", flush=True)
        # conformal under shift with LightGBM: calibration set taken from the training side
        cov, size = [], []
        for f, (tr, te) in enumerate(splits):
            rng = np.random.default_rng(f)
            tr = rng.permutation(tr); ncal = max(int(0.2 * len(tr)), 50)
            cal, fit = tr[:ncal], tr[ncal:]
            m = Encoded(protos["LightGBM"]).fit(X[fit], y[fit])
            Pc, Pt = full_proba(m, X[cal], K), full_proba(m, X[te], K)
            q = conformal_quantile(lac_scores(Pc, y[cal]), a.alpha)
            S = Pt >= 1 - q
            cov.extend(S[np.arange(len(te)), y[te]]); size.extend(S.sum(1))
        conf.append({"regime": regime, "alpha": a.alpha, "target_coverage": 1 - a.alpha,
                     "empirical_coverage": np.mean(cov), "mean_set_size": np.mean(size)})
    res = pd.DataFrame(rows); ps = pd.DataFrame(per_state); conf = pd.DataFrame(conf)
    res.to_csv(out / "raw_top1_per_fold.csv", index=False)
    w = lambda g: np.average(g, weights=res.loc[g.index, "n_test"])
    summ = res.groupby(["regime", "model"]).agg(
        accuracy=("accuracy", w), macro_f1=("macro_f1", w), top3_accuracy=("top3_accuracy", w),
        unseen_class_share=("unseen_class_share", w), folds=("fold", "count")).reset_index()
    save_table(summ, out / "tables", "tabD1_top1_by_regime")
    save_table(ps, out / "tables", "tabD2_top1_per_state")
    save_table(conf, out / "tables", "tabD3_conformal_under_shift")
    return summ, ps, conf, le


def run_multilabel(df, X, a, out):
    lab = [c for c in df.columns if c.startswith("grown_")]
    Y = df[lab].values
    keep = Y.sum(0) >= a.min_label_count
    lab = [l for l, k in zip(lab, keep) if k]; Y = Y[:, keep]
    protos = get_models(include=ML_MODELS, n_classes=2)
    rows = []
    for regime in REGIMES:
        splits = make_splits(df, regime, a.cutoff)
        for name, proto in protos.items():
            P_all, Y_all = [], []
            for tr, te in splits:
                m = PerLabel(proto).fit(X[tr], Y[tr])
                P_all.append(m.predict_proba(X[te])); Y_all.append(Y[te])
            P, T = np.vstack(P_all), np.vstack(Y_all)
            B = P >= 0.5
            top3 = np.argsort(-P, 1)[:, :3]
            hit = np.take_along_axis(T, top3, 1)
            valid = T.sum(0) > 0
            rows.append({"regime": regime, "model": name,
                         "micro_f1": f1_score(T, B, average="micro", zero_division=0),
                         "macro_f1": f1_score(T[:, valid], B[:, valid], average="macro", zero_division=0),
                         "mAP": average_precision_score(T[:, valid], P[:, valid], average="macro"),
                         "hamming_loss": hamming_loss(T, B),
                         "precision_at_3": hit.mean(),
                         "recall_at_3": (hit.sum(1) / np.maximum(T.sum(1), 1)).mean()})
            print(f"[multilabel] {regime:9s} {name} done", flush=True)
    res = pd.DataFrame(rows)
    save_table(res, out / "tables", "tabD4_multilabel_by_regime")
    return res


def shap_stability(df, X, feats, a, out):
    import shap
    le = LabelEncoder().fit(df["top1_crop"]); y = le.transform(df["top1_crop"])
    proto = get_models(include=["LightGBM"], n_classes=len(le.classes_))["LightGBM"]
    res = {}
    for regime in ["random", "state"]:
        imps, names = [], []
        for f, (tr, te) in enumerate(make_splits(df, regime, a.cutoff)):
            imp = SimpleImputer(strategy="median").fit(X[tr])
            m = clone(proto).fit(imp.transform(X[tr]), LabelEncoder().fit_transform(y[tr]))
            sub = te[:300]
            sv = shap.TreeExplainer(m).shap_values(imp.transform(X[sub]))
            sv = np.stack(sv, -1) if isinstance(sv, list) else sv
            v = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(0)
            imps.append(v / v.sum()); names.append(df["state"].values[te][0] if regime == "state" else f"fold {f+1}")
        V = np.array(imps)
        R = np.array([stats.rankdata(-v) for v in V])
        m_, n_ = R.shape
        W = 12 * ((R.sum(0) - m_ * (n_ + 1) / 2) ** 2).sum() / (m_ ** 2 * (n_ ** 3 - n_))
        res[regime] = (V, names, W)
        pd.DataFrame(V, index=names, columns=feats).to_csv(out / "tables" / f"tabD5_shap_by_{regime}.csv")
    save_table(pd.DataFrame({"regime": list(res), "kendalls_W": [res[r][2] for r in res]}),
               out / "tables", "tabD5_shap_stability_W")
    return res


def _save(fig, fd, name):
    if STAMP:  # unmistakable watermark on smoke-test figures
        fig.text(0.5, 0.5, "SMOKE TEST - FAKE FEATURES", rotation=25, ha="center", va="center",
                 fontsize=26, color="#e34948", alpha=0.35, weight="bold")
    savefig(fig, fd, name)


def figures(summ, ps, conf, ml, shp, df, out):
    set_style()
    fd = out / "figures"
    # D1: regime comparison, top-3 accuracy and macro-F1 --------------------------
    models = [m for m in summ.model.unique()]
    order = summ[summ.regime == "random"].sort_values("top3_accuracy", ascending=False).model.tolist()
    fig, axes = plt.subplots(2, 1, figsize=(COL2, 4.6), sharex=True)
    x = np.arange(len(order)); w = 0.2
    for ax, met, lab in zip(axes, ["top3_accuracy", "macro_f1"], ["Top-3 accuracy", "Macro-F1"]):
        for i, (reg, col) in enumerate(zip(REGIMES, CAT)):
            v = summ[summ.regime == reg].set_index("model").reindex(order)[met]
            ax.bar(x + (i - 1.5) * w, v, width=w * 0.92, color=col, label=REGIME_LABEL[reg])
        ax.set_ylabel(lab); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
    axes[0].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.98))
    axes[1].set_xticks(x, order, rotation=30, ha="right")
    axes[0].set_title(f"Same models, different validation: top-1 crop task", fontsize=9, pad=22)
    fig.tight_layout()
    _save(fig, fd, "figD1_validation_regimes_top1")

    # D2: per-state top-3 accuracy (leave-state-out), best random-CV model --------
    best = order[0] if order[0] != "Prior (class frequency)" else order[1]
    p = ps[ps.model == best].sort_values("top3_accuracy")
    pri = ps[ps.model == "Prior (class frequency)"].set_index("state").reindex(p.state)
    fig, ax = plt.subplots(figsize=(COL1, 3.6))
    yy = np.arange(len(p))
    ax.scatter(pri.top3_accuracy, yy, color=DIM, s=14, label="Prior baseline")
    ax.scatter(p.top3_accuracy, yy, color=HILITE, s=16, label=best, zorder=3)
    ax.set_yticks(yy, p.state); ax.set_xlim(0, 1.0); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Top-3 accuracy on the held-out state")
    ax.set_title(f"Leave-one-state-out", loc="left", pad=20)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=6.5, handletextpad=0.2)
    fig.tight_layout()
    _save(fig, fd, "figD2_per_state_top3")

    # D3: multilabel mAP by regime -----------------------------------------------
    o = ml[ml.regime == "random"].sort_values("mAP", ascending=False).model.tolist()
    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    x = np.arange(len(o))
    for i, (reg, col) in enumerate(zip(REGIMES, CAT)):
        v = ml[ml.regime == reg].set_index("model").reindex(o)["mAP"]
        ax.bar(x + (i - 1.5) * 0.2, v, width=0.18, color=col, label=REGIME_LABEL[reg])
    ax.set_xticks(x, o, rotation=25, ha="right"); ax.set_ylim(0, 1); ax.set_ylabel("Macro mAP")
    ax.legend(ncol=2, fontsize=6, loc="lower left", bbox_to_anchor=(0.0, 1.0)); ax.grid(axis="x", visible=False)
    ax.set_title(f"Multi-label: all major crops", loc="left", pad=30)
    fig.tight_layout()
    _save(fig, fd, "figD3_multilabel_map")

    # D4: conformal coverage under shift ------------------------------------------
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    c = conf.set_index("regime").reindex(REGIMES)
    short = {"random": "Random\nk-fold", "district": "Leave-\ndistrict-out", "state": "Leave-\nstate-out",
             "temporal": "Temporal\nhold-out"}
    ax.bar([short[r] for r in REGIMES], c.empirical_coverage, color=CAT[:4], width=0.6)
    ax.axhline(c.target_coverage.iloc[0], color=INK2, ls="--", lw=1)
    for i, (cv_, s_) in enumerate(zip(c.empirical_coverage, c.mean_set_size)):
        ax.text(i, 0.04, f"{cv_:.2f}\n|set| {s_:.1f}", ha="center", fontsize=6.3, color="#0b0b0b")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Coverage (dashed: target 1−α)"); ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_title(f"Conformal sets (LightGBM, α = {c.alpha.iloc[0]})", loc="left")
    fig.tight_layout()
    _save(fig, fd, "figD4_conformal_under_shift")

    # D5: SHAP importance per held-out state --------------------------------------
    if shp is not None:
        V, names, W = shp["state"]
        feats = shp["feats"]
        top = np.argsort(-V.mean(0))[:15]
        fig, ax = plt.subplots(figsize=(COL2 * 0.7, 3.4))
        im = ax.imshow(V[:, top].T, cmap=SEQ, aspect="auto")
        ax.set_yticks(range(len(top)), np.array(feats)[top]); ax.set_xticks(range(len(names)), names, rotation=60, ha="right")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.03, label="Share of mean |SHAP|")
        ax.set_title(f"Feature importance per held-out state (W = {W:.2f}; random folds W = "
                     f"{shp['random'][2]:.2f})", loc="left", fontsize=8)
        fig.tight_layout()
        _save(fig, fd, "figD5_shap_by_state")

    # D6: label distribution ----------------------------------------------------
    g = df[[c for c in df.columns if c.startswith("grown_")]].mean().sort_values()
    fig, ax = plt.subplots(figsize=(COL1, 3.6))
    ax.barh([s.replace("grown_", "").replace("_", " ") for s in g.index], g.values, color=HILITE, height=0.65)
    ax.set_xlabel("Share of district-years where crop is a major crop"); ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=6.5)
    ax.set_title(f"Label prevalence (real data, imbalanced)", loc="left")
    fig.tight_layout()
    _save(fig, fd, "figD6_label_prevalence")


def main(a):
    global STAMP
    out = Path(a.out); (out / "tables").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.data)
    df = df[(df.year >= a.year_min) & (df.year <= a.year_max)].reset_index(drop=True)
    if a.smoke_test:
        STAMP = "SMOKE TEST, FAKE FEATURES: "
        df = smoke_features(df, np.random.default_rng(SEED))
    prefixes = tuple(a.feature_prefixes.split(","))
    feats = [c for c in df.columns if c.startswith(prefixes)]
    if not feats:
        raise SystemExit(f"No feature columns starting with {prefixes}. Attach soil/climate data with "
                         f"build_dataset.py first, or run with --smoke-test to check the pipeline.")
    df = df[df[feats].notna().any(axis=1)].reset_index(drop=True)
    X = df[feats].astype(float).values
    print(f"{len(df)} district-years, {df.dist_code.nunique()} districts, {df.state.nunique()} states, "
          f"years {df.year.min()}-{df.year.max()}, {len(feats)} features", flush=True)
    summ, ps, conf, le = run_top1(df, X, feats, a, out)
    ml = run_multilabel(df, X, a, out) if not a.skip_multilabel else None
    shp = None
    if not a.skip_shap:
        shp = shap_stability(df, X, feats, a, out); shp["feats"] = feats
    if ml is None:
        ml = pd.DataFrame(columns=["regime", "model", "mAP"])
    figures(summ, ps, conf, ml, shp, df, out)
    print(summ.pivot(index="model", columns="regime", values="top3_accuracy").round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/crop_wide_district_year.csv")
    ap.add_argument("--out", default="results/district")
    ap.add_argument("--feature-prefixes", default="soil_,clim_")
    ap.add_argument("--year-min", type=int, default=2010)
    ap.add_argument("--year-max", type=int, default=2017)
    ap.add_argument("--cutoff", type=int, default=2014, help="temporal split: train <= cutoff < test")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--min-label-count", type=int, default=30)
    ap.add_argument("--skip-multilabel", action="store_true")
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--tabpfn", action="store_true", help="add TabPFN (Colab: needs tabpfn + internet)")
    main(ap.parse_args())
