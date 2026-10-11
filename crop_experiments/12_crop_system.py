"""Experiment 12 - Crop recommendation system for the conference paper.

One script for the three parts of the system:
  1. Crop recommendation: data analysis, 80/20 train/test split, 10 algorithms
     (5-fold CV on the training set + one test on the held-out 20%), confusion
     matrix of the system model (Random Forest, tied best), top-3
     recommendations for sample farmers.
  2. Explainable AI: SHAP (TreeExplainer) and LIME on the Random Forest -
     global importance, importance per crop, and one farmer's explanation.
  3. Fertilizer recommendation:
     (a) ML: 8 algorithms on the Fertilizer Prediction dataset (99 rows),
         repeated stratified 5-fold CV, permutation importance;
     (b) nutrient-gap advice: compare the farmer's soil N, P, K with the
         recommended crop's requirement (Harvestify fertilizer.csv table).

Outputs -> results/system/{figures,tables}, results/system/summary.txt
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import (train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
                                     cross_val_score, cross_val_predict)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix)

from common import (load_kaggle, KAGGLE_FEATURES, _xgb, _lgbm, set_style, savefig, save_table,
                    CAT, HILITE, DIM, INK2, MUTED, SEQ, DIV, COL1, COL2, SEED)

NICE = {"N": "N", "P": "P", "K": "K", "temperature": "Temperature", "humidity": "Humidity",
        "ph": "pH", "rainfall": "Rainfall"}


def crop_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "Naive Bayes": GaussianNB(),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "XGBoost": _xgb(SEED),
        "LightGBM": _lgbm(SEED),
        "AdaBoost (default)": AdaBoostClassifier(random_state=SEED),
        "AdaBoost (depth-3 trees)": AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=200,
                                                       learning_rate=0.5, random_state=SEED),
    }


def fert_models(pre):
    def P(m, scale=False):
        return make_pipeline(pre(scale), m)
    return {
        "Logistic Regression": P(LogisticRegression(max_iter=5000), True),
        "Naive Bayes": P(GaussianNB()),
        "KNN (k=5)": P(KNeighborsClassifier(n_neighbors=5), True),
        "SVM (RBF)": P(SVC(random_state=SEED), True),
        "Decision Tree": P(DecisionTreeClassifier(random_state=SEED)),
        "Random Forest": P(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
        "XGBoost": P(_xgb(SEED)),
        "AdaBoost (depth-3 trees)": P(AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=200,
                                                         learning_rate=0.5, random_state=SEED)),
    }


# --------------------------------------------------------------------------- 1. crop
def part_crop(a, fig_dir, tab_dir, log):
    df, X, y, le = load_kaggle(a.crop)
    crops = list(le.classes_)
    log.append(f"crop dataset: {len(df)} rows, {len(crops)} crops, per crop {df['label'].value_counts().min()}"
               f"-{df['label'].value_counts().max()}, missing {int(df.isna().sum().sum())}, "
               f"duplicates {int(df.duplicated().sum())}")
    desc = df[KAGGLE_FEATURES].describe().T[["mean", "std", "min", "max"]]
    desc.index = [NICE[c] for c in desc.index]
    save_table(desc.reset_index().rename(columns={"index": "feature"}), tab_dir, "tabS1_feature_summary",
               float_fmt="%.2f")

    # data analysis figure: correlation + mean N, P, K per crop
    corr = df[KAGGLE_FEATURES].corr()
    save_table(corr.reset_index(), tab_dir, "tabS1_correlation", float_fmt="%.3f")
    fig = plt.figure(figsize=(COL2, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.3], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(corr.values, cmap=DIV, vmin=-1, vmax=1)
    labs = [NICE[c] for c in KAGGLE_FEATURES]
    ax.set_xticks(range(7)); ax.set_xticklabels(labs, rotation=60, ha="right")
    ax.set_yticks(range(7)); ax.set_yticklabels(labs)
    for i in range(7):
        for j in range(7):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(v) > 0.6 else INK2)
    ax.grid(False); ax.set_title("(a) Feature correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = fig.add_subplot(gs[1])
    m = df.groupby("label")[["N", "P", "K"]].mean().loc[crops]
    xs = np.arange(len(crops)); w = 0.27
    for i, c in enumerate(["N", "P", "K"]):
        ax.bar(xs + (i - 1) * w, m[c].values, w, color=CAT[i], label=c)
    ax.set_xticks(xs); ax.set_xticklabels(crops, rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("Mean value in dataset"); ax.legend(ncol=3, loc="upper right")
    ax.set_title("(b) Mean N, P and K per crop"); ax.grid(axis="x", visible=False)
    savefig(fig, fig_dir, "figS1_data_analysis")

    # 80/20 split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    log.append(f"train {len(ytr)} rows, test {len(yte)} rows (stratified, seed {SEED})")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, fitted, preds = [], {}, {}
    for name, model in crop_models().items():
        cvs = cross_val_score(clone(model), Xtr, ytr, cv=cv, scoring="accuracy", n_jobs=1)
        m = clone(model)
        t0 = time.perf_counter(); m.fit(Xtr, ytr); fit_t = time.perf_counter() - t0
        t0 = time.perf_counter(); p = m.predict(Xte); pred_t = (time.perf_counter() - t0) / len(yte) * 1000
        fitted[name], preds[name] = m, p
        rows.append(dict(model=name, cv_accuracy_mean=cvs.mean(), cv_accuracy_std=cvs.std(),
                         test_accuracy=accuracy_score(yte, p),
                         test_precision=precision_score(yte, p, average="macro", zero_division=0),
                         test_recall=recall_score(yte, p, average="macro", zero_division=0),
                         test_f1=f1_score(yte, p, average="macro", zero_division=0),
                         test_errors=int((p != yte).sum()), fit_time_s=fit_t, predict_ms_per_sample=pred_t))
        print(f"  {name:26s} cv {cvs.mean():.4f} test {rows[-1]['test_accuracy']:.4f}")
    res = pd.DataFrame(rows).sort_values(["test_accuracy", "cv_accuracy_mean"], ascending=False)
    save_table(res, tab_dir, "tabS2_crop_model_comparison")
    best = a.system_model
    log.append("crop models (test acc, cv mean+-sd, errors of %d):" % len(yte))
    for _, r in res.iterrows():
        log.append(f"  {r.model:26s} test {r.test_accuracy:.4f}  F1 {r.test_f1:.4f}  cv {r.cv_accuracy_mean:.4f}"
                   f"+-{r.cv_accuracy_std:.4f}  errors {r.test_errors}  fit {r.fit_time_s:.2f}s")
    top = res[res.test_accuracy == res.test_accuracy.max()].model.tolist()
    log.append(f"highest test accuracy: {top}; model used in the system: {best}")

    # model comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.7), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.08))
    r = res.iloc[::-1]
    ys = np.arange(len(r))
    ax = axes[0]
    cols = [HILITE if v == res.test_accuracy.max() else DIM for v in r.test_accuracy]
    ax.barh(ys, r.test_accuracy * 100, color=cols, height=0.6, label="Test accuracy (20% hold-out)")
    ax.errorbar(r.cv_accuracy_mean * 100, ys, xerr=r.cv_accuracy_std * 100, fmt="o", color=CAT[1], ms=3,
                capsize=2, label="5-fold CV on training set (mean $\\pm$ sd)")
    for yy, v, c, sd in zip(ys, r.test_accuracy * 100, r.cv_accuracy_mean * 100, r.cv_accuracy_std * 100):
        ax.text(max(v, c + sd) + 1.5 if v < 90 else 101, yy, f"{v:.2f}", va="center", fontsize=6, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels(r.model)
    ax.set_xlim(0, 108); ax.set_xlabel("Accuracy (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=6)
    ax.set_title("(a) Accuracy", pad=26)
    ax = axes[1]
    ax.barh(ys, r.fit_time_s, color=cols, height=0.6)
    ax.set_xscale("log"); ax.set_yticks(ys); ax.set_yticklabels([])
    ax.set_xlabel("Training time (s, log scale)"); ax.set_title("(b) Training time", pad=26)
    savefig(fig, fig_dir, "figS2_model_comparison")

    # confusion matrix of the system model + per-crop report
    cm = confusion_matrix(yte, preds[best], labels=range(len(crops)))
    fig, ax = plt.subplots(figsize=(COL1, 3.3))
    ax.imshow(cm, cmap=SEQ)
    for i in range(len(crops)):
        for j in range(len(crops)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=4.5,
                        color="white" if cm[i, j] > 10 else INK2)
    ax.set_xticks(range(len(crops))); ax.set_xticklabels(crops, rotation=90, fontsize=5)
    ax.set_yticks(range(len(crops))); ax.set_yticklabels(crops, fontsize=5)
    ax.set_xlabel("Predicted crop"); ax.set_ylabel("Actual crop"); ax.grid(False)
    ax.set_title(f"{best}: {int((preds[best] == yte).sum())} of {len(yte)} test samples correct")
    savefig(fig, fig_dir, "figS3_confusion_best")
    per = pd.DataFrame(dict(crop=crops,
                            precision=precision_score(yte, preds[best], average=None, labels=range(len(crops)), zero_division=0),
                            recall=recall_score(yte, preds[best], average=None, labels=range(len(crops)), zero_division=0),
                            f1=f1_score(yte, preds[best], average=None, labels=range(len(crops)), zero_division=0)))
    save_table(per, tab_dir, "tabS3_per_crop_best")
    off = [(crops[i], crops[j], int(cm[i, j])) for i in range(len(crops)) for j in range(len(crops)) if i != j and cm[i, j]]
    log.append(f"{best} test errors (actual -> predicted, count): {off}")
    log.append(f"crops with F1 < 1 for {best}: " + ", ".join(f"{c} {f:.3f}" for c, f in zip(per.crop, per.f1) if f < 1))

    # testing with farmer inputs: top-3 recommendations of the system model
    rng = np.random.default_rng(SEED)
    pick_crops = rng.choice(len(crops), 6, replace=False)
    idx = [int(rng.choice(np.where(yte == c)[0])) for c in pick_crops]
    P = fitted[best].predict_proba(Xte[idx])
    ex = []
    for k, i in enumerate(idx):
        order = np.argsort(-P[k])[:3]
        ex.append(dict(farmer=f"F{k+1}", **{NICE[f]: Xte[i, j] for j, f in enumerate(KAGGLE_FEATURES)},
                       actual=crops[yte[i]],
                       top1=f"{crops[order[0]]} ({P[k, order[0]]:.2f})",
                       top2=f"{crops[order[1]]} ({P[k, order[1]]:.2f})",
                       top3=f"{crops[order[2]]} ({P[k, order[2]]:.2f})"))
    ex = pd.DataFrame(ex)
    save_table(ex, tab_dir, "tabS4_farmer_examples", float_fmt="%.1f")
    return dict(df=df, X=X, y=y, crops=crops, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, fitted=fitted,
                best=best, examples=ex, example_idx=idx)


# --------------------------------------------------------------------------- 2. XAI
def part_xai(a, C, fig_dir, tab_dir, log):
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    rf = C["fitted"]["Random Forest"]
    Xtr, Xte, yte, crops = C["Xtr"], C["Xte"], C["yte"], C["crops"]
    names = [NICE[f] for f in KAGGLE_FEATURES]
    sv = shap.TreeExplainer(rf).shap_values(Xte)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)   # (n, features, classes)
    glob = np.abs(sv).mean(axis=(0, 2))
    g = pd.DataFrame(dict(feature=names, mean_abs_shap=glob)).sort_values("mean_abs_shap", ascending=False)
    g["share"] = g.mean_abs_shap / g.mean_abs_shap.sum()
    save_table(g, tab_dir, "tabS5_shap_global")
    log.append("global SHAP share (RF, test set): " + ", ".join(f"{f} {s:.3f}" for f, s in zip(g.feature, g.share)))
    # per crop: mean |SHAP| of each crop's own output on its own test rows
    per = np.zeros((len(crops), len(names)))
    for c in range(len(crops)):
        per[c] = np.abs(sv[yte == c, :, c]).mean(0)
    per_df = pd.DataFrame(per, index=crops, columns=names)
    save_table(per_df.reset_index().rename(columns={"index": "crop"}), tab_dir, "tabS5_shap_per_crop")
    top_feat = per_df.idxmax(axis=1)
    log.append("top SHAP feature per crop: " + ", ".join(f"{c}:{f}" for c, f in top_feat.items()))
    log.append("count of crops by top feature: " + str(top_feat.value_counts().to_dict()))

    fig = plt.figure(figsize=(COL2, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    gg = g.iloc[::-1]
    ax.barh(gg.feature, gg.mean_abs_shap, color=HILITE, height=0.6)
    for yy, (v, s) in enumerate(zip(gg.mean_abs_shap, gg.share)):
        ax.text(v, yy, f" {s*100:.0f}%", va="center", fontsize=6, color=INK2)
    ax.set_xlabel("Mean |SHAP value|"); ax.set_title("(a) Global feature importance")
    ax = fig.add_subplot(gs[1])
    norm = per / per.sum(1, keepdims=True)
    ax.imshow(norm, cmap=SEQ, aspect="auto")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(crops))); ax.set_yticklabels(crops, fontsize=5.5); ax.grid(False)
    for c in range(len(crops)):
        j = int(np.argmax(norm[c])); ax.text(j, c, "•", ha="center", va="center", color="white", fontsize=7)
    ax.set_title("(b) Importance for each crop (row-normalised)")
    savefig(fig, fig_dir, "figS4_shap_global_per_crop")

    # one farmer: SHAP and LIME for the recommended crop
    i = C["example_idx"][0]
    x = Xte[i]
    pred = int(rf.predict(x[None])[0])
    prob = rf.predict_proba(x[None])[0, pred]
    s = sv[i, :, pred]
    lime_exp = LimeTabularExplainer(Xtr, feature_names=names, class_names=crops, discretize_continuous=True,
                                    random_state=SEED, mode="classification")
    e = lime_exp.explain_instance(x, rf.predict_proba, labels=[pred], num_features=7, num_samples=a.lime_samples)
    lw = dict(e.as_list(label=pred))
    lime_map = e.as_map()[pred]
    lvals = np.zeros(len(names))
    for j, w in lime_map:
        lvals[j] = w
    ldesc = {j: d for (d, _), (j, _) in zip(e.as_list(label=pred), lime_map)}
    farmer = pd.DataFrame(dict(feature=names, value=x, shap=s, lime=lvals,
                               lime_rule=[ldesc.get(j, "") for j in range(len(names))]))
    save_table(farmer, tab_dir, "tabS6_farmer_explanation")
    log.append(f"farmer F1 explanation: predicted {crops[pred]} (p={prob:.2f}), actual {crops[yte[i]]}; "
               f"SHAP {dict(zip(names, np.round(s, 3)))}; LIME {dict(zip(names, np.round(lvals, 3)))}")

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.4), gridspec_kw=dict(wspace=0.55))
    for ax, vals, ttl in [(axes[0], s, "(a) SHAP"), (axes[1], lvals, "(b) LIME")]:
        o = np.argsort(np.abs(vals))
        lab = [f"{names[j]} = {x[j]:.1f}" for j in o]
        ax.barh(range(len(o)), vals[o], color=[CAT[0] if v >= 0 else CAT[1] for v in vals[o]], height=0.6)
        ax.set_yticks(range(len(o))); ax.set_yticklabels(lab)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(f"Contribution to P({crops[pred]})")
        ax.set_title(f"{ttl}: why {crops[pred]} (p = {prob:.2f})")
    savefig(fig, fig_dir, "figS5_farmer_explanation")

    # SHAP vs LIME agreement on test rows
    n = min(a.n_agree, len(Xte))
    rows = np.random.default_rng(SEED).choice(len(Xte), n, replace=False)
    top1, top3 = [], []
    P = rf.predict(Xte[rows])
    for r_, p_ in zip(rows, P):
        e = lime_exp.explain_instance(Xte[r_], rf.predict_proba, labels=[p_], num_features=7,
                                      num_samples=a.lime_samples)
        lv = np.zeros(len(names))
        for j, w in e.as_map()[p_]:
            lv[j] = w
        sa, la = np.argsort(-np.abs(sv[r_, :, p_])), np.argsort(-np.abs(lv))
        top1.append(sa[0] == la[0]); top3.append(len(set(sa[:3]) & set(la[:3])) / 3)
    agree = pd.DataFrame([dict(instances=n, same_top_feature=np.mean(top1), top3_overlap=np.mean(top3))])
    save_table(agree, tab_dir, "tabS6_shap_lime_agreement")
    log.append(f"SHAP vs LIME on {n} test rows: same top feature {np.mean(top1):.3f}, top-3 overlap {np.mean(top3):.3f}")


# --------------------------------------------------------------------------- 3. fertilizer
def part_fert(a, C, fig_dir, tab_dir, log):
    df = pd.read_csv(a.fert)
    df.columns = [c.strip() for c in df.columns]
    num = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Phosphorous", "Potassium"]
    cat = ["Soil Type", "Crop Type"]
    X, y = df[num + cat], LabelEncoder().fit_transform(df["Fertilizer Name"])
    classes = sorted(df["Fertilizer Name"].unique())
    log.append(f"fertilizer dataset: {len(df)} rows, {len(classes)} classes {df['Fertilizer Name'].value_counts().to_dict()}")

    def pre(scale):
        return ColumnTransformer([("num", StandardScaler() if scale else "passthrough", num),
                                  ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    rows, models = [], fert_models(pre)
    for name, m in models.items():
        acc, f1 = [], []
        for tr, te in cv.split(X, y):
            mm = clone(m).fit(X.iloc[tr], y[tr]); p = mm.predict(X.iloc[te])
            acc.append(accuracy_score(y[te], p)); f1.append(f1_score(y[te], p, average="macro", zero_division=0))
        rows.append(dict(model=name, accuracy_mean=np.mean(acc), accuracy_std=np.std(acc),
                         macro_f1_mean=np.mean(f1), macro_f1_std=np.std(f1)))
        print(f"  fert {name:26s} {np.mean(acc):.4f}")
    res = pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False)
    save_table(res, tab_dir, "tabS7_fertilizer_model_comparison")
    best = res.iloc[0]["model"]
    log.append("fertilizer models (5x10 CV accuracy, macro-F1):")
    for _, r in res.iterrows():
        log.append(f"  {r.model:26s} {r.accuracy_mean:.4f}+-{r.accuracy_std:.4f}  F1 {r.macro_f1_mean:.4f}")

    # importance (held-out permutation, RF) and confusion of best (one 5-fold run)
    imp = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        mm = clone(models["Random Forest"]).fit(X.iloc[tr], y[tr])
        pi = permutation_importance(mm, X.iloc[te], y[te], n_repeats=20, random_state=SEED)
        imp.append(pi.importances_mean)
    imp = pd.DataFrame(dict(feature=num + cat, importance=np.mean(imp, 0))).sort_values("importance", ascending=False)
    imp["feature"] = imp.feature.replace({"Temparature": "Temperature", "Phosphorous": "Phosphorus"})
    save_table(imp, tab_dir, "tabS8_fertilizer_importance")
    log.append("fertilizer RF permutation importance: " + ", ".join(f"{f} {v:.3f}" for f, v in zip(imp.feature, imp.importance)))
    p = cross_val_predict(clone(models[best]), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    cm = confusion_matrix(y, p)
    log.append(f"fertilizer best {best}: one 5-fold run accuracy {accuracy_score(y, p):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5), gridspec_kw=dict(width_ratios=[1.2, 1, 1], wspace=0.6))
    r = res.iloc[::-1]
    ax = axes[0]
    ax.barh(r.model, r.accuracy_mean * 100, xerr=r.accuracy_std * 100, color=[HILITE if n == best else DIM for n in r.model],
            height=0.6, error_kw=dict(lw=0.8, capsize=2))
    ax.set_xlim(0, 100); ax.set_xlabel("Accuracy (%)"); ax.set_title("(a) Algorithms (5$\\times$10 CV)")
    ax = axes[1]
    ii = imp.iloc[::-1]
    ax.barh(ii.feature, ii.importance, color=CAT[2], height=0.6)
    ax.set_xlabel("Drop in accuracy when shuffled"); ax.set_title("(b) Feature importance (RF)")
    ax = axes[2]
    ax.imshow(cm, cmap=SEQ)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6, color="white" if cm[i, j] > 10 else INK2)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=6); ax.grid(False)
    ax.set_title(f"(c) {best}, confusion"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    savefig(fig, fig_dir, "figS6_fertilizer")

    # nutrient-gap advice for the sample farmers: rank-1 and rank-2 crop
    req = pd.read_csv(a.req, index_col=0).set_index("Crop")
    advice = {("N", "low"): "add nitrogen (e.g. urea)", ("N", "high"): "skip nitrogen this season",
              ("P", "low"): "add phosphate (e.g. DAP or SSP)", ("P", "high"): "skip phosphate this season",
              ("K", "low"): "add potash (e.g. MOP)", ("K", "high"): "skip potash this season"}
    out = []
    model = C["fitted"][C["best"]]
    for k, i in enumerate(C["example_idx"]):
        x = C["Xte"][i]
        order = np.argsort(-model.predict_proba(x[None])[0])[:2]
        for rank, ci in enumerate(order, 1):
            crop = C["crops"][int(ci)]
            need = req.loc[crop, ["N", "P", "K"]].astype(float).values
            gap = need - x[:3]
            j = int(np.argmax(np.abs(gap)))
            nut, status = "NPK"[j], "low" if gap[j] > 0 else "high"
            out.append(dict(farmer=f"F{k+1}", soil_NPK=f"{x[0]:.0f}-{x[1]:.0f}-{x[2]:.0f}", rank=rank, crop=crop,
                            required_NPK="-".join(f"{v:.0f}" for v in need),
                            gap_N=gap[0], gap_P=gap[1], gap_K=gap[2],
                            largest_gap=f"{nut} {status} by {abs(gap[j]):.0f}", advice=advice[(nut, status)]))
    out = pd.DataFrame(out)
    save_table(out, tab_dir, "tabS9_nutrient_gap_examples", float_fmt="%.0f")
    log.append("nutrient-gap examples:\n" + out.to_string(index=False))


def main(a):
    set_style()
    out = Path(a.out); fig_dir, tab_dir = out / "figures", out / "tables"
    log = []
    print("[1] crop recommendation"); C = part_crop(a, fig_dir, tab_dir, log)
    print("[2] explainable AI"); part_xai(a, C, fig_dir, tab_dir, log)
    print("[3] fertilizer"); part_fert(a, C, fig_dir, tab_dir, log)
    (out / "summary.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", default="data/Crop_recommendation.csv")
    ap.add_argument("--fert", default="data/Fertilizer_Prediction_99rows.csv")
    ap.add_argument("--req", default="data/provenance/harvestify_aac427c/FertilizerData.csv")
    ap.add_argument("--out", default="results/system")
    ap.add_argument("--lime-samples", type=int, default=2000)
    ap.add_argument("--n-agree", type=int, default=100)
    ap.add_argument("--system-model", default="Random Forest",
                    help="model used for the confusion matrix, farmer examples and advice")
    main(ap.parse_args())
