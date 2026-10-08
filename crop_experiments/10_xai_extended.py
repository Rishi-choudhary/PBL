"""Experiment 10 - Do explanations tell the truth, and can they guide action?

Both public benchmarks have a known (or strongly evidenced) data-generating
rule, which turns them into test beds for explanation methods:
  * Kaggle crop data: N, P, K = per-crop base value + bounded random offset
    (creator's notebook, data/provenance/harvestify_aac427c)
  * 99-row fertilizer data: the label follows N, P, K thresholds; temperature,
    humidity, moisture, soil type and crop type carry no signal (experiment 11)

Parts
  1. Sanity: a pure-noise feature is added; how much importance does each
     method (SHAP, LIME, permutation) give it? On the fertilizer data, how much
     goes to the known-uninformative features?
  2. Disagreement (Krishna et al., TMLR 2024): SHAP vs LIME top-3 feature
     agreement, rank correlation and sign agreement, per instance.
  3. Faithfulness by deletion: replace the top-k features (by SHAP, by LIME, at
     random) with training medians and measure the drop in the probability of
     the predicted crop.
  4. Actionable counterfactuals: only N, P, K and pH may change (fertiliser,
     lime/gypsum); temperature, humidity and rainfall are fixed. For each crop
     pair, can a soil change alone switch the recommendation, and does the
     counterfactual land inside the target crop's generator range?

Outputs -> results/kaggle/{tables,figures}
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

from common import (load_kaggle, get_models, KAGGLE_FEATURES, set_style, savefig, save_table,
                    CAT, HILITE, DIM, INK2, MUTED, SEQ, COL1, COL2, SEED)

MUTABLE = ["N", "P", "K", "ph"]
GRID = {"N": np.arange(0, 145, 10), "P": np.arange(5, 150, 10), "K": np.arange(5, 210, 10),
        "ph": np.round(np.arange(3.5, 10.0, 0.8), 2)}
RULE = {"N": (-20, 20), "P": (-5, 20), "K": (-5, 5)}


def shap_pred_class(model, X, pred):
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    sv = np.stack(sv, -1) if isinstance(sv, list) else sv
    if sv.ndim == 2:
        return sv
    return sv[np.arange(len(X)), :, pred]


def lime_pred_class(model, Xtr, X, pred, names, n_samples, seed):
    from lime.lime_tabular import LimeTabularExplainer
    exp = LimeTabularExplainer(Xtr, feature_names=names, discretize_continuous=True, random_state=seed,
                               mode="classification")
    out = np.zeros_like(X, dtype=float)
    for i in range(len(X)):
        e = exp.explain_instance(X[i], model.predict_proba, labels=[pred[i]], num_features=X.shape[1],
                                 num_samples=n_samples)
        for j, w in e.as_map()[pred[i]]:
            out[i, j] = w
    return out


def share(v):
    v = np.abs(np.asarray(v, float))
    return v / v.sum() if v.sum() > 0 else v


def load_fert(path):
    f = pd.read_csv(path); f.columns = [c.strip() for c in f.columns]
    y_names = f.pop("Fertilizer Name")
    cls = sorted(y_names.unique()); y = np.array([cls.index(v) for v in y_names])
    X = pd.get_dummies(f, columns=["Soil Type", "Crop Type"]).astype(float)
    return X, y, cls


def main(a):
    set_style()
    rng = np.random.default_rng(SEED)
    out_t, out_f = f"{a.out}/tables", f"{a.out}/figures"
    df, X, y, le = load_kaggle(a.data)
    K = len(le.classes_)
    names = KAGGLE_FEATURES + ["noise"]
    Xn = np.hstack([X, rng.normal(size=(len(X), 1))])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    proto = get_models(include=["LightGBM"], n_classes=K)["LightGBM"]

    # ---- parts 1-3 on Kaggle: one held-out fold, explained by SHAP and LIME ----------
    tr, te = next(cv.split(Xn, y))
    te = te[:a.n_explain]
    m = clone(proto).fit(Xn[tr], y[tr])
    pred = m.predict(Xn[te])
    S = shap_pred_class(m, Xn[te], pred)
    L = lime_pred_class(m, Xn[tr], Xn[te], pred, names, a.lime_samples, SEED)
    P = permutation_importance(m, Xn[te], y[te], scoring="f1_macro", n_repeats=10, random_state=SEED)
    sanity = pd.DataFrame({
        "method": ["SHAP", "LIME", "Permutation"],
        "noise_feature_share": [share(np.abs(S).mean(0))[-1], share(np.abs(L).mean(0))[-1],
                                share(np.clip(P.importances_mean, 0, None))[-1]],
        "noise_rank_of_8": [stats.rankdata(-np.abs(S).mean(0))[-1], stats.rankdata(-np.abs(L).mean(0))[-1],
                            stats.rankdata(-P.importances_mean)[-1]]})
    sanity["dataset"] = "Kaggle crop"

    # disagreement per instance
    k = 3
    top_s, top_l = np.argsort(-np.abs(S), 1)[:, :k], np.argsort(-np.abs(L), 1)[:, :k]
    feat_agree = np.array([len(set(a_) & set(b_)) / k for a_, b_ in zip(top_s, top_l)])
    rank_corr = np.array([stats.spearmanr(np.abs(s), np.abs(l))[0] for s, l in zip(S, L)])
    sign_agree = np.array([np.mean([np.sign(S[i, j]) == np.sign(L[i, j]) for j in set(top_s[i]) & set(top_l[i])])
                           if set(top_s[i]) & set(top_l[i]) else np.nan for i in range(len(S))])
    top1_same = np.mean(top_s[:, 0] == top_l[:, 0])
    dis = pd.DataFrame({"metric": ["top-3 feature agreement (mean)", "share with identical top feature",
                                   "Spearman rank corr. of |attribution| (mean)", "sign agreement on shared top-3"],
                        "value": [feat_agree.mean(), top1_same, np.nanmean(rank_corr), np.nanmean(sign_agree)]})
    save_table(dis, out_t, "tab10_shap_lime_disagreement")
    pd.DataFrame({"top3_agreement": feat_agree, "spearman": rank_corr}).to_csv(
        f"{out_t}/tab10_disagreement_per_instance.csv", index=False)
    print(dis.to_string(index=False))

    # deletion faithfulness
    med = np.median(Xn[tr], 0)
    p0 = m.predict_proba(Xn[te])[np.arange(len(te)), pred]
    rows = []
    for meth, A in [("SHAP", S), ("LIME", L), ("Random", rng.normal(size=S.shape))]:
        order = np.argsort(-np.abs(A) if meth != "Random" else A, 1)
        for kk in range(0, 5):
            Xd = Xn[te].copy()
            for i in range(len(te)):
                Xd[i, order[i, :kk]] = med[order[i, :kk]]
            pk = m.predict_proba(Xd)[np.arange(len(te)), pred]
            rows.append({"method": meth, "features_removed": kk, "mean_prob_predicted_crop": pk.mean(),
                         "mean_drop": (p0 - pk).mean()})
    dele = pd.DataFrame(rows)
    save_table(dele, out_t, "tab10_deletion_faithfulness")

    # ---- part 1 on the fertilizer data (known-uninformative features) -----------------
    Fx, Fy, fcls = load_fert(a.fert)
    Fx["noise"] = rng.normal(size=len(Fx))
    fnames = list(Fx.columns); FX = Fx.values
    npk = [i for i, c in enumerate(fnames) if c in ("Nitrogen", "Phosphorous", "Potassium")]
    sh, li, pe = np.zeros(len(fnames)), np.zeros(len(fnames)), np.zeros(len(fnames))
    fcv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    for ftr, fte in fcv.split(FX, Fy):
        fm = clone(rf).fit(FX[ftr], Fy[ftr]); fp = fm.predict(FX[fte])
        sh += np.abs(shap_pred_class(fm, FX[fte], fp)).sum(0)
        li += np.abs(lime_pred_class(fm, FX[ftr], FX[fte], fp, fnames, a.lime_samples, SEED)).sum(0)
        pe += np.clip(permutation_importance(fm, FX[fte], Fy[fte], n_repeats=10, random_state=SEED)
                      .importances_mean, 0, None)
    fert_rows = []
    for meth, v in [("SHAP", sh), ("LIME", li), ("Permutation", pe)]:
        s_ = share(v)
        fert_rows.append({"method": meth, "dataset": "Fertilizer (99 rows)", "NPK_share": s_[npk].sum(),
                          "uninformative_share": 1 - s_[npk].sum() - s_[-1], "noise_feature_share": s_[-1],
                          "noise_rank": stats.rankdata(-v)[-1], "n_features": len(fnames)})
    fert = pd.DataFrame(fert_rows)
    sanity = pd.concat([sanity, fert[["method", "dataset", "noise_feature_share", "noise_rank"]]
                        .rename(columns={"noise_rank": "noise_rank_of_8"})], ignore_index=True)
    sanity = sanity.rename(columns={"noise_rank_of_8": "noise_feature_rank"})
    save_table(sanity, out_t, "tab10_noise_feature_sanity")
    save_table(fert, out_t, "tab10_fertilizer_attribution_shares")
    print(sanity.to_string(index=False)); print(fert.to_string(index=False))

    # ---- part 4: actionable counterfactuals ---------------------------------------
    mi = [KAGGLE_FEATURES.index(f) for f in MUTABLE]
    rng_f = X[:, mi].max(0) - X[:, mi].min(0)
    G = np.array(np.meshgrid(*[GRID[f] for f in MUTABLE], indexing="ij")).reshape(len(MUTABLE), -1).T
    base = pd.read_csv(a.base).drop(columns=["Unnamed: 0"], errors="ignore").set_index("Crop").loc[le.classes_]
    cf_model = clone(proto).fit(X, y)
    feas = np.zeros((K, K)); dist = np.full((K, K), np.nan); inbox = []
    pick = np.concatenate([rng.choice(np.where(y == c)[0], a.cf_per_crop, replace=False) for c in range(K)])
    for n_i, i in enumerate(pick):
        Xg = np.repeat(X[i][None, :], len(G), 0); Xg[:, mi] = G
        gp = cf_model.predict(Xg)
        d = (np.abs(G - X[i, mi]) / rng_f).sum(1)
        for t in range(K):
            if t == y[i]:
                continue
            hit = gp == t
            if hit.any():
                j = np.argmin(np.where(hit, d, np.inf))
                feas[y[i], t] += 1
                dist[y[i], t] = np.nanmean([dist[y[i], t], d[j]]) if not np.isnan(dist[y[i], t]) else d[j]
                bx = all(RULE[f][0] - 10 <= G[j, MUTABLE.index(f)] - base.loc[le.classes_[t], f] <= RULE[f][1] + 10
                         for f in RULE)   # +-10 tolerance for the 10-unit grid
                inbox.append(bx)
        if n_i % 20 == 0:
            print(f"counterfactuals {n_i + 1}/{len(pick)}", flush=True)
    feas /= a.cf_per_crop
    np.fill_diagonal(feas, np.nan)
    cf_tab = pd.DataFrame(feas, index=le.classes_, columns=le.classes_)
    cf_tab.to_csv(f"{out_t}/tab10_counterfactual_feasibility.csv")
    cf_summary = pd.DataFrame({
        "metric": ["crop pairs reachable by soil change alone (share of instances)",
                   "counterfactuals landing in target crop's generator range (+-10)",
                   "instances explained"],
        "value": [np.nanmean(feas), np.mean(inbox) if inbox else np.nan, len(pick)]})
    save_table(cf_summary, out_t, "tab10_counterfactual_summary")
    print(cf_summary.to_string(index=False))

    # ---- figures ------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.4), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    ax = axs[0]
    ax.hist(feat_agree, bins=[-0.17, 0.17, 0.5, 0.83, 1.17], color=HILITE, rwidth=0.8)
    ax.set_xticks([0, 1 / 3, 2 / 3, 1], ["0/3", "1/3", "2/3", "3/3"])
    ax.set_xlabel("Top-3 features shared by SHAP and LIME"); ax.set_ylabel("Instances")
    ax.set_title("(a) SHAP vs LIME agreement", loc="left", pad=22)
    ax = axs[1]
    for c, meth in zip([CAT[0], CAT[1], MUTED], ["SHAP", "LIME", "Random"]):
        t = dele[dele.method == meth]
        ax.plot(t.features_removed, t.mean_prob_predicted_crop, color=c, marker="o", ms=3, label=meth,
                ls="--" if meth == "Random" else "-")
    ax.set_xlabel("Top-k features set to median"); ax.set_ylabel("P(predicted crop)")
    ax.set_ylim(0, 1.02); ax.set_title("(b) Deletion test", loc="left", pad=22); ax.legend(fontsize=6.5)
    ax = axs[2]
    w = 0.26; xs = np.arange(3)
    for i, (c, ds) in enumerate(zip([CAT[0], CAT[1]], ["Kaggle crop", "Fertilizer (99 rows)"])):
        v = sanity[sanity.dataset == ds].set_index("method").reindex(["SHAP", "LIME", "Permutation"])
        ax.bar(xs + (i - 0.5) * w, v.noise_feature_share, width=w * 0.92, color=c, label=ds)
    ax.set_xticks(xs, ["SHAP", "LIME", "Permut."]); ax.set_ylabel("Share given to noise feature")
    ax.set_title("(c) Noise-feature sanity check", loc="left", pad=22)
    ax.legend(fontsize=6, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0)); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    savefig(fig, out_f, "fig15_xai_disagreement_faithfulness")

    fig, ax = plt.subplots(figsize=(COL2 * 0.62, COL2 * 0.55))
    im = ax.imshow(feas, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(K), le.classes_, rotation=90); ax.set_yticks(range(K), le.classes_); ax.grid(False)
    ax.set_xlabel("Target crop"); ax.set_ylabel("Current recommendation")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Share of fields where N/P/K/pH change alone suffices")
    ax.set_title("Which switches can soil management alone achieve?\n(climate fixed; LightGBM; diagonal blank)",
                 fontsize=8)
    fig.tight_layout()
    savefig(fig, out_f, "fig16_counterfactual_feasibility")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Crop_recommendation.csv")
    ap.add_argument("--fert", default="data/Fertilizer_Prediction_99rows.csv")
    ap.add_argument("--base", default="data/provenance/harvestify_aac427c/FertilizerData.csv")
    ap.add_argument("--out", default="results/kaggle")
    ap.add_argument("--n-explain", type=int, default=200)
    ap.add_argument("--lime-samples", type=int, default=2000)
    ap.add_argument("--cf-per-crop", type=int, default=5)
    main(ap.parse_args())
