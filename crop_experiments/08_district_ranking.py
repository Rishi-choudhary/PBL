"""Experiment 8 - Crop recommendation as a ranking problem.

A recommender's output is an ordered list of crops, so this script learns to
rank all crops for each district-year (query) by their cropped-area share.

Methods
  Popularity        rank crops by mean share in the training data (no features)
  Pointwise GBDT    LightGBM regressor on area share, features + crop id
  LambdaRank        LightGBM LGBMRanker optimising NDCG directly
Metrics: NDCG@3, NDCG@5 (relevance = graded area share), precision@3 against
the multi-label "grown" sets. Same four validation regimes as experiment 7.

Outputs -> results/district/{tables,figures}
"""
import argparse, importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import ndcg_score
from lightgbm import LGBMRanker, LGBMRegressor

from common import set_style, savefig, save_table, CAT, COL1, SEED

exp7 = importlib.import_module("07_district_experiments")
GRADES = [0.01, 0.05, 0.15, 0.30]   # share thresholds -> relevance 0..4


def expand(X, shares):
    """(n, d) features + (n, L) shares -> (n*L, d+1) rows with crop id last."""
    n, L = shares.shape
    Xr = np.repeat(X, L, axis=0)
    crop = np.tile(np.arange(L), n)[:, None]
    return np.hstack([Xr, crop]), shares.reshape(-1)


def evaluate(scores, shares, grown, k_list=(3, 5)):
    out = {f"ndcg@{k}": ndcg_score(shares, scores, k=k) for k in k_list}
    top3 = np.argsort(-scores, 1)[:, :3]
    out["precision@3"] = np.take_along_axis(grown, top3, 1).mean()
    return out


def main(a):
    out = Path(a.out); (out / "tables").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.data)
    df = df[(df.year >= a.year_min) & (df.year <= a.year_max)].reset_index(drop=True)
    stamp = ""
    if a.smoke_test:
        stamp = "SMOKE TEST, FAKE FEATURES: "
        df = exp7.smoke_features(df, np.random.default_rng(SEED))
    feats = [c for c in df.columns if c.startswith(tuple(a.feature_prefixes.split(",")))]
    if not feats:
        raise SystemExit("No soil_/clim_ feature columns. Attach data first or use --smoke-test.")
    df = df[df[feats].notna().any(axis=1)].reset_index(drop=True)
    share_cols = sorted(c for c in df.columns if c.startswith("share_"))
    grown_cols = [c.replace("share_", "grown_") for c in share_cols]
    X = df[feats].astype(float).values
    S = df[share_cols].values
    G = df[grown_cols].values
    rel = np.digitize(S, GRADES)                     # graded relevance 0..4
    L, d = S.shape[1], X.shape[1]

    rows = []
    for regime in exp7.REGIMES:
        splits = exp7.make_splits(df, regime, a.cutoff)
        preds = {m: np.zeros_like(S) for m in ["Popularity", "Pointwise GBDT", "LambdaRank"]}
        mask = np.zeros(len(df), bool)
        for tr, te in splits:
            mask[te] = True
            preds["Popularity"][te] = S[tr].mean(0)
            Xtr, ytr = expand(X[tr], S[tr]); Xte, _ = expand(X[te], S[te])
            reg = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED,
                                verbose=-1, n_jobs=2)
            reg.fit(Xtr, ytr, categorical_feature=[d])
            preds["Pointwise GBDT"][te] = reg.predict(Xte).reshape(len(te), L)
            _, rtr = expand(X[tr], rel[tr])
            rk = LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=31,
                            random_state=SEED, verbose=-1, n_jobs=2)
            rk.fit(Xtr, rtr.astype(int), group=[L] * len(tr), categorical_feature=[d])
            preds["LambdaRank"][te] = rk.predict(Xte).reshape(len(te), L)
        for m, P in preds.items():
            r = evaluate(P[mask], S[mask], G[mask]); r.update({"regime": regime, "method": m})
            rows.append(r)
        print(f"[rank] {regime} done", flush=True)
    res = pd.DataFrame(rows)[["regime", "method", "ndcg@3", "ndcg@5", "precision@3"]]
    save_table(res, out / "tables", "tabD6_ranking_by_regime")
    print(res.round(3).to_string(index=False))

    set_style()
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    meths = ["Popularity", "Pointwise GBDT", "LambdaRank"]
    x = np.arange(len(exp7.REGIMES))
    for i, (m, c) in enumerate(zip(meths, CAT)):
        v = res[res.method == m].set_index("regime").reindex(exp7.REGIMES)["ndcg@3"]
        ax.bar(x + (i - 1) * 0.26, v, width=0.24, color=c, label=m)
    ax.set_xticks(x, [exp7.REGIME_LABEL[r] for r in exp7.REGIMES], rotation=15, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("NDCG@3"); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3)
    ax.set_title("Ranking all crops per district-year", loc="left", fontsize=8, pad=18)
    if stamp:
        fig.text(0.5, 0.5, "SMOKE TEST - FAKE FEATURES", rotation=25, ha="center", va="center",
                 fontsize=20, color="#e34948", alpha=0.35, weight="bold")
    fig.tight_layout()
    savefig(fig, out / "figures", "figD7_ranking_ndcg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/crop_wide_district_year.csv")
    ap.add_argument("--out", default="results/district")
    ap.add_argument("--feature-prefixes", default="soil_,clim_")
    ap.add_argument("--year-min", type=int, default=2010)
    ap.add_argument("--year-max", type=int, default=2017)
    ap.add_argument("--cutoff", type=int, default=2014)
    ap.add_argument("--smoke-test", action="store_true")
    main(ap.parse_args())
