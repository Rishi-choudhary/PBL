"""Experiment 15 - Can the crop dataset and the fertilizer dataset be combined, and does the chained
crop -> fertilizer system work? (mentor request)

A. Combine the two datasets?  Compare the 5 features they share (N, P, K, temperature, humidity): value
   ranges, crop-name vocabularies, and whether a classifier can tell which dataset a row came from.
B. How many of the 440 held-out crop farmers fall inside the range the ML fertilizer model was trained on?
C. Does multimodel.CropFertilizerSystem reproduce the crop and fertilizer-advice results of experiment 12
   (tabS4, tabS9) for the six sample farmers? Then show its two-route fertilizer output for them.
D. Two hand-entered inputs (NOT data, NOT used in any number) to show the full chain once.

Outputs -> results/system/tables/tabS13_* .. tabS15_*, figures/figS8_dataset_overlap, summary_multimodel.txt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from common import load_kaggle, KAGGLE_FEATURES, set_style, savefig, save_table, CAT, HILITE, DIM, INK2, COL2, SEED
from multimodel import CropFertilizerSystem, CROP_TO_FERT_TYPE, FARMER_TO_FERT, format_result

SHARED = ["N", "P", "K", "temperature", "humidity"]            # same meaning in both datasets
NICE = {"N": "N", "P": "P", "K": "K", "temperature": "Temperature", "humidity": "Humidity"}


def main(a):
    set_style()
    out = Path(a.out); fig_dir, tab_dir = out / "figures", out / "tables"
    log = []
    df, X, y, le = load_kaggle(a.crop)
    crops = list(le.classes_)
    fd = pd.read_csv(a.fert); fd.columns = [c.strip() for c in fd.columns]
    fcol = {k: v for k, v in FARMER_TO_FERT.items() if k in SHARED}          # shared feature -> fertilizer column

    # ================================================================ A. can the datasets be combined?
    rows = []
    for f in SHARED:
        c, g = df[f], fd[fcol[f]]
        rows.append(dict(feature=NICE[f], crop_min=c.min(), crop_max=c.max(), crop_mean=c.mean(),
                         fert_min=g.min(), fert_max=g.max(), fert_mean=g.mean(),
                         crop_rows_inside_fert_range=((c >= g.min()) & (c <= g.max())).mean(),
                         fert_rows_inside_crop_range=((g >= c.min()) & (g <= c.max())).mean()))
    rng_tab = pd.DataFrame(rows)
    save_table(rng_tab, tab_dir, "tabS13_shared_feature_ranges", float_fmt="%.3f")
    log.append(f"A. crop dataset {len(df)} rows x {len(df.columns)} columns; fertilizer dataset {len(fd)} rows x "
               f"{len(fd.columns)} columns; neither has a row identifier")
    log.append("A. shared features (crop min-max | fertilizer min-max | share of crop rows inside the fertilizer range):")
    for _, r in rng_tab.iterrows():
        log.append(f"   {r.feature:12s} {r.crop_min:7.1f}-{r.crop_max:6.1f} | {r.fert_min:5.1f}-{r.fert_max:5.1f} | "
                   f"{r.crop_rows_inside_fert_range:.3f}")
    log.append("A. columns only in the crop dataset: pH, rainfall; only in the fertilizer dataset: moisture, soil type, crop type")

    name_map = pd.DataFrame([dict(crop=c, fertilizer_crop_type=CROP_TO_FERT_TYPE.get(c, "none")) for c in crops])
    save_table(name_map, tab_dir, "tabS13_crop_name_map")
    ftypes = sorted(fd["Crop Type"].unique())
    matched = sorted(set(CROP_TO_FERT_TYPE.values()))
    log.append(f"A. crops with a fertilizer crop type: {int((name_map.fertilizer_crop_type != 'none').sum())} of {len(crops)} "
               f"({', '.join(c for c in crops if c in CROP_TO_FERT_TYPE)}); without: "
               f"{', '.join(c for c in crops if c not in CROP_TO_FERT_TYPE)}")
    log.append(f"A. fertilizer crop types with a crop-dataset counterpart: {len(matched)} of {len(ftypes)} ({', '.join(matched)}); "
               f"without: {', '.join(t for t in ftypes if t not in matched)}")

    Z = np.vstack([df[SHARED].values, fd[[fcol[f] for f in SHARED]].values.astype(float)])
    src = np.r_[np.zeros(len(df), int), np.ones(len(fd), int)]
    pr = cross_val_predict(RandomForestClassifier(200, class_weight="balanced", random_state=SEED, n_jobs=-1), Z, src,
                           cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    bal, auc = balanced_accuracy_score(src, pr > 0.5), roc_auc_score(src, pr)
    save_table(pd.DataFrame([dict(rows_crop=len(df), rows_fert=len(fd), balanced_accuracy=bal, roc_auc=auc)]),
               tab_dir, "tabS13_source_classifier")
    log.append(f"A. which dataset did a row come from? Random Forest on the 5 shared features, 5-fold CV: "
               f"balanced accuracy {bal:.4f}, ROC AUC {auc:.4f} (0.5 = cannot tell them apart)")

    # ================================================================ system trained on the 1,760 training rows
    tr_idx, te_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    assert np.array_equal(X[te_idx], Xte), "split differs from experiment 12"
    system = CropFertilizerSystem().fit(train_rows=tr_idx)

    # ================================================================ B. coverage of the ML fertilizer route
    inside = {}
    for f in SHARED:
        lo, hi = system.fert_range[fcol[f]]
        v = Xte[:, KAGGLE_FEATURES.index(f)]
        inside[f] = (v >= lo) & (v <= hi)
    all5 = np.logical_and.reduce(list(inside.values()))
    top1 = np.argmax(system.crop_probabilities(Xte), axis=1)
    mapped = np.array([crops[i] in CROP_TO_FERT_TYPE for i in top1])
    cov = [dict(condition=f"{NICE[f]} inside the fertilizer range", farmers=int(m.sum()), of=len(Xte), share=m.mean())
           for f, m in inside.items()]
    cov += [dict(condition="all five inside", farmers=int(all5.sum()), of=len(Xte), share=all5.mean()),
            dict(condition="top-1 crop has a fertilizer crop type", farmers=int(mapped.sum()), of=len(Xte), share=mapped.mean()),
            dict(condition="all five inside and top-1 crop mapped", farmers=int((all5 & mapped).sum()), of=len(Xte),
                 share=(all5 & mapped).mean()),
            dict(condition="soil moisture and soil type given (not columns of the crop dataset)", farmers=0, of=len(Xte), share=0.0)]
    save_table(pd.DataFrame(cov), tab_dir, "tabS14_ml_fertilizer_coverage", float_fmt="%.4f")
    log.append(f"B. held-out crop farmers ({len(Xte)}) inside the range of the fertilizer training data:")
    for r in cov:
        log.append(f"   {r['condition']:68s} {r['farmers']:4d} / {r['of']}  ({r['share']:.3f})")

    # ================================================================ C. six sample farmers
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(crops), 6, replace=False)
    idx = [int(rng.choice(np.where(yte == c)[0])) for c in pick]                   # same draw as experiment 12
    s4 = pd.read_csv(tab_dir / "tabS4_farmer_examples.csv")
    s9 = pd.read_csv(tab_dir / "tabS9_nutrient_gap_examples.csv")
    out_rows, ok4, ok9 = [], 0, 0
    for k, i in enumerate(idx):
        x = Xte[i]
        farmer = dict(zip(KAGGLE_FEATURES, x)); farmer.update(moisture=None, soil_type=None)
        r = system.recommend(farmer)
        tops = [f"{c['crop']} ({c['probability']:.2f})" for c in r["crops"]]
        ok4 += tops == s4.loc[k, ["top1", "top2", "top3"]].tolist()
        for rank, c in enumerate(r["crops"][:2], 1):
            ref = s9[(s9.farmer == f"F{k+1}") & (s9["rank"] == rank)].iloc[0]
            ok9 += (c["crop"] == ref.crop and c["nutrient_gap"]["largest_gap"] == ref.largest_gap
                    and c["nutrient_gap"]["advice"] == ref.advice)
        pr_ = r["fertilizer_product"]
        out_rows.append(dict(farmer=f"F{k+1}", soil_NPK=f"{x[0]:.0f}-{x[1]:.0f}-{x[2]:.0f}", top1=tops[0], top2=tops[1], top3=tops[2],
                             top1_required_NPK="-".join(map(str, r["crops"][0]["nutrient_gap"]["required_NPK"])),
                             top1_largest_gap=r["crops"][0]["nutrient_gap"]["largest_gap"],
                             top1_advice=r["crops"][0]["nutrient_gap"]["advice"],
                             ml_product="used: " + pr_["product"] if pr_["applicable"] else "not used: " + "; ".join(pr_["reasons"])))
    save_table(pd.DataFrame(out_rows), tab_dir, "tabS15_multimodel_six_farmers")
    log.append(f"C. multimodel reproduces experiment 12: top-3 crops identical for {ok4} of 6 farmers (tabS4); "
               f"nutrient-gap rows identical for {ok9} of 12 (tabS9)")
    for r in out_rows:
        log.append(f"   {r['farmer']} {r['soil_NPK']:10s} {r['top1']:22s} gap {r['top1_largest_gap']:14s} {r['top1_advice']}")
        log.append(f"       ML product: {r['ml_product']}")

    # ================================================================ D. hand-entered illustrations
    log.append("D. hand-entered inputs (typed by us, not from any dataset; they show the code path only, not a result):")
    for name, farmer in [("example 1, all inputs inside the fertilizer range",
                          dict(N=21, P=42, K=19, temperature=28, humidity=55, ph=6.8, rainfall=50, moisture=40, soil_type="Sandy")),
                         ("example 2, same farmer with N = 90 and K = 43",
                          dict(N=90, P=42, K=43, temperature=28, humidity=55, ph=6.8, rainfall=50, moisture=40, soil_type="Sandy"))]:
        log.append(f"   {name}: {farmer}")
        log.extend("   " + l for l in format_result(system.recommend(farmer)).splitlines())

    # ================================================================ figure
    fig = plt.figure(figsize=(COL2, 3.7))
    gs = fig.add_gridspec(5, 2, width_ratios=[1.35, 1], wspace=0.5, hspace=1.55, top=0.92)
    for r, (_, row) in enumerate(rng_tab.iterrows()):
        ax = fig.add_subplot(gs[r, 0])
        ax.barh([1], [row.crop_max - row.crop_min], left=row.crop_min, height=0.55, color=HILITE)
        ax.barh([0], [row.fert_max - row.fert_min], left=row.fert_min, height=0.55, color=CAT[1])
        ax.set_yticks([1, 0]); ax.set_yticklabels(["Crop data", "Fertilizer data"], fontsize=6)
        ax.set_ylim(-0.55, 1.55); ax.set_title(row.feature, loc="left", fontsize=7, pad=2)
        ax.tick_params(axis="x", labelsize=6); ax.grid(axis="y", visible=False)
        ax.set_xlim(min(0, row.crop_min) - 0.02 * row.crop_max, row.crop_max * 1.2)
        ax.text(row.crop_max, 1, f" {row.crop_min:.0f}-{row.crop_max:.0f}", va="center", ha="left", fontsize=5.5, color=INK2)
        ax.text(row.fert_max, 0, f" {row.fert_min:.0f}-{row.fert_max:.0f}", va="center", ha="left", fontsize=5.5, color=INK2)
        if r == 0:
            ax_first = ax
    ax = fig.add_subplot(gs[:, 1])
    labs = [NICE[f] for f in SHARED] + ["All five together"]
    vals = [m.mean() * 100 for m in inside.values()] + [all5.mean() * 100]
    cnts = [int(m.sum()) for m in inside.values()] + [int(all5.sum())]
    ys = np.arange(len(labs))[::-1]
    ax.barh(ys, vals, height=0.55, color=[DIM] * 5 + [CAT[7]])
    for yy, v, n in zip(ys, vals, cnts):
        ax.text(v + 1.5, yy, f"{n} of {len(Xte)}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels(labs); ax.set_xlim(0, 100); ax.set_xlabel("Held-out crop farmers (%)")
    ax.grid(axis="y", visible=False)
    fig.text(ax_first.get_position().x0 - 0.14, 0.965, "(a) Range of each shared input (min to max)", fontsize=8, ha="left")
    fig.text(ax.get_position().x0 - 0.1, 0.965, "(b) Inside the fertilizer data's range", fontsize=8, ha="left")
    h = [plt.Line2D([], [], marker="s", ls="", color=HILITE, ms=5, label="Crop dataset"),
         plt.Line2D([], [], marker="s", ls="", color=CAT[1], ms=5, label="Fertilizer dataset")]
    fig.legend(handles=h, loc="lower left", ncol=2, bbox_to_anchor=(0.04, -0.06), fontsize=6.5)
    savefig(fig, fig_dir, "figS8_dataset_overlap")

    (out / "summary_multimodel.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", default="data/Crop_recommendation.csv")
    ap.add_argument("--fert", default="data/Fertilizer_Prediction_99rows.csv")
    ap.add_argument("--out", default="results/system")
    main(ap.parse_args())
