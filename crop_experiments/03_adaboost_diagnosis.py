"""Experiment 3 - Why does AdaBoost sometimes collapse on crop recommendation?

The earlier report (and Singh & Sharma 2025, Sci. Rep. 15:9138, AdaBoost 11.5%)
show AdaBoost scoring far below every other model without explanation. This
script tests the candidate causes under identical 5-fold CV:
  * base learner (decision stump, depth-3 tree, Gaussian NB, RBF-SVM)
  * feature scaling for the SVM base learner
  * number of boosting rounds and learning rate
and records how many rounds AdaBoost actually kept (it stops early when a weak
learner is no better than chance for 22 classes, i.e. weighted error >= 1-1/K).

Outputs -> results/kaggle/{tables,figures}
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from common import load_kaggle, set_style, savefig, save_table, CAT, HILITE, DIM, INK2, MUTED, COL2, SEED

CONFIGS = [
    # name, base learner factory, scale, n_estimators, learning_rate, boosted?
    ("Stump alone", lambda: DecisionTreeClassifier(max_depth=1, random_state=SEED), False, None, None, False),
    ("AdaBoost + stump (50, lr=1)", lambda: DecisionTreeClassifier(max_depth=1, random_state=SEED), False, 50, 1.0, True),
    ("AdaBoost + stump (500, lr=1)", lambda: DecisionTreeClassifier(max_depth=1, random_state=SEED), False, 500, 1.0, True),
    ("Depth-3 tree alone", lambda: DecisionTreeClassifier(max_depth=3, random_state=SEED), False, None, None, False),
    ("AdaBoost + depth-3 (200, lr=0.5)", lambda: DecisionTreeClassifier(max_depth=3, random_state=SEED), False, 200, 0.5, True),
    ("Gaussian NB alone", lambda: GaussianNB(), False, None, None, False),
    ("AdaBoost + Gaussian NB (50, lr=1)", lambda: GaussianNB(), False, 50, 1.0, True),
    ("RBF-SVM alone, unscaled", lambda: SVC(), False, None, None, False),
    ("AdaBoost + RBF-SVM, unscaled (50, lr=1)", lambda: SVC(), False, 50, 1.0, True),
    ("RBF-SVM alone, standardised", lambda: SVC(), True, None, None, False),
    ("AdaBoost + RBF-SVM, standardised (50, lr=1)", lambda: SVC(), True, 50, 1.0, True),
    ("AdaBoost + RBF-SVM, standardised (50, lr=0.1)", lambda: SVC(), True, 50, 0.1, True),
]


def main(a):
    set_style()
    df, X, y, le = load_kaggle(a.data)
    K = len(le.classes_)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, staged = [], {}
    for name, base, scale, n_est, lr, boosted in CONFIGS:
        accs, f1s, kept, first_err = [], [], [], []
        for fold, (tr, te) in enumerate(cv.split(X, y)):
            Xtr, Xte = X[tr], X[te]
            if scale:
                sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            if boosted:
                m = AdaBoostClassifier(base(), n_estimators=n_est, learning_rate=lr, random_state=SEED)
            else:
                m = base()
            m.fit(Xtr, y[tr])
            pred = m.predict(Xte)
            accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
            if boosted:
                kept.append(len(m.estimators_))
                first_err.append(m.estimator_errors_[0])
                if fold == 0:
                    staged[name] = [accuracy_score(y[te], p) for p in m.staged_predict(Xte)]
        rows.append({"configuration": name, "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs, ddof=1),
                     "macro_f1_mean": np.mean(f1s),
                     "rounds_kept_mean": np.mean(kept) if kept else np.nan,
                     "first_round_weighted_error": np.mean(first_err) if first_err else np.nan,
                     "chance_error_threshold_1_minus_1_over_K": 1 - 1 / K})
        print(f"{name:48s} acc={np.mean(accs):.4f}  rounds kept={np.mean(kept) if kept else '-'}")
    res = pd.DataFrame(rows)
    save_table(res, f"{a.out}/tables", "tab03_adaboost_diagnosis")

    # figure: accuracy per configuration (bars from zero) + staged curves -----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 3.3), gridspec_kw={"width_ratios": [1.35, 1]})
    r = res.iloc[::-1]
    boosted = r.configuration.str.startswith("AdaBoost")
    a1.barh(r.configuration, r.accuracy_mean, xerr=r.accuracy_std, color=np.where(boosted, HILITE, DIM),
            height=0.65, error_kw={"lw": 0.8, "ecolor": INK2})
    for i, (v, sd, bst) in enumerate(zip(r.accuracy_mean, r.accuracy_std, boosted)):
        if v > 0.5:   # label inside the bar, clear of the error whisker
            a1.text(0.03, i, f"{v:.3f}", va="center", ha="left", fontsize=6.3,
                    color="white" if bst else "#0b0b0b")
        else:
            a1.text(v + sd + 0.03, i, f"{v:.3f}", va="center", fontsize=6.3, color=INK2)
    a1.set_xlim(0, 1.05); a1.set_xlabel("5-fold CV accuracy"); a1.grid(axis="y", visible=False)
    a1.set_title("(a) Alone (grey) vs AdaBoost (blue)", loc="left")
    a1.tick_params(axis="y", labelsize=6.3)
    sel = ["AdaBoost + stump (500, lr=1)", "AdaBoost + depth-3 (200, lr=0.5)",
           "AdaBoost + Gaussian NB (50, lr=1)", "AdaBoost + RBF-SVM, unscaled (50, lr=1)"]
    short = ["stump", "depth-3 tree", "Gaussian NB", "SVM (unscaled)"]
    for c, name, lab in zip(CAT, sel, short):
        if name in staged:
            cur = staged[name]
            a2.plot(np.arange(1, len(cur) + 1), cur, color=c, label=lab, lw=1.0)
    a2.set_xscale("log"); a2.set_ylim(0, 1.02)
    a2.set_xlabel("Boosting round (log scale)"); a2.set_ylabel("Test accuracy (fold 1)")
    a2.set_title("(b) Test accuracy by boosting round", loc="left")
    a2.legend(loc="center right", bbox_to_anchor=(1.0, 0.6))
    fig.tight_layout()
    savefig(fig, f"{a.out}/figures", "fig08_adaboost_diagnosis")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Crop_recommendation.csv")
    ap.add_argument("--out", default="results/kaggle")
    main(ap.parse_args())
