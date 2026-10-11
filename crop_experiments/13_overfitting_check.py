"""Experiment 13 - Is the 99.5% crop accuracy overfitting?

Same 80/20 split and models as 12_crop_system.py:
  (a) training vs test accuracy for each algorithm (the gap shows overfitting)
  (b) decision-tree depth curve, training vs test
  (c) label-permutation test for the Random Forest (5-fold CV on the training set)

Outputs -> results/system/tables/tabS10_*.csv
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, permutation_test_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from common import load_kaggle, save_table, SEED

spec = importlib.util.spec_from_file_location("crop_system", Path(__file__).with_name("12_crop_system.py"))
crop_system = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crop_system)


def main(a):
    tab_dir = Path(a.out) / "tables"
    _, X, y, _ = load_kaggle(a.crop)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    rows = []
    for name, m in crop_system.crop_models().items():
        m.fit(Xtr, ytr)
        tr, te = accuracy_score(ytr, m.predict(Xtr)), accuracy_score(yte, m.predict(Xte))
        rows.append(dict(model=name, train_accuracy=tr, test_accuracy=te, gap_pp=100 * (tr - te)))
    gap = pd.DataFrame(rows)
    save_table(gap, tab_dir, "tabS10_train_vs_test")

    depth = []
    for d in [2, 4, 6, 8, 10, 12, 15, None]:
        t = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(Xtr, ytr)
        depth.append(dict(max_depth="none" if d is None else d,
                          train_accuracy=accuracy_score(ytr, t.predict(Xtr)),
                          test_accuracy=accuracy_score(yte, t.predict(Xte))))
    save_table(pd.DataFrame(depth), tab_dir, "tabS10_tree_depth")

    score, perm, p = permutation_test_score(
        RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1), Xtr, ytr,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_permutations=a.permutations, random_state=SEED)
    save_table(pd.DataFrame([dict(real_label_cv_accuracy=score, shuffled_mean=perm.mean(), shuffled_max=perm.max(),
                                  chance=1 / len(np.unique(y)), p_value=p, n_permutations=a.permutations)]),
               tab_dir, "tabS10_permutation_test")
    print(gap.to_string(index=False))
    print(pd.DataFrame(depth).to_string(index=False))
    print(f"RF real {score:.4f}, shuffled mean {perm.mean():.4f} max {perm.max():.4f}, p {p:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", default="data/Crop_recommendation.csv")
    ap.add_argument("--out", default="results/system")
    ap.add_argument("--permutations", type=int, default=100)
    main(ap.parse_args())
