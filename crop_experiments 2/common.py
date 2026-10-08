"""Shared utilities: data loading, model zoo, metrics, statistics, conformal
prediction and figure style. Every experiment script imports from here so the
models and metrics are identical across experiments."""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, AdaBoostClassifier,
                              BaggingClassifier, VotingClassifier, StackingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, log_loss,
                             top_k_accuracy_score)

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).resolve().parent

KAGGLE_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_kaggle(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    le = LabelEncoder()
    y = le.fit_transform(df["label"].astype(str).str.strip())
    X = df[KAGGLE_FEATURES].astype(float).values
    return df, X, y, le


# ----------------------------------------------------------------------------
# Model zoo
# ----------------------------------------------------------------------------

def _xgb(seed):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                         subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                         n_jobs=2, random_state=seed, eval_metric="mlogloss")


def _lgbm(seed):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          min_child_samples=10, random_state=seed, verbose=-1, n_jobs=2)


def _cat(seed):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1,
                              random_seed=seed, verbose=0, thread_count=2,
                              allow_writing_files=False)


def _tabpfn(seed, n_classes):
    """TabPFN needs `pip install tabpfn "tabpfn-extensions[many_class]"` and
    internet access to HuggingFace for the weights (works on Colab). TabPFN-2.5/2.6
    support at most 10 classes natively, so >10 classes uses ManyClassClassifier."""
    from tabpfn import TabPFNClassifier
    base = TabPFNClassifier(random_state=seed)
    if n_classes <= 10:
        return base
    from tabpfn_extensions.many_class import ManyClassClassifier
    return ManyClassClassifier(estimator=base, n_estimators_redundancy=4, random_state=seed)


def get_models(seed=SEED, include=None, n_classes=22, with_tabpfn=False):
    """Return an ordered dict name -> unfitted estimator. `include` filters by name."""
    m = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=10)),
        "Gaussian NB": GaussianNB(),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=seed)),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
        "Hist. Gradient Boosting": HistGradientBoostingClassifier(random_state=seed),
        "XGBoost": _xgb(seed),
        "LightGBM": _lgbm(seed),
        "CatBoost": _cat(seed),
        "MLP (128-64)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=2000,
                                                                       early_stopping=True, random_state=seed)),
        "AdaBoost (tree d=3)": AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=200,
                                                  learning_rate=0.5, random_state=seed),
        "Bagging (DT)": BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, n_jobs=-1, random_state=seed),
        "Soft Voting (RF+XGB+LGBM)": VotingClassifier(
            [("rf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)),
             ("xgb", _xgb(seed)), ("lgbm", _lgbm(seed))], voting="soft"),
        "Stacking (RF,ET,SVM,LGBM -> LR)": StackingClassifier(
            [("rf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)),
             ("et", ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=seed)),
             ("svm", make_pipeline(StandardScaler(), SVC(C=10, probability=True, random_state=seed))),
             ("lgbm", _lgbm(seed))],
            final_estimator=LogisticRegression(max_iter=5000), cv=5, n_jobs=1),
    }
    if with_tabpfn:
        try:
            m["TabPFN v2"] = _tabpfn(seed, n_classes)
        except Exception as e:  # not installed / no weights
            print(f"[TabPFN skipped: {type(e).__name__}: {e}]")
    if include:
        m = {k: v for k, v in m.items() if k in include}
    return m


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def aligned_proba(model, X, n_classes):
    """predict_proba mapped onto the global class index space (handles folds in
    which some classes are absent from training)."""
    P = model.predict_proba(X)
    out = np.zeros((P.shape[0], n_classes))
    cls = np.asarray(getattr(model, "classes_", np.arange(P.shape[1]))).astype(int)
    out[:, cls] = P
    return out


def ece_score(y, P, n_bins=15):
    conf = P.max(1)
    correct = (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return ece


def brier_multiclass(y, P):
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1
    return np.mean(np.sum((P - onehot) ** 2, axis=1))


def all_metrics(y, P, n_classes):
    pred = P.argmax(1)
    labels = np.arange(n_classes)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0),
        "macro_precision": precision_score(y, pred, average="macro", labels=np.unique(y), zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", labels=np.unique(y), zero_division=0),
        "top3_accuracy": top_k_accuracy_score(y, P, k=min(3, n_classes), labels=labels),
        "log_loss": log_loss(y, np.clip(P, 1e-15, 1), labels=labels),
        "brier": brier_multiclass(y, P),
        "ece": ece_score(y, P),
    }


def timed_fit_predict(model, Xtr, ytr, Xte, n_classes):
    t0 = time.perf_counter(); model.fit(Xtr, ytr); fit_s = time.perf_counter() - t0
    t0 = time.perf_counter(); P = aligned_proba(model, Xte, n_classes); pred_s = time.perf_counter() - t0
    return P, fit_s, pred_s


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------

def corrected_repeated_cv_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test, in the repeated k-fold
    form of Bouckaert & Frank (2004). a, b: per-fold scores of two models on the
    same folds. Returns (t, p two-sided)."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1.0 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return t, p


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def mean_ci(x, conf=0.95):
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2:
        return m, m, m
    h = stats.t.ppf((1 + conf) / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


# ----------------------------------------------------------------------------
# Split conformal prediction (classification)
# ----------------------------------------------------------------------------

def conformal_quantile(scores, alpha):
    n = len(scores)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    return np.quantile(scores, min(q, 1.0), method="higher")


def lac_scores(P, y):
    return 1.0 - P[np.arange(len(y)), y]


def lac_sets(P, qhat):
    return P >= 1.0 - qhat


def aps_scores(P, y, rng):
    order = np.argsort(-P, axis=1)
    Ps = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(Ps, axis=1)
    rank = np.argmax(order == y[:, None], axis=1)
    u = rng.uniform(size=len(y))
    return cum[np.arange(len(y)), rank] - u * Ps[np.arange(len(y)), rank]


def aps_sets(P, qhat, rng):
    order = np.argsort(-P, axis=1)
    Ps = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(Ps, axis=1)
    u = rng.uniform(size=(P.shape[0], 1))
    include_sorted = (cum - u * Ps) <= qhat
    include_sorted[:, 0] = True  # never return an empty set
    S = np.zeros_like(P, dtype=bool)
    np.put_along_axis(S, order, include_sorted, axis=1)
    return S


# ----------------------------------------------------------------------------
# Figure style (print, light background; palette from the dataviz reference)
# ----------------------------------------------------------------------------

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
HILITE = CAT[0]
DIM = "#b9b8b1"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f7f9fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b8302f"])

COL1 = 3.5   # IEEE single column width (in)
COL2 = 7.16  # IEEE double column width (in)


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False,
        "lines.linewidth": 1.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def savefig(fig, out_dir, name):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure {name}")


def save_table(df, out_dir, name, float_fmt="%.4f", index=False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=index, float_format=float_fmt)
    try:
        (out_dir / f"{name}.tex").write_text(df.to_latex(index=index, float_format=lambda v: f"{v:.4f}"))
    except Exception:
        pass
    print(f"  saved table {name}")
