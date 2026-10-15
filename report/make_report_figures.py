"""Figures used only in the PBL report (the rest come from crop_experiments/results/system/figures).

  fig1_1_architecture.png  system architecture
  fig2_5_overfitting.png   tree-depth curve + train vs test per algorithm (tabS10_*)
  fig2_9_submission.png    steps of the paper submission process
  fig4_1_deck.png          structure of the presentation deck
  paper_page-N.png         pages of paper/main.pdf (Fig. 2.1 and Appendix A)
Run from this folder: python3 make_report_figures.py
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "crop_experiments"))
from common import set_style, CAT, HILITE, DIM, INK, INK2  # noqa: E402

OUT = HERE / "figures"
TAB = HERE.parent / "crop_experiments" / "results" / "system" / "tables"
GREEN, LIGHT, EDGE = "#2F6B45", "#EAF0E6", "#9DB59F"


def box(ax, x, y, w, h, title, body, strong=False, title_size=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=LIGHT if strong else "#FAFAF7", ec=GREEN if strong else EDGE,
                                lw=1.6 if strong else 1.0))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.42, body, ha="center", va="top", fontsize=7, color=INK2, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.3, shrinkA=0, shrinkB=0))


def architecture():
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.6); ax.axis("off")
    box(ax, 0.1, 4.6, 2.2, 1.8, "Farmer input", "Soil test: N, P, K, pH\nWeather: temperature,\nhumidity, rainfall")
    box(ax, 2.6, 4.6, 2.2, 1.8, "Pre-processing", "Label encoding;\nscaling for KNN,\nSVM, Logistic Reg.")
    box(ax, 5.1, 4.6, 2.2, 1.8, "Crop model", "Random Forest\n(chosen from 10\nalgorithms)", strong=True)
    box(ax, 7.6, 4.6, 2.3, 1.8, "Top-3 crops", "Each crop with\na probability")
    box(ax, 5.1, 2.2, 2.2, 1.7, "Explainable AI", "SHAP and LIME:\nwhich inputs led\nto the crop")
    box(ax, 7.6, 2.2, 2.3, 1.7, "Fertilizer module", "XGBoost classifier\n+ nutrient-gap rule\nfor the chosen crop")
    box(ax, 2.6, 0.15, 7.3, 1.35, "Output to the farmer", "Recommended crops, the reasons behind them, and fertilizer advice", strong=True)
    arrow(ax, 2.3, 5.5, 2.6, 5.5); arrow(ax, 4.8, 5.5, 5.1, 5.5); arrow(ax, 7.3, 5.5, 7.6, 5.5)
    arrow(ax, 6.2, 4.6, 6.2, 3.9); arrow(ax, 8.75, 4.6, 8.75, 3.9)
    arrow(ax, 6.2, 2.2, 6.2, 1.5); arrow(ax, 8.75, 2.2, 8.75, 1.5)
    fig.savefig(OUT / "fig1_1_architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def overfitting():
    depth = pd.read_csv(TAB / "tabS10_tree_depth.csv")
    gap = pd.read_csv(TAB / "tabS10_train_vs_test.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.55))
    ax = axes[0]
    xs = range(len(depth))
    ax.plot(xs, depth.train_accuracy * 100, "o-", color=CAT[1], ms=3.5, label="Training")
    ax.plot(xs, depth.test_accuracy * 100, "o-", color=CAT[0], ms=3.5, label="Test")
    ax.set_xticks(list(xs)); ax.set_xticklabels([str(v) for v in depth.max_depth])
    ax.set_xlabel("Decision-tree maximum depth"); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105); ax.legend(loc="lower right"); ax.set_title("(a) Training and test rise together")
    ax = axes[1]
    g = gap.iloc[::-1]
    ys = range(len(g)); h = 0.38
    ax.barh([y + h / 2 for y in ys], g.train_accuracy * 100, h, color=CAT[1], label="Training")
    ax.barh([y - h / 2 for y in ys], g.test_accuracy * 100, h, color=CAT[0], label="Test")
    ax.set_yticks(list(ys)); ax.set_yticklabels(g.model, fontsize=6.5)
    ax.set_xlabel("Accuracy (%)"); ax.set_xlim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.5)
    ax.set_title("(b) Training vs test for each algorithm")
    fig.savefig(OUT / "fig2_5_overfitting.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def submission():
    steps = [("1. Internal review", "Guide checks the\nmanuscript; numbers\nchecked against results"),
             ("2. Plagiarism check", "University tool;\nsimilarity report"),
             ("3. Format check", "Template, page limit,\nPDF compatibility"),
             ("4. Online submission", "PDF, abstract and\nauthor details"),
             ("5. Peer review", "2–3 reviewers score\nand comment"),
             ("6. Revision", "Changes + response\nto reviewers"),
             ("7. Camera-ready", "Final PDF, copyright\nform, registration"),
             ("8. Presentation", "Talk at the conference;\nproceedings published")]
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    w, hgt = 2.35, 1.65
    xs = [0.05, 2.55, 5.05, 7.55]
    for i, (t, b) in enumerate(steps):
        row = 0 if i < 4 else 1
        col = i if i < 4 else 7 - i
        x, y = xs[col], 2.75 if row == 0 else 0.25
        box(ax, x, y, w, hgt, t, b, strong=(i == 4), title_size=7.6)
    for c in range(3):
        arrow(ax, xs[c] + w, 3.6, xs[c + 1], 3.6)
        arrow(ax, xs[c + 1], 1.1, xs[c] + w, 1.1)
    arrow(ax, xs[3] + w / 2, 2.75, xs[3] + w / 2, 0.25 + hgt)
    fig.savefig(OUT / "fig2_9_submission.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def deck():
    sections = [("Introduction", 3, "Cover,\nproblem,\nsystem overview"),
                ("Datasets", 4, "Datasets,\nwhy chosen,\ndata analysis,\ntrain/test setup"),
                ("Algorithms", 8, "10 algorithms,\nresults, why RF,\nAdaBoost failure,\nconfusion matrix,\noverfitting, farmers"),
                ("Explainable AI", 2, "Global and\nper-crop SHAP,\none farmer\n(SHAP vs LIME)"),
                ("Fertilizer", 6, "7 fertilizers,\nresults, importance,\nnutrient-gap rule,\nexamples,\none farmer"),
                ("Wrap-up", 3, "Limits,\nfuture work,\nsummary and\nquestions")]
    fig, ax = plt.subplots(figsize=(6.8, 1.9))
    ax.set_xlim(0, 6); ax.set_ylim(0.6, 3.25); ax.axis("off")
    for i, (name, n, body) in enumerate(sections):
        ax.add_patch(FancyBboxPatch((i + 0.04, 2.35), 0.92, 0.8, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=GREEN if i % 2 == 0 else "#5E8C6A", ec="none"))
        ax.text(i + 0.5, 2.75, f"{name}\n{n} slides", ha="center", va="center", fontsize=7, color="white",
                fontweight="bold")
        ax.text(i + 0.5, 2.2, body, ha="center", va="top", fontsize=5.9, color=INK2, linespacing=1.35)
    fig.savefig(OUT / "fig4_1_deck.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def paper_pages():
    pdf = HERE.parent / "paper" / "main.pdf"
    subprocess.run(["pdftoppm", "-r", "150", "-png", str(pdf), str(OUT / "paper_page")], check=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    set_style()
    architecture(); overfitting(); submission(); deck(); paper_pages()
    print(sorted(p.name for p in OUT.iterdir()))
