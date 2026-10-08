from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# ── Define all 4 models ───────────────────────────────────────────────────

models = {
    "Random Forest\n(Baseline)": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "AdaBoost +\nSVM": AdaBoostClassifier(
        estimator=SVC(probability=True, kernel='rbf', C=1.0),
        n_estimators=50,
        random_state=42,

    ),
    "AdaBoost +\nNaive Bayes": AdaBoostClassifier(
        estimator=GaussianNB(),
        n_estimators=50,
        random_state=42,

    ),
    "AdaBoost +\nLogistic Reg.": AdaBoostClassifier(
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        n_estimators=50,
        random_state=42,

    ),
}

# ── Train all models and collect results ─────────────────────────────────

results      = {}
trained      = {}
colors       = ["#6B7280", "#6B1A2A", "#C8952A", "#028090"]

print("Training models...\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds    = model.predict(X_test)
    acc      = accuracy_score(y_test, preds)
    results[name] = round(acc * 100, 2)
    trained[name] = (model, preds)
    print(f"{name.replace(chr(10),' ')}: {acc:.4f}")

# ── RESULT: Bar chart comparison ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
names  = list(results.keys())
values = list(results.values())

bars = ax.bar(names, values, color=colors, width=0.5, edgecolor='white')

# Value labels on top of bars
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.08,
        f"{val}%",
        ha='center', va='bottom',
        fontsize=12, fontweight='bold', color='#1F2937'
    )

ax.set_ylim(85, 102)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title(
    "Model Accuracy Comparison\nBaseline vs AdaBoost Ensemble Methods",
    fontsize=13, fontweight='bold', pad=14
)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(axis='x', labelsize=10)
ax.axhline(y=99, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.text(3.4, 99.1, 'existing baseline', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig('result_adaboost_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved result_adaboost_comparison.png")
plt.close()

# ── RESULT: Full metrics table ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 5))
ax.axis('off')

# Build table data
table_data   = []
col_headers  = ["Model", "Accuracy %", "Precision", "Recall", "F1-Score", "Type"]
model_types  = ["Baseline", "Ensemble", "Ensemble", "Ensemble"]

for (name, (model, preds)), mtype in zip(trained.items(), model_types):
    preds_lbl = le.inverse_transform(preds)
    test_lbl  = le.inverse_transform(y_test)
    rep       = classification_report(
        test_lbl, preds_lbl, output_dict=True, zero_division=0
    )
    wa        = rep['weighted avg']
    table_data.append([
        name.replace('\n', ' '),
        f"{results[name]}%",
        f"{wa['precision']:.3f}",
        f"{wa['recall']:.3f}",
        f"{wa['f1-score']:.3f}",
        mtype
    ])

tbl = ax.table(
    cellText=table_data,
    colLabels=col_headers,
    cellLoc='center',
    loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.2)

# Header style
for j in range(len(col_headers)):
    tbl[0, j].set_facecolor('#6B1A2A')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# Row styles
row_colors = ['#F9FAFB', '#FEF9EE', '#FEF9EE', '#FEF9EE']
for i in range(1, len(table_data) + 1):
    for j in range(len(col_headers)):
        tbl[i, j].set_facecolor(row_colors[i-1])
    # Highlight best accuracy
    best_acc = max(results.values())
    if float(table_data[i-1][1].replace('%','')) == best_acc:
        tbl[i, 1].set_facecolor('#D1FAE5')
        tbl[i, 1].set_text_props(color='#065F46', fontweight='bold')

plt.title(
    "Complete Performance Metrics — All Models\n(Weighted Average across 22 Crop Classes)",
    fontsize=12, fontweight='bold', pad=16
)
plt.tight_layout()
plt.savefig('result_full_metrics_table.png', dpi=150, bbox_inches='tight')
print("Saved result_full_metrics_table.png")
plt.close()

print("\nDone. Paste both images into your PPT.")
