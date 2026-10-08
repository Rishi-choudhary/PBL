# ═══════════════════════════════════════════════════════════════════════════
#  Crop Recommendation System — Research & Visualization
#  Trains 6 ML models, generates 6 PPT-quality result images
# ═══════════════════════════════════════════════════════════════════════════

# ── Imports ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from matplotlib.patches import Patch

# ── Global Style ──────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titleweight': 'bold',
    'axes.titlesize': 14,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

PALETTE = ['#378ADD', '#7F77DD', '#1D9E75', '#BA7517', '#E05A5A', '#2DBDAD']
DPI = 200

# ═══════════════════════════════════════════════════════════════════════════
#  1. LOAD & SPLIT DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  CROP RECOMMENDATION SYSTEM — RESEARCH RESULTS")
print("=" * 60)

df = pd.read_csv('Crop_recommendation.csv')
X = df.drop('label', axis=1)
y = df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print(f"\nDataset: {len(df)} samples, {X.shape[1]} features, {len(le.classes_)} crops")
print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

# ═══════════════════════════════════════════════════════════════════════════
#  2. TRAIN 6 MODELS
# ═══════════════════════════════════════════════════════════════════════════
models = {
    'Random Forest':  RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':        XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                                     random_state=42),
    'Decision Tree':  DecisionTreeClassifier(random_state=42),
    'Naive Bayes':    GaussianNB(),
    'KNN':            KNeighborsClassifier(),
    'SVM':            SVC(random_state=42),
}

results = {}
print("Model Accuracies:")
print("-" * 40)
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = round(acc * 100, 2)
    print(f"  {name:20s} {acc:.4f}  ({results[name]}%)")

# Also print NB train score for reference
nb = models['Naive Bayes']
print(f"\n  Naive Bayes Train Acc: {nb.score(X_train, y_train):.4f}")
print(f"  Naive Bayes Test Acc:  {nb.score(X_test, y_test):.4f}")

# Best model
rf = models['Random Forest']
y_pred = rf.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)


# ═══════════════════════════════════════════════════════════════════════════
#  3. RESULT IMAGE 1 — Model Accuracy Comparison (Bar Chart)
# ═══════════════════════════════════════════════════════════════════════════
sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(sorted_results.keys(), sorted_results.values(),
              color=PALETTE[:len(sorted_results)], edgecolor='white', width=0.6)

ax.set_ylim(92, 101)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Model Accuracy Comparison\nCrop Recommendation System', fontsize=14, pad=12)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(axis='x', rotation=15)

for bar, val in zip(bars, sorted_results.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
            f'{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('result1_model_comparison.png', dpi=DPI)
print("\n✓ Saved result1_model_comparison.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  4. RESULT IMAGE 2 — Confusion Matrix (Random Forest)
# ═══════════════════════════════════════════════════════════════════════════
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)

fig, ax = plt.subplots(figsize=(16, 13))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=le.classes_, yticklabels=le.classes_,
    linewidths=0.5, linecolor='#e0e0e0',
    cbar_kws={'shrink': 0.75, 'label': 'Count'},
    ax=ax
)
ax.set_title('Confusion Matrix — Random Forest\n(Best Performing Model: {:.2f}% Accuracy)'.format(
    results['Random Forest']), fontsize=14, pad=16)
ax.set_xlabel('Predicted Crop', fontsize=12, labelpad=10)
ax.set_ylabel('Actual Crop', fontsize=12, labelpad=10)
ax.tick_params(axis='x', rotation=45, labelsize=9)
ax.tick_params(axis='y', rotation=0, labelsize=9)
plt.tight_layout()
plt.savefig('result2_confusion_matrix.png', dpi=DPI)
print("✓ Saved result2_confusion_matrix.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  5. RESULT IMAGE 3 — Classification Report Table
# ═══════════════════════════════════════════════════════════════════════════
report = classification_report(
    y_test_labels, y_pred_labels,
    target_names=le.classes_, output_dict=True
)

report_df = pd.DataFrame(report).T
report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
report_df = report_df[['precision', 'recall', 'f1-score', 'support']]
report_df = report_df.round(3)
report_df['support'] = report_df['support'].astype(int)
report_df.index.name = 'Crop'
report_df = report_df.reset_index()

fig, ax = plt.subplots(figsize=(12, len(report_df) * 0.5 + 2.5))
ax.axis('off')

tbl = ax.table(
    cellText=report_df.values,
    colLabels=['Crop', 'Precision', 'Recall', 'F1-Score', 'Support'],
    cellLoc='center', loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.7)

# Style header row
for j in range(5):
    tbl[0, j].set_facecolor('#2c5f8a')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# Color rows by F1 score
for i in range(1, len(report_df) + 1):
    f1 = float(report_df.iloc[i - 1]['f1-score'])
    if f1 >= 0.99:
        color = '#d4edda'   # green  — excellent
    elif f1 >= 0.95:
        color = '#fff3cd'   # yellow — good
    else:
        color = '#f8d7da'   # red    — needs attention
    for j in range(5):
        tbl[i, j].set_facecolor(color)

ax.set_title(
    'Classification Report — Random Forest\n'
    'Green: F1 >= 0.99  |  Yellow: F1 >= 0.95  |  Red: F1 < 0.95',
    fontsize=13, fontweight='bold', pad=20
)
plt.tight_layout()
plt.savefig('result3_classification_report.png', dpi=DPI)
print("✓ Saved result3_classification_report.png")
plt.close()

# Print to terminal as well
print("\nClassification Report (Random Forest):")
print(classification_report(y_test_labels, y_pred_labels, target_names=le.classes_))


# ═══════════════════════════════════════════════════════════════════════════
#  6. RESULT IMAGE 4 — SHAP Feature Importance
# ═══════════════════════════════════════════════════════════════════════════
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Mean absolute SHAP across all classes → one value per feature
mean_shap = np.abs(shap_values).mean(axis=(0, 2))
feature_names = X.columns.tolist()

# Sort by importance
sorted_idx = np.argsort(mean_shap)
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_values = mean_shap[sorted_idx]

# Clean readable labels
label_map = {
    'N': 'Nitrogen (N)', 'P': 'Phosphorus (P)', 'K': 'Potassium (K)',
    'temperature': 'Temperature', 'humidity': 'Humidity',
    'ph': 'Soil pH', 'rainfall': 'Rainfall',
}
sorted_labels = [label_map.get(f, f) for f in sorted_features]

# Color by importance level
colors = ['#d4edda' if v < 0.3 else '#fff3cd' if v < 0.5 else '#378ADD'
          for v in sorted_values]

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.barh(sorted_labels, sorted_values, color=colors,
               edgecolor='white', height=0.6)

for bar, val in zip(bars, sorted_values):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=10, fontweight='bold', color='#333')

ax.set_xlabel('Mean |SHAP Value| — Average Impact on Prediction', fontsize=11)
ax.set_title('SHAP Feature Importance\nWhich soil/weather factor drives crop recommendation most?',
             fontsize=13, fontweight='bold', pad=14)
ax.set_xlim(0, sorted_values.max() * 1.2)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(axis='y', labelsize=11)

legend = [
    Patch(facecolor='#378ADD', label='High importance'),
    Patch(facecolor='#fff3cd', label='Medium importance'),
    Patch(facecolor='#d4edda', label='Low importance'),
]
ax.legend(handles=legend, loc='lower right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('result4_shap_importance.png', dpi=DPI)
print("✓ Saved result4_shap_importance.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  7. RESULT IMAGE 5 — Feature Correlation Heatmap (NEW)
# ═══════════════════════════════════════════════════════════════════════════
clean_names = [label_map.get(c, c) for c in X.columns]
corr_df = X.copy()
corr_df.columns = clean_names
corr = corr_df.corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f',
    cmap='RdYlBu_r', center=0, vmin=-1, vmax=1,
    linewidths=1, linecolor='white',
    cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
    ax=ax
)
ax.set_title('Feature Correlation Matrix\nUnderstanding Relationships Between Soil & Weather Features',
             fontsize=13, fontweight='bold', pad=14)
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.tick_params(axis='y', rotation=0, labelsize=10)
plt.tight_layout()
plt.savefig('result5_feature_correlation.png', dpi=DPI)
print("✓ Saved result5_feature_correlation.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  8. RESULT IMAGE 6 — Fertilizer Recommendations Table
# ═══════════════════════════════════════════════════════════════════════════
fertilizer_rules = {
    'rice':        {'Fertilizer': 'Urea + DAP',        'Reason': 'High N and P needed'},
    'maize':       {'Fertilizer': 'NPK 20:20:20',      'Reason': 'Balanced nutrients needed'},
    'chickpea':    {'Fertilizer': 'SSP + MOP',          'Reason': 'P and K boost required'},
    'kidneybeans': {'Fertilizer': 'Rhizobium + DAP',    'Reason': 'Nitrogen fixing crop'},
    'pigeonpeas':  {'Fertilizer': 'DAP',                'Reason': 'Phosphorus deficiency likely'},
    'mothbeans':   {'Fertilizer': 'Low N fertilizer',   'Reason': 'Legume — self-fixes N'},
    'mungbean':    {'Fertilizer': 'MOP',                'Reason': 'Potassium important'},
    'blackgram':   {'Fertilizer': 'SSP',                'Reason': 'Low phosphorus soil'},
    'lentil':      {'Fertilizer': 'DAP + Gypsum',       'Reason': 'Sulphur and P needed'},
    'pomegranate': {'Fertilizer': 'NPK 13:0:45',        'Reason': 'K-heavy fruit crop'},
    'banana':      {'Fertilizer': 'Urea + MOP',         'Reason': 'N and K for growth'},
    'mango':       {'Fertilizer': 'NPK 10:10:10',       'Reason': 'Balanced for fruit'},
    'grapes':      {'Fertilizer': 'Potassium Sulfate',  'Reason': 'High K for quality'},
    'watermelon':  {'Fertilizer': 'NPK 6:12:12',        'Reason': 'P+K for fruit fill'},
    'muskmelon':   {'Fertilizer': 'Urea + MOP',         'Reason': 'N and K required'},
    'apple':       {'Fertilizer': 'Calcium Nitrate',    'Reason': 'Ca for fruit firmness'},
    'orange':      {'Fertilizer': 'NPK 15:5:20',        'Reason': 'N and K dominant'},
    'papaya':      {'Fertilizer': 'Urea + SSP',         'Reason': 'N and P for growth'},
    'coconut':     {'Fertilizer': 'NPK 13:13:13',       'Reason': 'Balanced macro needs'},
    'cotton':      {'Fertilizer': 'Urea + DAP',         'Reason': 'High N demand crop'},
    'jute':        {'Fertilizer': 'Urea',               'Reason': 'Nitrogen-heavy crop'},
    'coffee':      {'Fertilizer': 'NPK 17:17:17',       'Reason': 'Balanced nutrition needed'},
}

fert_df = pd.DataFrame(fertilizer_rules).T.reset_index()
fert_df.columns = ['Crop', 'Recommended Fertilizer', 'Reason']

fig, ax = plt.subplots(figsize=(13, len(fert_df) * 0.48 + 2))
ax.axis('off')

tbl = ax.table(
    cellText=fert_df.values,
    colLabels=fert_df.columns,
    cellLoc='left', loc='center',
    colWidths=[0.2, 0.35, 0.45]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.5)

# Style header
for j in range(3):
    tbl[0, j].set_facecolor('#2c5f8a')
    tbl[0, j].set_text_props(color='white', fontweight='bold', fontsize=10)

# Alternating row colors for readability
for i in range(1, len(fert_df) + 1):
    color = '#f0f6ff' if i % 2 == 0 else 'white'
    for j in range(3):
        tbl[i, j].set_facecolor(color)
        tbl[i, j].set_edgecolor('#cccccc')

ax.set_title('Fertilizer Recommendations per Crop\n(Proposed Smart Recommendation System)',
             fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('result6_fertilizer_table.png', dpi=DPI)
print("✓ Saved result6_fertilizer_table.png")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ALL 6 RESULT IMAGES SAVED SUCCESSFULLY")
print("=" * 60)
print("  1. result1_model_comparison.png")
print("  2. result2_confusion_matrix.png")
print("  3. result3_classification_report.png")
print("  4. result4_shap_importance.png")
print("  5. result5_feature_correlation.png")
print("  6. result6_fertilizer_table.png")
print("=" * 60)
print("  Paste them into your PPT. All images are 200 DPI.")
print("=" * 60)