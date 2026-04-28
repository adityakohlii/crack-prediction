"""
=============================================================================
CRACK PREDICTION SYSTEM - Main Training Pipeline
=============================================================================
Executes full training, validation, and model saving workflow.
Run: python train.py
=============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from data.generate_dataset import generate_ml_dataset
from utils.preprocessing import build_preprocessing_pipeline, prepare_data, NUMERICAL_FEATURES
from models.crack_models import (
    build_binary_classifier, build_stage_classifier,
    build_crack_length_regressor, build_rul_regressor,
    evaluate_classifier, evaluate_regressor, get_feature_importance
)

os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("models/saved", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. DATA GENERATION
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  CRACK PREDICTION ML SYSTEM — Training Pipeline")
print("  Based on: Fatigue Crack Analysis Project Report")
print("  Physics: Paris-Erdogan Law + Irwin SIF + Murakami Model")
print("="*65)

print("\n[1/6] Generating physics-based dataset...")
df = generate_ml_dataset(n_samples=4000, random_state=42)
df.to_csv("data/crack_dataset.csv", index=False)
print(f"      Dataset shape: {df.shape}")
print(f"      Crack stage distribution:\n{df['crack_stage'].value_counts().sort_index().to_string()}")

# ─────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Preprocessing and feature engineering...")

X_raw, y_binary, y_multi, y_length, y_rul = prepare_data(df)

# Train/test split using stratified split on crack stage
X_train_raw, X_test_raw, \
y_bin_train, y_bin_test, \
y_multi_train, y_multi_test, \
y_len_train, y_len_test, \
y_rul_train, y_rul_test = train_test_split(
    X_raw, y_binary, y_multi, y_length, y_rul,
    test_size=0.2,
    stratify=y_multi,
    random_state=42
)

# Fit preprocessing pipeline on training data only
preprocessor = build_preprocessing_pipeline()
X_train = preprocessor.fit_transform(X_train_raw)
X_test  = preprocessor.transform(X_test_raw)

# Get feature names after engineering
temp_df = X_raw.copy()
from utils.preprocessing import PhysicsFeatureEngineer, MaterialEncoder
pfe = PhysicsFeatureEngineer()
me = MaterialEncoder()
temp_df = pfe.fit_transform(temp_df)
temp_df = me.fit_transform(temp_df)
feature_names = [c for c in temp_df.columns if c != "crack_length_m"]

print(f"      Training samples: {X_train.shape[0]}")
print(f"      Test samples:     {X_test.shape[0]}")
print(f"      Features:         {X_train.shape[1]}")

# ─────────────────────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────────────────────
print("\n[3/6] Training all four prediction models...")

results = {}

# ── Model 1: Binary crack detection ──────────────────────────
print("\n  ▶ Training Binary Classifier (Crack Present/Absent)...")
clf_binary = build_binary_classifier()
clf_binary.fit(X_train, y_bin_train)
res1 = evaluate_classifier(clf_binary, X_test, y_bin_test,
                            "TASK 1: Binary Crack Detection")
results["binary_detection"] = res1

# ── Model 2: Crack Stage classifier ──────────────────────────
print("\n  ▶ Training Multi-class Stage Classifier...")
clf_stage = build_stage_classifier()
clf_stage.fit(X_train, y_multi_train)
res2 = evaluate_classifier(clf_stage, X_test, y_multi_test,
                            "TASK 2: Crack Stage Classification",
                            multi_class=True)
results["stage_classification"] = res2

# ── Model 3: Crack Length Regression ─────────────────────────
print("\n  ▶ Training Crack Length Regressor...")
reg_length = build_crack_length_regressor()
reg_length.fit(X_train, y_len_train)
res3 = evaluate_regressor(reg_length, X_test, y_len_test,
                           "TASK 3: Crack Length Prediction (mm)")
results["crack_length"] = res3

# ── Model 4: RUL Regression ──────────────────────────────────
print("\n  ▶ Training Remaining Useful Life Regressor...")
reg_rul = build_rul_regressor()
reg_rul.fit(X_train, y_rul_train)
res4 = evaluate_regressor(reg_rul, X_test, y_rul_test,
                           "TASK 4: Remaining Useful Life (cycles)")
results["rul_prediction"] = res4

# ─────────────────────────────────────────────────────────────
# 4. SAVE MODELS
# ─────────────────────────────────────────────────────────────
print("\n[4/6] Saving models...")
joblib.dump(preprocessor,  "models/saved/preprocessor.pkl")
joblib.dump(clf_binary,    "models/saved/clf_binary.pkl")
joblib.dump(clf_stage,     "models/saved/clf_stage.pkl")
joblib.dump(reg_length,    "models/saved/reg_length.pkl")
joblib.dump(reg_rul,       "models/saved/reg_rul.pkl")

with open("results/metrics.json", "w") as f:
    # Convert numpy floats
    clean = {k: {kk: float(vv) if vv is not None else None 
                 for kk, vv in v.items()} 
             for k, v in results.items()}
    json.dump(clean, f, indent=2)

print("      All models saved to models/saved/")

# ─────────────────────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
print("\n[5/6] Computing feature importances...")

# Use stage classifier as representative
importance = get_feature_importance(clf_stage, feature_names[:X_train.shape[1]])
if not importance.empty:
    top_features = importance.head(15)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top_features)))
    bars = ax.barh(top_features.index[::-1], top_features.values[::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance Score", fontsize=12)
    ax.set_title("Top 15 Features — Crack Stage Classifier\n(Physics-informed + NDT Features)", 
                 fontsize=13, fontweight='bold')
    ax.axvline(top_features.mean(), color='navy', linestyle='--', alpha=0.7, label='Mean importance')
    ax.legend(fontsize=10)
    
    # Annotate physics-derived features
    physics_features = ["log_da_dN", "log_delta_K", "SIF_normalized", "fatigue_damage_index",
                        "SIF_utilization", "K_murakami", "sqrt_area_mm"]
    for bar, name in zip(bars[::-1], top_features.index[::-1]):
        if name in physics_features:
            bar.set_edgecolor("darkblue")
            bar.set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("      Feature importance plot saved.")

# ─────────────────────────────────────────────────────────────
# 6. VISUALIZATION SUITE
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Generating visualizations...")

fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Confusion Matrix — Binary ──
ax1 = fig.add_subplot(gs[0, 0])
y_pred_bin = clf_binary.predict(X_test)
cm1 = confusion_matrix(y_bin_test, y_pred_bin)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=["No Crack", "Crack"],
            yticklabels=["No Crack", "Crack"])
ax1.set_title("Binary Detection\nConfusion Matrix", fontweight='bold')
ax1.set_ylabel("Actual")
ax1.set_xlabel("Predicted")

# ── Plot 2: Confusion Matrix — Stage ──
ax2 = fig.add_subplot(gs[0, 1])
y_pred_stage = clf_stage.predict(X_test)
cm2 = confusion_matrix(y_multi_test, y_pred_stage)
stage_labels = ["None", "Init", "Stable", "Accel", "Critical"]
sns.heatmap(cm2, annot=True, fmt='d', cmap='YlOrRd', ax=ax2,
            xticklabels=stage_labels, yticklabels=stage_labels)
ax2.set_title("Crack Stage Classification\nConfusion Matrix", fontweight='bold')
ax2.set_ylabel("Actual")
ax2.set_xlabel("Predicted")
plt.setp(ax2.get_xticklabels(), rotation=35, fontsize=8)
plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)

# ── Plot 3: Crack Length Prediction vs Actual ──
ax3 = fig.add_subplot(gs[0, 2])
y_pred_len = np.clip(reg_length.predict(X_test), 0, None)
idx = np.argsort(y_len_test.values)
ax3.scatter(y_len_test, y_pred_len, alpha=0.4, s=15, c='steelblue')
lim = max(y_len_test.max(), y_pred_len.max()) * 1.05
ax3.plot([0, lim], [0, lim], 'r--', linewidth=2, label="Perfect")
ax3.set_xlabel("Actual Crack Length (mm)", fontsize=10)
ax3.set_ylabel("Predicted Crack Length (mm)", fontsize=10)
ax3.set_title(f"Crack Length Prediction\nR² = {res3['r2']:.4f}", fontweight='bold')
ax3.legend(fontsize=9)

# ── Plot 4: RUL Prediction ──
ax4 = fig.add_subplot(gs[1, 0])
y_pred_rul = np.clip(reg_rul.predict(X_test), 0, None)
ax4.scatter(y_rul_test, y_pred_rul, alpha=0.4, s=15, c='darkorange')
lim_rul = max(y_rul_test.max(), y_pred_rul.max()) * 1.05
ax4.plot([0, lim_rul], [0, lim_rul], 'r--', linewidth=2, label="Perfect")
ax4.set_xlabel("Actual RUL (cycles)", fontsize=10)
ax4.set_ylabel("Predicted RUL (cycles)", fontsize=10)
ax4.set_title(f"Remaining Useful Life\nR² = {res4['r2']:.4f}", fontweight='bold')
ax4.legend(fontsize=9)

# ── Plot 5: Paris Law Crack Growth Simulation ──
ax5 = fig.add_subplot(gs[1, 1])
from data.generate_dataset import simulate_crack_growth
sim = simulate_crack_growth("aluminum_2024", sigma_max=150, sigma_min=30,
                             a_initial=1e-3, n_cycles=150000)
ax5.plot(sim["cycles"], sim["crack_length"] * 1000, 'b-', linewidth=2, label="a(N)")
ax5.set_xlabel("Load Cycles N", fontsize=10)
ax5.set_ylabel("Crack Length (mm)", fontsize=10)
ax5.set_title("Paris Law Crack Growth\nAl-2024, σ_max=150MPa", fontweight='bold')
ax5.axhline(y=sim["crack_length"][-1] * 1000, color='red', linestyle='--', 
            alpha=0.7, label='Near-Failure')
ax5.legend(fontsize=9)

# ── Plot 6: da/dN vs ΔK (Paris Law Regime) ──
ax6 = fig.add_subplot(gs[1, 2])
delta_K_range = np.linspace(2, 40, 200)
C, m = 3.6e-10, 2.9
da_dN_paris = C * delta_K_range**m
ax6.loglog(delta_K_range, da_dN_paris, 'b-', linewidth=2.5, label="Paris Law")
ax6.axvline(x=3.0, color='green', linestyle='--', label="K_threshold")
ax6.axvline(x=33.0, color='red', linestyle='--', label="K_fracture")
ax6.fill_betweenx([1e-14, 1e-6], 3.0, 33.0, alpha=0.1, color='blue', label="Paris Regime B")
ax6.set_xlabel("ΔK (MPa·√m)", fontsize=10)
ax6.set_ylabel("da/dN (m/cycle)", fontsize=10)
ax6.set_title("Paris-Erdogan Law\nda/dN = C·ΔK^m (Al-2024)", fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, which='both', alpha=0.3)

# ── Plot 7: SIF Distribution by Stage ──
ax7 = fig.add_subplot(gs[2, 0])
stage_colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
stage_labels_full = ["Stage 0\n(None)", "Stage 1\n(Init)", "Stage 2\n(Stable)",
                     "Stage 3\n(Accel)", "Stage 4\n(Critical)"]
for stage in sorted(df["crack_stage"].unique()):
    subset = df[df["crack_stage"] == stage]["K_max_MPa_sqrtm"]
    ax7.hist(subset, bins=30, alpha=0.6, label=f"Stage {stage}",
             color=stage_colors[stage], density=True)
ax7.set_xlabel("K_max (MPa·√m)", fontsize=10)
ax7.set_ylabel("Density", fontsize=10)
ax7.set_title("SIF Distribution\nby Crack Stage", fontweight='bold')
ax7.legend(fontsize=8)

# ── Plot 8: Crack Stage Pie Chart ──
ax8 = fig.add_subplot(gs[2, 1])
stage_counts = df["crack_stage"].value_counts().sort_index()
ax8.pie(stage_counts.values, labels=stage_labels_full, colors=stage_colors,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
ax8.set_title("Dataset Composition\nby Crack Stage", fontweight='bold')

# ── Plot 9: Model Performance Summary ──
ax9 = fig.add_subplot(gs[2, 2])
tasks = ["Binary\nDetect", "Stage\nClassif", "Crack\nLength\n(R²)", "RUL\n(R²)"]
scores = [
    res1["accuracy"],
    res2["accuracy"],
    res3["r2"],
    res4["r2"],
]
bar_colors = ['#27ae60' if s > 0.9 else '#f39c12' if s > 0.75 else '#e74c3c' for s in scores]
bars = ax9.bar(tasks, scores, color=bar_colors, edgecolor='black', linewidth=0.5)
ax9.set_ylim(0, 1.1)
ax9.axhline(y=0.9, color='green', linestyle='--', alpha=0.6, label='90% threshold')
for bar, score in zip(bars, scores):
    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax9.set_ylabel("Score", fontsize=10)
ax9.set_title("Model Performance Summary\n(All Four Tasks)", fontweight='bold')
ax9.legend(fontsize=9)

fig.suptitle("STRUCTURAL CRACK PREDICTION — ML SYSTEM\nBased on Fatigue Mechanics (Paris Law + Irwin SIF + Murakami √area)",
             fontsize=14, fontweight='bold', y=1.01)

plt.savefig("plots/complete_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("      Complete dashboard saved to plots/complete_dashboard.png")

# ─────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  TRAINING COMPLETE — RESULTS SUMMARY")
print("="*65)
print(f"\n  Task 1 — Binary Crack Detection:")
print(f"    Accuracy: {res1['accuracy']:.4f}  |  F1: {res1['f1']:.4f}  |  AUC: {res1['roc_auc']:.4f}")
print(f"\n  Task 2 — Crack Stage Classification (5 classes):")
print(f"    Accuracy: {res2['accuracy']:.4f}  |  F1: {res2['f1']:.4f}")
print(f"\n  Task 3 — Crack Length Regression:")
print(f"    R²: {res3['r2']:.4f}  |  RMSE: {res3['rmse']:.4f}mm  |  MAE: {res3['mae']:.4f}mm")
print(f"\n  Task 4 — Remaining Useful Life:")
print(f"    R²: {res4['r2']:.4f}  |  RMSE: {res4['rmse']:.0f} cycles")
print("\n  Files saved:")
print("    models/saved/*.pkl     — Trained models")
print("    results/metrics.json   — All metrics")
print("    plots/*.png            — Visualizations")
print("    data/crack_dataset.csv — Synthetic dataset")
print("="*65)
