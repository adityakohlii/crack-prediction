"""
=============================================================================
CRACK PREDICTION SYSTEM - ML Model Definitions
=============================================================================
Four prediction tasks derived from the project report:
  1. Binary Classification  → Crack Present / Not Present
  2. Multi-class           → Crack Stage (0=None, 1=Init, 2=Stable, 3=Accel, 4=Critical)
  3. Regression            → Crack Length (mm)
  4. Regression            → Remaining Useful Life (cycles)

Model: Gradient Boosted Ensemble (XGBoost + RandomForest) with physics priors
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_validate,
    RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[INFO] XGBoost not available, using GradientBoosting as fallback")


# ─────────────────────────────────────────────────────────────
# MODEL BUILDERS
# ─────────────────────────────────────────────────────────────

def build_binary_classifier():
    """
    Task 1: Binary crack detection (crack present / not present)
    Ensemble: XGBoost + RandomForest + GradientBoosting
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    
    estimators = [("rf", rf), ("gb", gb)]
    
    if XGB_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        estimators.append(("xgb", xgb))
    
    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    return ensemble


def build_stage_classifier():
    """
    Task 2: Multi-class crack stage classifier (0-4)
    Stage 0: No crack, 1: Initiation, 2: Stable, 3: Accelerated, 4: Critical
    """
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    
    estimators = [("rf", rf), ("gb", gb)]
    
    if XGB_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        estimators.append(("xgb", xgb))
    
    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    return ensemble


def build_crack_length_regressor():
    """
    Task 3: Crack length regression (mm)
    Physics-aware: crack length is bounded (0, panel_width/2)
    """
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    gb = GradientBoostingRegressor(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        loss="huber",   # robust to outliers in crack measurement
    )
    
    estimators = [("rf", rf), ("gb", gb)]
    
    if XGB_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        estimators.append(("xgb", xgb))
    
    return VotingRegressor(estimators=estimators, n_jobs=-1)


def build_rul_regressor():
    """
    Task 4: Remaining Useful Life (RUL) prediction in cycles.
    More complex target — use individual model with higher depth.
    """
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    
    if XGB_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,   # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        return VotingRegressor([("rf", rf), ("xgb", xgb)], n_jobs=-1)
    
    return rf


# ─────────────────────────────────────────────────────────────
# EVALUATION FUNCTIONS
# ─────────────────────────────────────────────────────────────

def evaluate_classifier(model, X_test, y_test, task_name="", multi_class=False):
    """Evaluate and print classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    if multi_class and y_prob is not None:
        roc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    elif y_prob is not None and not multi_class:
        roc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        roc = None
    
    print(f"\n{'='*55}")
    print(f"  {task_name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    if roc is not None:
        print(f"  ROC-AUC  : {roc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
    
    return {"accuracy": acc, "f1": f1, "roc_auc": roc}


def evaluate_regressor(model, X_test, y_test, task_name=""):
    """Evaluate and print regression metrics."""
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)   # physical constraint: non-negative
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-10))) * 100
    
    print(f"\n{'='*55}")
    print(f"  {task_name}")
    print(f"{'='*55}")
    print(f"  R²    : {r2:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  MAPE  : {mape:.2f}%")
    
    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape}


def get_feature_importance(model, feature_names):
    """Extract and return feature importances from ensemble model."""
    importances = []
    
    if hasattr(model, "estimators_"):
        # VotingClassifier / VotingRegressor
        for name, est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                importances.append(est.feature_importances_)
        if importances:
            avg_importance = np.mean(importances, axis=0)
            return pd.Series(avg_importance, index=feature_names).sort_values(ascending=False)
    
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    return pd.Series(dtype=float)
