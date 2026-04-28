"""
=============================================================================
CRACK PREDICTION SYSTEM - Preprocessing & Feature Engineering
=============================================================================
Physics-informed feature engineering from Paris Law + Fracture Mechanics
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import joblib
import os

class PhysicsFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Physics-informed feature engineering grounded in the project report:
    - Paris Law interaction terms
    - Stress Intensity Factor ratios
    - Log-transformed crack growth rate
    - SIF-based fatigue damage indicators
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # ── Paris Law: log(da/dN) is linear with log(ΔK) ──
        # This is the fundamental Paris regime relationship
        if "da_dN_current" in X.columns:
            X["log_da_dN"] = np.log1p(X["da_dN_current"])
        
        if "delta_K_MPa_sqrtm" in X.columns:
            X["log_delta_K"] = np.log1p(np.abs(X["delta_K_MPa_sqrtm"]))
        
        # ── Stress Intensity Factor utilization ratio ──
        # How close to fracture toughness
        if "K_max_MPa_sqrtm" in X.columns and "material_yield_MPa" in X.columns:
            # Proxy fracture toughness from yield strength (Irwin relation)
            X["K_Ic_proxy"] = 0.1 * X["material_yield_MPa"]  # rough estimate
            X["SIF_utilization"] = X["K_max_MPa_sqrtm"] / (X["K_Ic_proxy"] + 1e-10)
        
        # ── Crack length / panel width ratio (report geometry factor) ──
        if "crack_length_m" in X.columns and "panel_width_m" in X.columns:
            X["crack_width_ratio"] = X["crack_length_m"] / (X["panel_width_m"] + 1e-10)
        
        # ── Murakami √area feature ──
        if "crack_area_m2" in X.columns:
            X["sqrt_area_mm"] = np.sqrt(X["crack_area_m2"]) * 1000  # mm^0.5
        
        # ── Fatigue damage index (cycles * crack growth per cycle) ──
        if "load_cycles_N" in X.columns and "da_dN_current" in X.columns:
            X["fatigue_damage_index"] = np.log1p(
                X["load_cycles_N"] * X["da_dN_current"]
            )
        
        # ── Loading severity feature ──
        if "delta_sigma_MPa" in X.columns and "material_yield_MPa" in X.columns:
            X["load_severity"] = X["delta_sigma_MPa"] / (X["material_yield_MPa"] + 1e-10)
        
        # ── ToFD signal quality feature ──
        if "tofd_depth_mm" in X.columns and "crack_length_mm" in X.columns:
            X["tofd_crack_agreement"] = np.abs(
                X["tofd_depth_mm"] - X["crack_length_mm"]
            )
        
        # ── SIF range (ΔK) normalized by material threshold ──
        if "delta_K_MPa_sqrtm" in X.columns:
            X["delta_K_normalized"] = X["delta_K_MPa_sqrtm"] / 5.0  # ~K_threshold
        
        # ── Cyclic load ratio feature ──
        if "sigma_max_MPa" in X.columns and "sigma_min_MPa" in X.columns:
            X["cyclic_plasticity"] = (
                X["sigma_max_MPa"] - X["sigma_min_MPa"]
            ) / (X["sigma_max_MPa"] + 1e-10)
        
        return X


class MaterialEncoder(BaseEstimator, TransformerMixin):
    """
    Encode material type as ordinal feature based on fracture toughness ordering.
    """
    MATERIAL_ORDER = {
        "aluminum_2024": 0,
        "titanium_6al4v": 1,
        "steel_4340": 2,
    }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if "material" in X.columns:
            X["material_encoded"] = X["material"].map(self.MATERIAL_ORDER).fillna(0)
            X = X.drop(columns=["material"])
        return X


NUMERICAL_FEATURES = [
    "material_E_GPa",
    "material_yield_MPa",
    "panel_width_m",
    "sigma_max_MPa",
    "sigma_min_MPa",
    "delta_sigma_MPa",
    "stress_ratio_R",
    "load_cycles_N",
    "beta_ratio",
    "crack_length_mm",
    "crack_area_m2",
    "initial_crack_m",
    "crack_growth_from_initial",
    "K_max_MPa_sqrtm",
    "K_min_MPa_sqrtm",
    "delta_K_MPa_sqrtm",
    "geometry_factor_alpha",
    "SIF_normalized",
    "K_murakami",
    "da_dN_current",
    "crack_growth_index",
    "tofd_depth_mm",
    "ut_amplitude",
    "tofd_t1_us",
    "material_encoded",
    # Engineered
    "log_da_dN",
    "log_delta_K",
    "SIF_utilization",
    "crack_width_ratio",
    "sqrt_area_mm",
    "fatigue_damage_index",
    "load_severity",
    "tofd_crack_agreement",
    "delta_K_normalized",
    "cyclic_plasticity",
]

TARGET_CLASSIFICATION_BINARY = "crack_present"
TARGET_CLASSIFICATION_MULTI  = "crack_stage"
TARGET_REGRESSION_LENGTH     = "crack_length_label"
TARGET_REGRESSION_RUL        = "RUL_cycles"


def build_preprocessing_pipeline():
    """Build scikit-learn preprocessing pipeline."""
    pipe = Pipeline([
        ("physics_features", PhysicsFeatureEngineer()),
        ("material_encode", MaterialEncoder()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),   # robust to outliers in crack data
    ])
    return pipe


def prepare_data(df):
    """
    Full data preparation: split features/targets, run preprocessing.
    Returns preprocessed X array + all target series.
    """
    # Drop target columns and non-feature columns from X
    drop_cols = [
        TARGET_CLASSIFICATION_BINARY,
        TARGET_CLASSIFICATION_MULTI,
        TARGET_REGRESSION_LENGTH,
        TARGET_REGRESSION_RUL,
        "crack_length_m",      # raw version (mm version is kept)
        "paris_C",
        "paris_m",
        "tofd_t1_us",          # keep processed version
        "tofd_tL_us",
    ]
    
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    y_binary  = df[TARGET_CLASSIFICATION_BINARY]
    y_multi   = df[TARGET_CLASSIFICATION_MULTI]
    y_length  = df[TARGET_REGRESSION_LENGTH]
    y_rul     = df[TARGET_REGRESSION_RUL]
    
    return X_raw, y_binary, y_multi, y_length, y_rul


if __name__ == "__main__":
    from generate_dataset import generate_ml_dataset
    df = generate_ml_dataset(500)
    X_raw, y_bin, y_multi, y_len, y_rul = prepare_data(df)
    pipe = build_preprocessing_pipeline()
    X_processed = pipe.fit_transform(X_raw)
    print(f"Input shape:  {X_raw.shape}")
    print(f"Output shape: {X_processed.shape}")
