"""
=============================================================================
CRACK PREDICTION SYSTEM - Inference / Prediction Module
=============================================================================
Use this module for real-time predictions on new specimens.
Usage:
    from predict import CrackPredictor
    predictor = CrackPredictor()
    result = predictor.predict(specimen_data)
=============================================================================
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
from dataclasses import dataclass, field
from typing import Optional

STAGE_LABELS = {
    0: "No Significant Crack",
    1: "Crack Initiation",
    2: "Stable Propagation (Paris Regime)",
    3: "Accelerated Growth",
    4: "CRITICAL — Near Failure",
}

STAGE_COLORS = {0: "green", 1: "yellow", 2: "orange", 3: "darkorange", 4: "red"}

RECOMMENDATIONS = {
    0: "Structure is safe. Continue routine inspection per scheduled intervals.",
    1: "Crack initiated. Increase inspection frequency. Monitor crack length trend.",
    2: "Crack growing in Paris regime. Schedule maintenance in next window.",
    3: "Accelerated crack growth detected. Reduce load. Inspect within 500 cycles.",
    4: "CRITICAL: Immediate inspection required. Risk of catastrophic failure.",
}


@dataclass
class SpecimenData:
    """
    Input data for a structural specimen inspection.
    All fields correspond to features from the project report.
    """
    # Material
    material: str = "aluminum_2024"  # aluminum_2024, steel_4340, titanium_6al4v
    material_E_GPa: float = 73.0
    material_yield_MPa: float = 345.0
    
    # Geometry
    panel_width_m: float = 0.10
    
    # Loading
    sigma_max_MPa: float = 100.0
    sigma_min_MPa: float = 10.0
    load_cycles_N: int = 50000
    
    # Crack geometry
    crack_length_mm: float = 2.0
    initial_crack_m: float = 0.001
    crack_area_m2: Optional[float] = None
    
    # Fracture mechanics (can be auto-computed)
    K_max_MPa_sqrtm: Optional[float] = None
    K_min_MPa_sqrtm: Optional[float] = None
    delta_K_MPa_sqrtm: Optional[float] = None
    geometry_factor_alpha: Optional[float] = None
    SIF_normalized: Optional[float] = None
    K_murakami: Optional[float] = None
    
    # Paris law
    da_dN_current: Optional[float] = None
    crack_growth_index: Optional[float] = None
    
    # NDT (ToFD measurements)
    tofd_depth_mm: Optional[float] = None
    ut_amplitude: float = 0.3
    tofd_t1_us: float = 1.5
    tofd_tL_us: float = 1.0
    
    def auto_compute_physics(self):
        """Auto-compute physics features from basic inputs using report equations."""
        from data.generate_dataset import (
            stress_intensity_factor, geometry_factor_alpha,
            murakami_K_max, paris_law_growth, MATERIALS
        )
        
        a = self.crack_length_mm * 1e-3
        b = self.panel_width_m / 2
        mat = MATERIALS.get(self.material, MATERIALS["aluminum_2024"])
        
        self.crack_area_m2 = self.crack_area_m2 or np.pi * a**2
        self.geometry_factor_alpha = geometry_factor_alpha(a, b)
        self.K_max_MPa_sqrtm = stress_intensity_factor(self.sigma_max_MPa, a, b)
        self.K_min_MPa_sqrtm = stress_intensity_factor(self.sigma_min_MPa, a, b)
        self.delta_K_MPa_sqrtm = self.K_max_MPa_sqrtm - self.K_min_MPa_sqrtm
        self.SIF_normalized = self.K_max_MPa_sqrtm / mat["K_fracture"]
        self.K_murakami = murakami_K_max(self.sigma_max_MPa, self.crack_area_m2)
        
        C, m_paris = mat["C"], mat["m"]
        self.da_dN_current = C * max(self.delta_K_MPa_sqrtm, 0)**m_paris
        self.crack_growth_index = self.da_dN_current * self.load_cycles_N
        
        if self.tofd_depth_mm is None:
            self.tofd_depth_mm = self.crack_length_mm + np.random.normal(0, 0.1)
        
        return self
    
    def to_dataframe(self):
        """Convert to pandas DataFrame for ML pipeline."""
        self.auto_compute_physics()
        
        record = {
            "material": self.material,
            "material_E_GPa": self.material_E_GPa,
            "material_yield_MPa": self.material_yield_MPa,
            "panel_width_m": self.panel_width_m,
            "sigma_max_MPa": self.sigma_max_MPa,
            "sigma_min_MPa": self.sigma_min_MPa,
            "delta_sigma_MPa": self.sigma_max_MPa - self.sigma_min_MPa,
            "stress_ratio_R": self.sigma_min_MPa / max(self.sigma_max_MPa, 1e-10),
            "load_cycles_N": self.load_cycles_N,
            "beta_ratio": self.sigma_max_MPa / max(self.sigma_min_MPa, 0.01),
            "crack_length_mm": self.crack_length_mm,
            "crack_area_m2": self.crack_area_m2,
            "initial_crack_m": self.initial_crack_m,
            "crack_growth_from_initial": (self.crack_length_mm * 1e-3) - self.initial_crack_m,
            "K_max_MPa_sqrtm": self.K_max_MPa_sqrtm,
            "K_min_MPa_sqrtm": self.K_min_MPa_sqrtm,
            "delta_K_MPa_sqrtm": self.delta_K_MPa_sqrtm,
            "geometry_factor_alpha": self.geometry_factor_alpha,
            "SIF_normalized": self.SIF_normalized,
            "K_murakami": self.K_murakami,
            "da_dN_current": self.da_dN_current,
            "crack_growth_index": self.crack_growth_index,
            "tofd_depth_mm": self.tofd_depth_mm,
            "ut_amplitude": self.ut_amplitude,
            "tofd_t1_us": self.tofd_t1_us,
        }
        return pd.DataFrame([record])


class CrackPredictor:
    """
    Production-ready inference class for structural crack prediction.
    Loads all four trained models and provides a unified predict() method.
    """
    
    def __init__(self, model_dir="models/saved"):
        self.model_dir = model_dir
        self._load_models()
    
    def _load_models(self):
        try:
            self.preprocessor  = joblib.load(f"{self.model_dir}/preprocessor.pkl")
            self.clf_binary    = joblib.load(f"{self.model_dir}/clf_binary.pkl")
            self.clf_stage     = joblib.load(f"{self.model_dir}/clf_stage.pkl")
            self.reg_length    = joblib.load(f"{self.model_dir}/reg_length.pkl")
            self.reg_rul       = joblib.load(f"{self.model_dir}/reg_rul.pkl")
            print("[CrackPredictor] All models loaded successfully.")
        except FileNotFoundError:
            print("[CrackPredictor] Models not found. Run train.py first.")
            raise
    
    def predict(self, specimen: SpecimenData) -> dict:
        """
        Full prediction pipeline for a single specimen.
        Returns:
            dict with crack_present, crack_stage, crack_length_mm, rul_cycles,
                  confidence, recommendation, risk_level
        """
        df_input = specimen.to_dataframe()
        X = self.preprocessor.transform(df_input)
        
        # ── Task 1: Binary Detection ──
        crack_present = bool(self.clf_binary.predict(X)[0])
        crack_prob = float(self.clf_binary.predict_proba(X)[0][1])
        
        # ── Task 2: Stage Classification ──
        stage = int(self.clf_stage.predict(X)[0])
        stage_probs = self.clf_stage.predict_proba(X)[0].tolist()
        
        # ── Task 3: Crack Length ──
        crack_length_pred = float(max(0, self.reg_length.predict(X)[0]))
        
        # ── Task 4: RUL ──
        rul_pred = float(max(0, self.reg_rul.predict(X)[0]))
        
        # ── Risk Assessment ──
        if stage >= 4:
            risk = "CRITICAL"
        elif stage == 3:
            risk = "HIGH"
        elif stage == 2:
            risk = "MODERATE"
        elif stage == 1:
            risk = "LOW"
        else:
            risk = "SAFE"
        
        return {
            "crack_present": crack_present,
            "crack_probability": round(crack_prob, 4),
            "crack_stage": stage,
            "crack_stage_label": STAGE_LABELS[stage],
            "crack_stage_probabilities": {i: round(p, 4) for i, p in enumerate(stage_probs)},
            "predicted_crack_length_mm": round(crack_length_pred, 4),
            "remaining_useful_life_cycles": int(rul_pred),
            "risk_level": risk,
            "recommendation": RECOMMENDATIONS[stage],
            "physics_features": {
                "K_max_MPa_sqrtm": round(specimen.K_max_MPa_sqrtm, 3),
                "delta_K_MPa_sqrtm": round(specimen.delta_K_MPa_sqrtm, 3),
                "SIF_normalized": round(specimen.SIF_normalized, 4),
                "da_dN": f"{specimen.da_dN_current:.3e}",
            }
        }
    
    def batch_predict(self, specimens: list) -> list:
        """Predict for a list of SpecimenData objects."""
        return [self.predict(s) for s in specimens]
    
    def print_report(self, result: dict):
        """Pretty-print prediction result."""
        sep = "="*60
        print(f"\n{sep}")
        print("  STRUCTURAL CRACK PREDICTION REPORT")
        print(sep)
        print(f"  Crack Present   : {'YES ⚠️' if result['crack_present'] else 'NO ✓'}")
        print(f"  Crack Probability: {result['crack_probability']*100:.1f}%")
        print(f"  Stage           : {result['crack_stage']} — {result['crack_stage_label']}")
        print(f"  Risk Level      : {result['risk_level']}")
        print(f"  Crack Length    : {result['predicted_crack_length_mm']:.3f} mm")
        print(f"  RUL             : {result['remaining_useful_life_cycles']:,} cycles")
        print(f"\n  Physics:")
        for k, v in result['physics_features'].items():
            print(f"    {k}: {v}")
        print(f"\n  Action Required:")
        print(f"  → {result['recommendation']}")
        print(sep)


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = CrackPredictor()
    
    # Test Case 1: Healthy aluminum panel
    specimen1 = SpecimenData(
        material="aluminum_2024",
        panel_width_m=0.10,
        sigma_max_MPa=80,
        sigma_min_MPa=8,
        load_cycles_N=10000,
        crack_length_mm=0.8,
        ut_amplitude=0.05,
    )
    result1 = predictor.predict(specimen1)
    print("\n--- TEST CASE 1: Healthy Panel ---")
    predictor.print_report(result1)
    
    # Test Case 2: Critically cracked steel beam
    specimen2 = SpecimenData(
        material="steel_4340",
        panel_width_m=0.20,
        sigma_max_MPa=400,
        sigma_min_MPa=40,
        load_cycles_N=150000,
        crack_length_mm=18.0,
        ut_amplitude=0.85,
    )
    result2 = predictor.predict(specimen2)
    print("\n--- TEST CASE 2: Critical Crack ---")
    predictor.print_report(result2)
    
    # Test Case 3: Titanium aerospace component
    specimen3 = SpecimenData(
        material="titanium_6al4v",
        panel_width_m=0.08,
        sigma_max_MPa=300,
        sigma_min_MPa=30,
        load_cycles_N=75000,
        crack_length_mm=5.5,
        ut_amplitude=0.45,
    )
    result3 = predictor.predict(specimen3)
    print("\n--- TEST CASE 3: Titanium Aerospace Component ---")
    predictor.print_report(result3)
