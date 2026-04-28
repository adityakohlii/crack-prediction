"""
=============================================================================
CRACK PREDICTION SYSTEM - Dataset Generator
=============================================================================
Based on: Minor Project Report - Fatigue Crack Analysis
Physics: Paris-Erdogan Law, Irwin Stress Intensity Factor, Murakami √area Model
Author: ML System (Generated from Project Report Analysis)
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────
# PHYSICS CONSTANTS FROM REPORT
# ─────────────────────────────────────────────────
# Paris Law: da/dN = C * (ΔK)^m
# K = α * σ * √a   (Stress Intensity Factor)
# Murakami: K_Imax ≈ 0.629 * σ * √(π * area)
# ToFD Depth: D = sqrt((c*Δt/2)^2 + S*c*Δt)

MATERIALS = {
    "aluminum_2024": {
        "C": 3.6e-10,   # Paris law coefficient (m/cycle, MPa√m)
        "m": 2.9,        # Paris law exponent
        "K_threshold": 3.0,   # MPa√m  
        "K_fracture": 33.0,   # MPa√m (fracture toughness)
        "E": 73.0,       # GPa (elastic modulus)
        "yield": 345.0,  # MPa (yield strength)
        "density": 2780, # kg/m³
    },
    "steel_4340": {
        "C": 1.0e-11,
        "m": 3.1,
        "K_threshold": 5.0,
        "K_fracture": 60.0,
        "E": 210.0,
        "yield": 1170.0,
        "density": 7850,
    },
    "titanium_6al4v": {
        "C": 5.0e-12,
        "m": 3.2,
        "K_threshold": 4.0,
        "K_fracture": 55.0,
        "E": 114.0,
        "yield": 950.0,
        "density": 4430,
    },
}

def geometry_factor_alpha(a, b):
    """
    Geometry correction factor α from the project report.
    α = [(4 + 2(a/b)^4) / (2 - (a/b)^2 - (a/b)^4)]^(1/2)
    For a center-cracked panel of width 2b.
    """
    ratio = a / b
    numerator = 4 + 2 * ratio**4
    denominator = 2 - ratio**2 - ratio**4
    denominator = max(denominator, 1e-10)  # prevent division by zero
    return np.sqrt(numerator / denominator)

def stress_intensity_factor(sigma, a, b=0.1):
    """
    K = α * σ * √a   (Irwin/Paris formulation from report)
    sigma: applied stress [MPa]
    a: half-crack length [m]
    b: half-panel width [m]
    """
    alpha = geometry_factor_alpha(a, b)
    return alpha * sigma * np.sqrt(np.pi * a)

def murakami_K_max(sigma, area):
    """
    Murakami √area model from report:
    K_Imax ≈ 0.629 * σ * √(π * area)
    """
    return 0.629 * sigma * np.sqrt(np.pi * area)

def paris_law_growth(a, N, C, m, sigma_max, sigma_min, b=0.1):
    """
    Paris-Erdogan Law (da/dN) from report:
    da/dN = C * (ΔK)^m
    ΔK = K_max - K_min
    """
    K_max = stress_intensity_factor(sigma_max, a, b)
    K_min = stress_intensity_factor(sigma_min, a, b)
    delta_K = K_max - K_min
    if delta_K <= 0:
        return 0.0
    return C * delta_K**m

def tofd_depth(t1, tL, c_sound=5920, S=0.05):
    """
    ToFD (Time of Flight Diffraction) depth calculation from report:
    D = sqrt((c*(t1-tL)/2)^2 + S*c*(t1-tL))
    c_sound: speed of sound in material [m/s] (steel default)
    S: transducer separation [m]
    """
    dt = t1 - tL
    if dt <= 0:
        return 0.0
    term1 = (c_sound * dt / 2)**2
    term2 = S * c_sound * dt
    return np.sqrt(term1 + term2)

def simulate_crack_growth(material_name, sigma_max, sigma_min, a_initial,
                           panel_width=0.1, n_cycles=100000, cycle_step=500):
    """
    Simulate full crack propagation history using Paris law integration.
    Returns cycle history, crack length, K values, growth rates.
    """
    mat = MATERIALS[material_name]
    C, m = mat["C"], mat["m"]
    K_fracture = mat["K_fracture"]
    
    cycles = []
    crack_lengths = []
    K_values = []
    growth_rates = []
    beta_ratios = []
    
    a = a_initial
    N = 0
    b = panel_width / 2
    
    while N <= n_cycles:
        K_max = stress_intensity_factor(sigma_max, a, b)
        K_min = stress_intensity_factor(sigma_min, a, b)
        delta_K = K_max - K_min
        
        if K_max >= K_fracture or a >= b * 0.9:
            break  # fracture
        
        da_dN = C * max(delta_K, 0)**m
        beta = sigma_max / max(sigma_min, 0.01)
        
        cycles.append(N)
        crack_lengths.append(a)
        K_values.append(K_max)
        growth_rates.append(da_dN)
        beta_ratios.append(beta)
        
        # Euler integration
        a += da_dN * cycle_step
        N += cycle_step
    
    return {
        "cycles": np.array(cycles),
        "crack_length": np.array(crack_lengths),
        "K_max": np.array(K_values),
        "growth_rate": np.array(growth_rates),
        "beta": np.array(beta_ratios),
    }

def generate_ml_dataset(n_samples=5000, random_state=42):
    """
    Generate a realistic ML-ready dataset from physics simulation.
    Each sample = one inspection snapshot of a structural specimen.
    Features are based on report variables and NDT measurements.
    """
    np.random.seed(random_state)
    
    records = []
    material_names = list(MATERIALS.keys())
    
    for i in range(n_samples):
        # ── Sample random specimen parameters ──
        mat_name = np.random.choice(material_names)
        mat = MATERIALS[mat_name]
        
        sigma_max = np.random.uniform(50, min(0.7 * mat["yield"], 500))  # MPa
        stress_ratio = np.random.uniform(0.05, 0.5)
        sigma_min = sigma_max * stress_ratio
        delta_sigma = sigma_max - sigma_min   # stress range
        
        panel_width = np.random.uniform(0.05, 0.30)  # m (5cm - 30cm)
        b = panel_width / 2
        
        a_initial = np.random.uniform(0.5e-3, 5e-3)  # 0.5mm to 5mm
        N_current = np.random.randint(1000, 200000)   # current cycle count
        cycle_step = 200
        
        # ── Simulate crack growth to current cycle ──
        mat_C = mat["C"]
        mat_m = mat["m"]
        a = a_initial
        for _ in range(0, N_current, cycle_step):
            K_max = stress_intensity_factor(sigma_max, a, b)
            if K_max >= mat["K_fracture"] or a >= b * 0.9:
                break
            delta_K = K_max - stress_intensity_factor(sigma_min, a, b)
            da_dN = mat_C * max(delta_K, 0)**mat_m
            a += da_dN * cycle_step
        
        a_current = a
        K_current = stress_intensity_factor(sigma_max, a_current, b)
        delta_K_current = K_current - stress_intensity_factor(sigma_min, a_current, b)
        da_dN_current = mat_C * max(delta_K_current, 0)**mat_m
        
        # ── Murakami area model feature ──
        crack_area = np.pi * a_current**2  # approximate circular crack area
        K_murakami = murakami_K_max(sigma_max, crack_area)
        
        # ── Simulated ToFD measurement (with noise) ──
        c_sound = 5920 if "steel" in mat_name else (6420 if "aluminum" in mat_name else 6100)
        S_trans = np.random.uniform(0.03, 0.08)
        true_depth = a_current + np.random.uniform(-0.1e-3, 0.1e-3)
        # Reverse-calculate t1 from depth
        # D ≈ c*(t1-tL)/2  →  t1-tL ≈ 2D/c
        dt_signal = 2 * true_depth / c_sound + np.random.normal(0, 1e-8)
        tL_signal = 1e-6
        t1_signal = tL_signal + max(dt_signal, 0)
        tofd_measured = tofd_depth(t1_signal, tL_signal, c_sound, S_trans)
        
        # ── UT amplitude (proxy for defect severity) ──
        ut_amplitude = min(1.0, (a_current / (b * 0.5)) + np.random.normal(0, 0.05))
        ut_amplitude = max(0.0, ut_amplitude)
        
        # ── Derived physics features ──
        alpha_geom = geometry_factor_alpha(a_current, b)
        beta_ratio = sigma_max / max(sigma_min, 0.01)
        SIF_normalized = K_current / mat["K_fracture"]
        crack_growth_index = da_dN_current * N_current  # total accumulated growth
        
        # ── Crack stage classification (labels) ──
        # Based on SIF ratio and crack length
        if K_current < mat["K_threshold"]:
            crack_stage = 0  # No significant crack / sub-threshold
        elif SIF_normalized < 0.3:
            crack_stage = 1  # Crack initiation
        elif SIF_normalized < 0.6:
            crack_stage = 2  # Stable propagation (Paris regime)
        elif SIF_normalized < 0.85:
            crack_stage = 3  # Accelerated growth
        else:
            crack_stage = 4  # Critical / near-failure
        
        crack_present = int(crack_stage > 0)
        
        # Remaining cycles to failure estimate
        # Integrate da/dN from current a to critical a
        a_critical = b * 0.9
        if a_current < a_critical and da_dN_current > 0:
            rul_approx = (a_critical - a_current) / (da_dN_current * cycle_step) * cycle_step
        else:
            rul_approx = 0
        rul_approx = max(rul_approx, 0)
        
        # ── Assemble record ──
        record = {
            # Material / geometry
            "material": mat_name,
            "material_E_GPa": mat["E"],
            "material_yield_MPa": mat["yield"],
            "panel_width_m": panel_width,
            
            # Loading conditions
            "sigma_max_MPa": sigma_max,
            "sigma_min_MPa": sigma_min,
            "delta_sigma_MPa": delta_sigma,
            "stress_ratio_R": stress_ratio,
            "load_cycles_N": N_current,
            "beta_ratio": beta_ratio,
            
            # Crack geometry
            "crack_length_m": a_current,
            "crack_length_mm": a_current * 1000,
            "crack_area_m2": crack_area,
            "initial_crack_m": a_initial,
            "crack_growth_from_initial": a_current - a_initial,
            
            # Fracture mechanics features (from report equations)
            "K_max_MPa_sqrtm": K_current,
            "K_min_MPa_sqrtm": stress_intensity_factor(sigma_min, a_current, b),
            "delta_K_MPa_sqrtm": delta_K_current,
            "geometry_factor_alpha": alpha_geom,
            "SIF_normalized": SIF_normalized,
            "K_murakami": K_murakami,
            
            # Paris law features
            "da_dN_current": da_dN_current,
            "paris_C": mat_C,
            "paris_m": mat_m,
            "crack_growth_index": crack_growth_index,
            
            # NDT / ToFD features
            "tofd_depth_m": tofd_measured,
            "tofd_depth_mm": tofd_measured * 1000,
            "ut_amplitude": ut_amplitude,
            "tofd_t1_us": t1_signal * 1e6,
            "tofd_tL_us": tL_signal * 1e6,
            
            # TARGET LABELS
            "crack_present": crack_present,          # Binary: 0/1
            "crack_stage": crack_stage,              # Multi-class: 0-4
            "crack_length_label": a_current * 1000,  # Regression (mm)
            "RUL_cycles": rul_approx,                # Remaining Useful Life
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    print("Generating physics-based crack dataset...")
    df = generate_ml_dataset(n_samples=5000)
    df.to_csv("data/crack_dataset.csv", index=False)
    print(f"Dataset shape: {df.shape}")
    print(f"\nCrack stage distribution:\n{df['crack_stage'].value_counts().sort_index()}")
    print(f"\nCrack present (%):\n{df['crack_present'].value_counts(normalize=True)*100}")
    print(f"\nSample features:\n{df.describe().T[['mean','std','min','max']].round(4)}")
