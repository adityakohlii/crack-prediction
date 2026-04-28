# Structural Crack Prediction ML System
## Based on Fatigue Crack Analysis — Minor Project Report

---

## Overview

This ML system converts the project report on **fatigue crack mechanics** into a
fully working, four-task prediction system grounded in:

- **Paris-Erdogan Law** → `da/dN = C·(ΔK)^m`
- **Irwin Stress Intensity Factor** → `K = α·σ·√(πa)`
- **Murakami √area Model** → `K_Imax ≈ 0.629·σ·√(π·area)`
- **ToFD / Ultrasonic Testing** → depth and height of internal cracks

---

## Four Prediction Tasks

| # | Task | Type | Target |
|---|------|------|--------|
| 1 | Crack Detection | Binary Classification | Crack Present / Absent |
| 2 | Damage Stage | Multi-class (0-4) | None / Initiation / Stable / Accel / Critical |
| 3 | Crack Length | Regression | Length in mm |
| 4 | Remaining Useful Life | Regression | Cycles to failure |

---

## Project Structure

```
crack_ml_system/
├── data/
│   ├── generate_dataset.py   ← Physics simulation (Paris Law integration)
│   └── crack_dataset.csv     ← Generated after training
├── models/
│   ├── crack_models.py       ← All 4 model definitions
│   └── saved/                ← Saved .pkl files after training
├── utils/
│   └── preprocessing.py      ← Physics feature engineering pipeline
├── plots/                    ← Generated visualizations
├── results/
│   └── metrics.json          ← Performance metrics
├── train.py                  ← Main training pipeline (RUN THIS)
├── predict.py                ← Inference module
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all models
```bash
cd crack_ml_system
python train.py
```

### 3. Predict on new specimens
```python
from predict import CrackPredictor, SpecimenData

predictor = CrackPredictor()

specimen = SpecimenData(
    material="aluminum_2024",
    panel_width_m=0.10,
    sigma_max_MPa=150,
    sigma_min_MPa=15,
    load_cycles_N=80000,
    crack_length_mm=6.0,
    ut_amplitude=0.5,
)

result = predictor.predict(specimen)
predictor.print_report(result)
```

---

## Physics Features Engineered from Report

| Feature | Source Equation |
|---------|-----------------|
| `K_max_MPa_sqrtm` | `K = α·σ·√(πa)` — Irwin (from report) |
| `delta_K_MPa_sqrtm` | `ΔK = K_max - K_min` |
| `geometry_factor_alpha` | `α = [(4+2(a/b)⁴)/(2-(a/b)²-(a/b)⁴)]^½` |
| `da_dN_current` | `da/dN = C·(ΔK)^m` — Paris-Erdogan |
| `K_murakami` | `K_Imax ≈ 0.629·σ·√(π·area)` — Murakami |
| `log_da_dN` | Log-transformed (Paris regime is log-linear) |
| `fatigue_damage_index` | `N × da/dN` — cumulative damage proxy |
| `SIF_normalized` | `K_max / K_fracture` — proximity to failure |
| `tofd_depth_mm` | `D = √((c·Δt/2)² + S·c·Δt)` — ToFD equation |

---

## Crack Stage Labels

| Stage | Label | SIF Threshold |
|-------|-------|--------------|
| 0 | No Significant Crack | K < K_threshold |
| 1 | Crack Initiation | K_norm < 0.30 |
| 2 | Stable Propagation (Paris) | K_norm < 0.60 |
| 3 | Accelerated Growth | K_norm < 0.85 |
| 4 | CRITICAL / Near-Failure | K_norm ≥ 0.85 |

---

## Advanced Upgrade Ideas (from Project Report Analysis)

1. **Physics-Informed Neural Network (PINN)** — embed Paris Law ODE as a loss term
2. **Digital Twin** — real-time crack state estimation using sensor + ML
3. **LSTM/GRU** — time-series model for sensor streaming data
4. **Remaining Useful Life** with uncertainty bounds (Bayesian prediction)
5. **IoT-SHM Dashboard** — live inspection data + crack prediction API

---

## References (from Report)

- Irwin, G.R. — Stress singularity near crack tip
- Paris, P. & Erdogan, F. — Crack growth law `da/dN = C·ΔK^m`
- Murakami, Y. — √area model for 3D cracks
- ToFD (Time of Flight Diffraction) — ultrasonic NDT for internal crack sizing
