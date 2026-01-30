# VER 2.0 - ML-Based Peak Detection for CV Analysis

## Overview

**VER 2.0** is a machine learning pipeline for automatic peak detection and deconvolution in Cyclic Voltammetry (CV) data. It uses a 1D U-Net to detect peak locations, then fits Gaussian curves to extract quantitative parameters.

### Key Features
- ✅ **Automatic peak detection** - no manual peak picking
- ✅ **Multi-peak capability** - handles 1-3 overlapping peaks
- ✅ **High accuracy** - 97.5% peak count detection, ±53mV position error
- ✅ **Full parameter extraction** - center, height, width, area (for concentration)
- ✅ **Trained on realistic data** - MechanisticSimulator generates CV-shaped curves

### Validation Results (9×9 Grid = 81 Test Cases)
```
Overall Peak Count Accuracy: 79/81 (97.5%)

By Complexity:
  1 peak:  27/27 (100.0%)
  2 peaks: 25/27 (92.6%)  ← minor issues with very close peaks
  3 peaks: 27/27 (100.0%)

Position Accuracy (when count correct):
  Mean error: 53.0 mV
  Std error:  22.2 mV
  Max error:  101.8 mV
```

---

## Quick Start

### 1. Analyze a CV File
```bash
source ../venv/bin/activate
cd VER_2.0
python3 analyze_real_cv.py
```

This will process `data/PureHydration_9Jan.csv` and output:
- Detected peak count
- Peak positions (V)
- Peak heights (µA)
- Peak areas (for concentration calculation)
- Visualization saved to `real_cv_analysis.png`

### 2. Run Validation Demo
```bash
python3 validate_pipeline.py
```

Generates 9×9 grid of test cases and reports accuracy metrics.

---

## Project Structure

```
VER_2.0/
├── models/                          # Trained models
│   ├── peak_detector_cv.keras       # U-Net model (20k training samples)
│   └── normalization_params_cv.npz  # Signal normalization constants
│
├── src/                             # Source code
│   ├── generator_cv.py              # Training data generator
│   ├── gaussian_fitter.py           # Peak fitting module
│   ├── predict.py                   # Inference wrapper
│   └── cv_simulator.py              # MechanisticSimulator (VER 1.0)
│
├── data/                            # CV data files
│   └── PureHydration_9Jan.csv       # Example real CV data
│
├── notebooks/                       # Training notebooks
│   └── train_unet_cv.ipynb          # Google Colab training script
│
├── outputs/                         # Generated outputs
│   ├── images/                      # Visualizations
│   ├── option_A/                    # Experimental (Gaussian-based)
│   └── option_B/                    # Production (CV-based)
│
├── analyze_real_cv.py               # Main script for real data analysis
├── validate_pipeline.py             # Validation demo (9×9 grid)
├── PIPELINE_EXPLAINED.md            # Technical explanation
├── METHODOLOGY.md                   # Design decisions
└── README.md                        # This file
```

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Input: Raw CV File                                       │
│    ├─ Parse (UTF-16 encoding, header rows)                  │
│    ├─ Split forward/reverse scans                           │
│    └─ Interpolate to 500-point grid                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Baseline Subtraction                                     │
│    └─ Linear baseline (connect endpoints)                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. U-Net Peak Detection                                     │
│    ├─ Normalize signal (zero mean, unit variance)           │
│    ├─ Predict heatmap (500 probabilities)                   │
│    └─ High values (→1.0) indicate peak centers              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Peak Extraction                                          │
│    ├─ find_peaks() on heatmap (threshold=0.2)               │
│    └─ Output: List of voltage positions [E₁, E₂, ...]       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Gaussian Fitting (Deconvolution)                         │
│    ├─ For each detected peak position:                      │
│    │   ├─ Extract window around peak                        │
│    │   ├─ Fit Gaussian: I(V) = h·exp(-(V-E)²/2σ²)          │
│    │   └─ Extract: {center, height, sigma, FWHM, area}      │
│    └─ Output: Full peak parameters                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Output Parameters

For each detected peak, the pipeline extracts:

| Parameter | Symbol | Units | Description |
|-----------|--------|-------|-------------|
| **Center** | E₀ | V | Peak position (formal potential) |
| **Height** | h | µA | Peak current amplitude |
| **Sigma** | σ | V | Gaussian standard deviation |
| **FWHM** | Δ | mV | Full width at half maximum |
| **Area** | A | µA·V | Total peak area (∝ concentration) |
| **Confidence** | - | 0-1 | Heatmap peak confidence |

---

## Training Data

**Dataset**: 20,000 synthetic CV curves  
**Generator**: `MechanisticSimulator` (Option B - production)  
**Complexity**: 1-3 peaks per curve  
**Peak separation**: Minimum 80mV  
**Noise**: Gaussian white noise (σ = 0.1-0.5 µA)  

**Why MechanisticSimulator?**
- Produces realistic CV shapes (asymmetric peaks, diffusion tails)
- Better than pure Gaussians (Option A - experimental)
- Mimics real electrochemical behavior

---

## Next Steps for Calibration

To convert peak **area** → **concentration** (ppm):

1. **Prepare standard solutions** (CaCl₂, FeSO₄, ZnSO₄ at 0.1, 0.5, 1, 5, 10 ppm)
2. **Run CV scans** on each standard (3 replicates)
3. **Extract peak areas** using `analyze_real_cv.py`
4. **Plot area vs concentration** and fit linear regression:
   ```
   concentration (ppm) = k × area
   ```
5. **Use calibration constant `k`** for unknown samples

---

## Files Explained

### Core Scripts

**`analyze_real_cv.py`**
- Main script for analyzing real CV data
- Handles file parsing, full pipeline execution
- Outputs: detected peaks, parameters, visualization

**`validate_pipeline.py`**
- Validation demo with 9×9 grid
- Tests peak detection accuracy
- Reports metrics: count accuracy, position error

### Key Modules

**`src/generator_cv.py`**
- Training data generator using MechanisticSimulator
- Creates signal + heatmap pairs
- Saves to `.npz` format

**`src/gaussian_fitter.py`**
- Peak extraction from heatmap (scipy.signal.find_peaks)
- Gaussian curve fitting (scipy.optimize.curve_fit)
- Parameter calculation (center, height, FWHM, area)

**`src/cv_simulator.py`**
- MechanisticSimulator from VER 1.0
- Generates realistic CV curves with diffusion physics

---

## Documentation

- **PIPELINE_EXPLAINED.md** - How the ML model and deconvolution work
- **METHODOLOGY.md** - Why we chose U-Net, training approach
- **CV_SIMULATION_OPTIONS.md** - Comparison of data generation methods
- **NEXT_STEPS_REDUCTION.md** - Future work (adding reduction scans)

---

## Troubleshooting

**Issue**: "ModuleNotFoundError: No module named 'cv_simulator'"  
**Fix**: Run from VER_2.0/ directory, not root

**Issue**: Model predicts flat heatmap (no peaks)  
**Fix**: Check signal normalization - should have mean≈0, std≈1

**Issue**: Too many false peaks detected  
**Fix**: Increase threshold in `fit_all_peaks(threshold=0.2)` → `0.3` or `0.4`

**Issue**: Peak positions off by ~50mV consistently  
**Fix**: Likely baseline subtraction issue - check input data voltage range

---

## Performance Benchmarks

**Training**: ~2 hours for 20k samples (Google Colab T4 GPU)  
**Inference**: ~50ms per CV curve (on CPU)  
**Gaussian Fitting**: ~10ms per peak  

**Total pipeline time**: < 100ms per CV curve

---

## Version History

**VER 2.0** (Current)
- ML-based peak detection (U-Net)
- Gaussian fitting for parameters
- 97.5% peak count accuracy

**VER 1.0** (Archived → `../VER_1.0_ARCHIVE/`)
- Heuristic-based peak detection
- Lasso deconvolution
- Issues with multi-peak detection

---

## Citation

If you use this pipeline, please cite:

```
VER 2.0 Peak Detection Pipeline
Neotrient CV Analysis Project
2026
```

---

## Contact

For questions or issues, refer to:
- Implementation plan: `../brain/.../implementation_plan.md`
- Technical docs: `PIPELINE_EXPLAINED.md`
