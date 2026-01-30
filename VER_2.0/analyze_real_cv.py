"""
VER 2.0 - Real CV Data Analysis
================================
Run the full pipeline on real experimental CV data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os

sys.path.insert(0, 'src')
from gaussian_fitter import GaussianFitter, gaussian


def parse_real_cv(filepath):
    """Parse real CV file (handles UTF-16LE encoding with headers)."""
    import subprocess
    
    # Convert UTF-16LE to UTF-8
    result = subprocess.run(['iconv', '-f', 'UTF-16LE', '-t', 'UTF-8', filepath], 
                           capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    
    # Find the data start (skip header lines)
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('0.') or line.strip().startswith('-0.'):
            data_start = i
            break
    
    print(f"Data starts at line {data_start}")
    
    # Parse data lines
    v_data = []
    i_data = []
    for line in lines[data_start:]:
        parts = line.split(',')
        if len(parts) >= 2:
            try:
                v = float(parts[0])
                i = float(parts[1])
                v_data.append(v)
                i_data.append(i)
            except:
                continue
    
    return np.array(v_data), np.array(i_data)


def process_real_cv(v_raw, i_raw, target_points=500, v_range=(-0.6, 0.6)):
    """
    Process raw CV data to match the training format.
    
    1. Split into forward/reverse scans
    2. Interpolate to fixed grid
    3. Subtract linear baseline
    """
    # Detect turn point
    start_v = v_raw[0]
    direction = np.sign(v_raw[1] - v_raw[0])
    
    # Find where direction changes
    turn_idx = None
    for i in range(1, len(v_raw)):
        if direction > 0 and v_raw[i] < v_raw[i-1]:
            turn_idx = i
            break
        elif direction < 0 and v_raw[i] > v_raw[i-1]:
            turn_idx = i
            break
    
    if turn_idx is None:
        turn_idx = len(v_raw) // 2
    
    # Split scans
    fwd_v, fwd_i = v_raw[:turn_idx], i_raw[:turn_idx]
    rev_v, rev_i = v_raw[turn_idx:], i_raw[turn_idx:]
    
    # Use the scan that matches model training direction (0.5 to -0.5)
    # Our model was trained on v_start=0.5 to v_end=-0.5
    if fwd_v[0] > fwd_v[-1]:  # Forward goes from high to low
        v_scan, i_scan = fwd_v, fwd_i
    else:
        v_scan, i_scan = rev_v, rev_i
    
    # Create target voltage grid
    voltage_grid = np.linspace(v_range[1], v_range[0], target_points)
    
    # Interpolate to fixed grid
    sort_idx = np.argsort(v_scan)
    signal = np.interp(voltage_grid, v_scan[sort_idx], i_scan[sort_idx], 
                       left=np.nan, right=np.nan)
    
    # Handle NaN at edges (extrapolate)
    signal = pd.Series(signal).interpolate(method='linear', limit_direction='both').values
    
    # Baseline subtraction
    baseline = np.linspace(signal[0], signal[-1], len(signal))
    signal_subtracted = signal - baseline
    
    return voltage_grid, signal, signal_subtracted


def run_pipeline(filepath):
    """Run full pipeline on real CV data."""
    print(f"Processing: {filepath}")
    
    # Load model
    model = tf.keras.models.load_model('models/peak_detector_cv.keras')
    norm = np.load('models/normalization_params_cv.npz')
    
    # Parse and process data
    v_raw, i_raw = parse_real_cv(filepath)
    print(f"Raw data: {len(v_raw)} points, V=[{v_raw.min():.2f}, {v_raw.max():.2f}]")
    
    voltage_grid, signal_raw, signal_sub = process_real_cv(v_raw, i_raw)
    
    # Normalize for model
    signal_norm = (signal_raw - norm['signal_mean']) / norm['signal_std']
    
    # Predict
    heatmap = model.predict(signal_norm.reshape(1, -1, 1), verbose=0).squeeze()
    
    # Fit Gaussians
    fitter = GaussianFitter(voltage_grid)
    peaks = fitter.fit_all_peaks(signal_sub, heatmap, threshold=0.2)
    
    print("\n" + fitter.format_results(peaks))
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Real CV Analysis: {os.path.basename(filepath)}", fontweight='bold')
    
    # Raw CV
    axes[0, 0].plot(v_raw, i_raw, 'b-', lw=0.5, alpha=0.7)
    axes[0, 0].set_title("Raw CV Data")
    axes[0, 0].set_xlabel("V")
    axes[0, 0].set_ylabel("µA")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Heatmap
    axes[0, 1].fill_between(voltage_grid, heatmap, alpha=0.5, color='orange')
    axes[0, 1].set_title("U-Net Predicted Heatmap")
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].grid(True, alpha=0.3)
    for p in peaks:
        axes[0, 1].axvline(p['center'], color='red', ls='--', alpha=0.7)
    
    # Signal with fits
    axes[1, 0].plot(voltage_grid, signal_sub, 'b-', lw=1, alpha=0.7, label='Signal (baseline sub)')
    for p in peaks:
        g = gaussian(voltage_grid, p['center'], p['height'], p['sigma'])
        axes[1, 0].plot(voltage_grid, g, '--', lw=2, label=f"Peak @ {p['center']:.3f}V")
    axes[1, 0].set_title("Fitted Gaussians")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Results table
    axes[1, 1].axis('off')
    if peaks:
        table_data = [["Peak", "Center (V)", "Height (µA)", "FWHM (mV)", "Area", "Conf"]]
        for i, p in enumerate(peaks):
            table_data.append([
                f"{i+1}",
                f"{p['center']:.4f}",
                f"{p['height']:.1f}",
                f"{p['fwhm']*1000:.1f}",
                f"{p['area']:.1f}",
                f"{p['confidence']:.2f}"
            ])
        table = axes[1, 1].table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
    axes[1, 1].set_title("Extracted Parameters", pad=20)
    
    plt.tight_layout()
    plt.savefig("real_cv_analysis.png", dpi=150)
    print("\nSaved real_cv_analysis.png")
    plt.show()
    
    return peaks


if __name__ == "__main__":
    peaks = run_pipeline('data/PureHydration_9Jan.csv')
