"""
VER 2.0 - Gaussian Fitting Module
==================================
After the U-Net detects peak positions, this module fits Gaussians
to extract complete parameters: center, height, and width.

Pipeline:
Signal → U-Net → Heatmap → Peak Detection → Gaussian Fitting → Parameters
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from typing import List, Dict, Tuple, Optional


def gaussian(x: np.ndarray, center: float, height: float, sigma: float) -> np.ndarray:
    """Single Gaussian function."""
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def multi_gaussian(x: np.ndarray, *params) -> np.ndarray:
    """
    Sum of multiple Gaussians.
    params: [center1, height1, sigma1, center2, height2, sigma2, ...]
    """
    y = np.zeros_like(x)
    n_peaks = len(params) // 3
    for i in range(n_peaks):
        center = params[i * 3]
        height = params[i * 3 + 1]
        sigma = params[i * 3 + 2]
        y += gaussian(x, center, height, sigma)
    return y


class GaussianFitter:
    """
    Fit Gaussians to signal at detected peak positions.
    """
    
    def __init__(self, voltage_grid: np.ndarray):
        self.voltage_grid = voltage_grid
        self.v_range = voltage_grid.max() - voltage_grid.min()
    
    def extract_peak_positions(self, heatmap: np.ndarray, 
                                threshold: float = 0.3,
                                min_distance: int = 10) -> List[float]:
        """
        Extract peak centers from heatmap.
        
        Args:
            heatmap: Predicted probability array from U-Net
            threshold: Minimum probability to consider a peak
            min_distance: Minimum separation between peaks (points)
            
        Returns:
            List of voltage positions where peaks are detected
        """
        peaks_idx, properties = find_peaks(heatmap, 
                                           height=threshold,
                                           distance=min_distance)
        
        centers = [self.voltage_grid[idx] for idx in peaks_idx]
        confidences = [heatmap[idx] for idx in peaks_idx]
        
        return centers, confidences
    
    def fit_single_gaussian(self, signal: np.ndarray, 
                            center_guess: float,
                            window_width: float = 0.15) -> Optional[Dict]:
        """
        Fit a single Gaussian around a detected peak.
        
        Args:
            signal: The CV signal
            center_guess: Initial guess for peak center (from heatmap)
            window_width: Width of fitting window in V
            
        Returns:
            Dict with {center, height, sigma, fwhm, area} or None if fit fails
        """
        # Extract window around the peak
        mask = np.abs(self.voltage_grid - center_guess) < window_width / 2
        x_window = self.voltage_grid[mask]
        y_window = signal[mask]
        
        if len(x_window) < 5:
            return None
        
        # Initial guesses
        height_guess = np.max(y_window) - np.min(y_window)
        sigma_guess = 0.05  # 50mV
        
        try:
            popt, pcov = curve_fit(
                gaussian, 
                x_window, 
                y_window,
                p0=[center_guess, height_guess, sigma_guess],
                bounds=(
                    [center_guess - 0.1, 0, 0.01],      # Lower bounds
                    [center_guess + 0.1, 500, 0.15]     # Upper bounds
                ),
                maxfev=1000
            )
            
            center, height, sigma = popt
            perr = np.sqrt(np.diag(pcov))
            
            # Calculate derived quantities
            fwhm = 2.355 * sigma  # Full width at half maximum
            area = height * sigma * np.sqrt(2 * np.pi)  # Total peak area
            
            return {
                'center': center,
                'height': height,
                'sigma': sigma,
                'fwhm': fwhm,
                'area': area,
                'center_err': perr[0],
                'height_err': perr[1],
                'sigma_err': perr[2]
            }
        except Exception as e:
            return None
    
    def fit_all_peaks(self, signal: np.ndarray, 
                      heatmap: np.ndarray,
                      threshold: float = 0.3) -> List[Dict]:
        """
        Main method: detect peaks from heatmap and fit Gaussians.
        
        Args:
            signal: The CV signal (baseline-subtracted)
            heatmap: U-Net predicted heatmap
            threshold: Detection threshold
            
        Returns:
            List of fitted peak parameters
        """
        centers, confidences = self.extract_peak_positions(heatmap, threshold)
        
        fitted_peaks = []
        for center, confidence in zip(centers, confidences):
            result = self.fit_single_gaussian(signal, center)
            if result:
                result['confidence'] = confidence
                fitted_peaks.append(result)
        
        # Sort by center position
        fitted_peaks.sort(key=lambda x: x['center'])
        return fitted_peaks
    
    def format_results(self, peaks: List[Dict]) -> str:
        """Pretty-print fitted peak parameters."""
        if not peaks:
            return "No peaks detected"
        
        lines = ["=" * 60]
        lines.append(f"{'PEAK':<6} {'CENTER (V)':<12} {'HEIGHT (µA)':<14} {'FWHM (mV)':<12} {'AREA':<10}")
        lines.append("-" * 60)
        
        for i, p in enumerate(peaks):
            lines.append(
                f"  {i+1:<4} {p['center']:>10.4f}   {p['height']:>11.2f}   "
                f"{p['fwhm']*1000:>9.1f}    {p['area']:>8.2f}"
            )
        
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Demo / Test
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys
    import os
    
    # Check if model exists
    model_path = 'models/peak_detector_cv.keras'
    norm_path = 'models/normalization_params_cv.npz'
    
    if not os.path.exists(model_path):
        print("Model not found. Please run Colab training first.")
        exit()
    
    import tensorflow as tf
    sys.path.insert(0, 'src')
    from generator_cv import CVDataGenerator
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    norm = np.load(norm_path)
    
    # Generate test signal
    gen = CVDataGenerator()
    signal, true_heatmap, true_params = gen.generate_sample(n_peaks=2)
    
    # Apply simple baseline subtraction (like ingestion.py does)
    # Linear baseline connecting endpoints
    baseline = np.linspace(signal[0], signal[-1], len(signal))
    signal_subtracted = signal - baseline
    
    # Predict
    signal_norm = (signal - norm['signal_mean']) / norm['signal_std']
    pred_heatmap = model.predict(signal_norm.reshape(1, -1, 1), verbose=0).squeeze()
    
    # Fit Gaussians on baseline-subtracted signal
    fitter = GaussianFitter(gen.voltage_grid)
    fitted_peaks = fitter.fit_all_peaks(signal_subtracted, pred_heatmap)
    
    # Print results
    print("\n" + fitter.format_results(fitted_peaks))
    print("\nTrue parameters:")
    for i, p in enumerate(true_params):
        print(f"  Peak {i+1}: E0={p['E0']:.4f} V, height={p['height']:.2f} µA")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Full Pipeline: Signal → U-Net → Gaussian Fitting", fontweight='bold')
    
    # Original signal
    ax1 = axes[0, 0]
    ax1.plot(gen.voltage_grid, signal, 'b-', lw=1, label='CV Signal')
    ax1.set_title("Step 1: Input CV Signal")
    ax1.set_xlabel("V")
    ax1.set_ylabel("µA")
    ax1.grid(True, alpha=0.3)
    
    # Heatmap comparison
    ax2 = axes[0, 1]
    ax2.fill_between(gen.voltage_grid, true_heatmap, alpha=0.5, color='green', label='True')
    ax2.fill_between(gen.voltage_grid, pred_heatmap, alpha=0.5, color='orange', label='Predicted')
    ax2.set_title("Step 2: U-Net Heatmap")
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Fitted Gaussians on baseline-subtracted signal
    ax3 = axes[1, 0]
    ax3.plot(gen.voltage_grid, signal_subtracted, 'b-', lw=1, alpha=0.5, label='Signal (baseline sub)')
    
    # Reconstruct fitted Gaussians
    reconstructed = np.zeros_like(signal_subtracted)
    for p in fitted_peaks:
        g = gaussian(gen.voltage_grid, p['center'], p['height'], p['sigma'])
        reconstructed += g
        ax3.plot(gen.voltage_grid, g, '--', lw=2, label=f"Peak @ {p['center']:.3f}V")
    
    ax3.set_title("Step 3: Fitted Gaussians")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Results table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = [["Peak", "Center (V)", "Height (µA)", "FWHM (mV)", "Area"]]
    for i, p in enumerate(fitted_peaks):
        table_data.append([
            f"{i+1}",
            f"{p['center']:.4f}",
            f"{p['height']:.1f}",
            f"{p['fwhm']*1000:.1f}",
            f"{p['area']:.1f}"
        ])
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    ax4.set_title("Step 4: Extracted Parameters", pad=20)
    
    plt.tight_layout()
    plt.savefig("full_pipeline_demo.png", dpi=150)
    print("\nSaved full_pipeline_demo.png")
    plt.show()
