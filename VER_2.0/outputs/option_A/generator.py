"""
VER 2.0 - Synthetic Training Data Generator
============================================
Generates CV-like signals with known Gaussian peak parameters.
Used to create supervised training data for the peak detection model.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import json


@dataclass
class PeakParams:
    """Ground truth parameters for a single Gaussian peak."""
    center: float      # Peak center voltage (V)
    height: float      # Peak height (µA)
    width: float       # Peak width σ (V)
    
    def to_dict(self):
        return {'center': self.center, 'height': self.height, 'width': self.width}


class SignalGenerator:
    """
    Generates synthetic signals with known Gaussian components.
    
    The "inverted pipeline" approach:
    1. Generate random Gaussian parameters (ground truth)
    2. Sum Gaussians to create pure signal
    3. Add realistic artifacts (baseline, noise)
    4. Output: (signal, ground_truth_params)
    """
    
    def __init__(self, 
                 n_points: int = 500,
                 v_min: float = -0.6,
                 v_max: float = 0.6):
        """
        Args:
            n_points: Number of voltage points in signal
            v_min: Minimum voltage
            v_max: Maximum voltage
        """
        self.n_points = n_points
        self.v_min = v_min
        self.v_max = v_max
        self.voltage_grid = np.linspace(v_min, v_max, n_points)
    
    def generate_gaussian(self, center: float, height: float, width: float) -> np.ndarray:
        """Generate a single Gaussian peak."""
        return height * np.exp(-0.5 * ((self.voltage_grid - center) / width) ** 2)
    
    def generate_random_peaks(self, n_peaks: int) -> List[PeakParams]:
        """
        Generate random peak parameters ensuring sufficient separation.
        
        Args:
            n_peaks: Number of peaks (1-3)
            
        Returns:
            List of PeakParams with random but valid values
        """
        peaks = []
        used_centers = []
        
        for _ in range(n_peaks):
            # Keep trying until we find a valid center
            for attempt in range(100):
                center = np.random.uniform(-0.4, 0.4)
                
                # Check minimum separation from existing peaks
                # Allow overlapping (30-80mV separation) - this is what we want to train on!
                min_sep = 0.03  # 30mV minimum
                if all(abs(center - c) > min_sep for c in used_centers):
                    used_centers.append(center)
                    break
            else:
                # If we couldn't find a spot, skip this peak
                continue
            
            # Random height and width
            height = np.random.uniform(10.0, 100.0)
            width = np.random.uniform(0.03, 0.12)
            
            peaks.append(PeakParams(center=center, height=height, width=width))
        
        return peaks
    
    def generate_signal(self, peaks: List[PeakParams], 
                        add_baseline: bool = True,
                        add_noise: bool = True) -> np.ndarray:
        """
        Generate a signal by summing Gaussians and adding artifacts.
        
        Args:
            peaks: List of PeakParams defining the Gaussians
            add_baseline: Whether to add linear baseline tilt
            add_noise: Whether to add random noise
            
        Returns:
            1D signal array
        """
        # Sum all Gaussians
        signal = np.zeros(self.n_points)
        for p in peaks:
            signal += self.generate_gaussian(p.center, p.height, p.width)
        
        # Add linear baseline (capacitive-like tilt)
        if add_baseline:
            slope = np.random.uniform(-2.0, 2.0)
            intercept = np.random.uniform(-5.0, 5.0)
            baseline = slope * self.voltage_grid + intercept
            signal += baseline
        
        # Add Gaussian noise
        if add_noise:
            noise_std = np.random.uniform(0.5, 2.0)
            signal += np.random.normal(0, noise_std, self.n_points)
        
        return signal
    
    def create_heatmap_target(self, peaks: List[PeakParams]) -> np.ndarray:
        """
        Create a heatmap target for training (peak center probability).
        
        Each peak creates a narrow Gaussian centered at its position.
        The model learns to predict this heatmap from the signal.
        """
        heatmap = np.zeros(self.n_points)
        target_width = 0.02  # Narrow Gaussian for precise localization
        
        for p in peaks:
            heatmap += np.exp(-0.5 * ((self.voltage_grid - p.center) / target_width) ** 2)
        
        # Clip to [0, 1] range
        heatmap = np.clip(heatmap, 0, 1)
        return heatmap
    
    def generate_sample(self, n_peaks: int = None) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """
        Generate a single training sample.
        
        Args:
            n_peaks: Number of peaks (random 1-3 if None)
            
        Returns:
            (signal, heatmap, params_list)
        """
        if n_peaks is None:
            n_peaks = np.random.randint(1, 4)  # 1, 2, or 3 peaks
        
        peaks = self.generate_random_peaks(n_peaks)
        signal = self.generate_signal(peaks)
        heatmap = self.create_heatmap_target(peaks)
        params = [p.to_dict() for p in peaks]
        
        return signal, heatmap, params
    
    def generate_dataset(self, n_samples: int = 20000, 
                         save_path: str = None) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Generate a full training dataset.
        
        Args:
            n_samples: Number of samples to generate
            save_path: Optional path to save as .npz
            
        Returns:
            (signals, heatmaps, all_params)
        """
        signals = []
        heatmaps = []
        all_params = []
        
        print(f"Generating {n_samples} samples...")
        for i in range(n_samples):
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{n_samples}")
            
            signal, heatmap, params = self.generate_sample()
            signals.append(signal)
            heatmaps.append(heatmap)
            all_params.append(params)
        
        signals = np.array(signals)
        heatmaps = np.array(heatmaps)
        
        if save_path:
            np.savez(save_path, 
                     signals=signals, 
                     heatmaps=heatmaps,
                     voltage_grid=self.voltage_grid)
            # Save params as JSON (numpy can't store list of dicts)
            params_path = save_path.replace('.npz', '_params.json')
            with open(params_path, 'w') as f:
                json.dump(all_params, f)
            print(f"Saved to {save_path} and {params_path}")
        
        return signals, heatmaps, all_params


# =============================================================================
# Quick test / visualization
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    gen = SignalGenerator()
    
    # Generate a few examples
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("VER 2.0 Generator - Sample Outputs", fontsize=14, fontweight='bold')
    
    for row in range(3):
        n_peaks = row + 1
        signal, heatmap, params = gen.generate_sample(n_peaks=n_peaks)
        
        # Plot signal
        ax1 = axes[row, 0]
        ax1.plot(gen.voltage_grid, signal, 'b-', lw=1)
        for p in params:
            ax1.axvline(p['center'], color='red', ls='--', alpha=0.7)
        ax1.set_title(f"Signal ({n_peaks} peak{'s' if n_peaks > 1 else ''})")
        ax1.set_xlabel("Voltage (V)")
        ax1.set_ylabel("Current (µA)")
        ax1.grid(True, alpha=0.3)
        
        # Plot heatmap
        ax2 = axes[row, 1]
        ax2.fill_between(gen.voltage_grid, heatmap, alpha=0.5, color='green')
        ax2.plot(gen.voltage_grid, heatmap, 'g-', lw=1.5)
        ax2.set_title("Target Heatmap (Peak Centers)")
        ax2.set_xlabel("Voltage (V)")
        ax2.set_ylabel("Probability")
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("generator_demo.png", dpi=150)
    print("Saved demo to generator_demo.png")
    plt.show()
