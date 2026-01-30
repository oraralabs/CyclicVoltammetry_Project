"""
VER 2.0 Option C - Gaussian → CV Curve → Recover Gaussians
============================================================
This generator explicitly:
1. Creates Gaussians with known parameters (center, height, width)
2. Transforms them into CV-like curves (add baseline, create loop)
3. The model learns to recover the original Gaussian parameters

This is the INVERSE of ingestion.py which does:
CV Curve → baseline subtraction → spectra (Gaussians)
"""
import numpy as np
import json


class OptionCGenerator:
    """
    Full pipeline: Gaussians → CV Curve → Training Data
    
    The transformation mimics what a real CV looks like:
    - Gaussians become the "spectra" (after baseline subtraction)
    - We add a linear baseline back ON
    - We add capacitive current (constant offset)
    - We create both forward (ox) and reverse (red) scans
    """
    
    def __init__(self, n_points=500, v_start=0.5, v_end=-0.5):
        self.n_points = n_points
        self.v_start = v_start
        self.v_end = v_end
        self.voltage_grid = np.linspace(v_start, v_end, n_points)
        
    def generate_gaussians(self, n_peaks=None):
        """
        Generate random Gaussian peaks with known parameters.
        
        Returns:
            spectra: The pure Gaussian signal (what we want to recover)
            params: List of {center, height, width} dicts
        """
        if n_peaks is None:
            n_peaks = np.random.randint(1, 4)
        
        spectra = np.zeros(self.n_points)
        params = []
        used_centers = []
        
        for _ in range(n_peaks):
            # Find valid center (minimum 50mV separation)
            for attempt in range(100):
                center = np.random.uniform(-0.4, 0.4)
                if all(abs(center - used) > 0.05 for used in used_centers):
                    used_centers.append(center)
                    break
            else:
                continue
            
            # Random height and width
            height = np.random.uniform(10.0, 100.0)
            width = np.random.uniform(0.03, 0.10)  # 30-100mV sigma
            
            # Add Gaussian to spectra
            gaussian = height * np.exp(-0.5 * ((self.voltage_grid - center) / width) ** 2)
            spectra += gaussian
            
            params.append({
                'center': center,
                'height': height,
                'width': width
            })
        
        if len(params) == 0:
            # Fallback
            center, height, width = 0.0, 50.0, 0.05
            spectra = height * np.exp(-0.5 * ((self.voltage_grid - center) / width) ** 2)
            params = [{'center': center, 'height': height, 'width': width}]
        
        return spectra, params
    
    def add_cv_artifacts(self, spectra):
        """
        Transform pure Gaussian spectra into a CV-like curve.
        This is the REVERSE of ingestion.py's baseline subtraction.
        
        Adds:
        1. Linear baseline (slope + intercept)
        2. Capacitive offset
        3. Random noise
        """
        v = self.voltage_grid
        
        # 1. Linear baseline (like fit_linear_baseline creates)
        slope = np.random.uniform(-50, 50)  # µA/V
        intercept = np.random.uniform(-20, 20)  # µA
        baseline = slope * v + intercept
        
        # 2. Capacitive current (constant offset that differs for ox/red)
        capacitive = np.random.uniform(-10, 10)
        
        # 3. Combine: CV = spectra + baseline + capacitive
        cv_signal = spectra + baseline + capacitive
        
        # 4. Add noise
        noise_std = np.random.uniform(0.1, 2.0)
        cv_signal += np.random.normal(0, noise_std, self.n_points)
        
        return cv_signal, baseline
    
    def create_heatmap_target(self, params):
        """
        Create narrow heatmap spikes at each Gaussian center.
        """
        heatmap = np.zeros(self.n_points)
        target_width = 0.02  # Narrow for precise localization
        
        for p in params:
            center = p['center']
            heatmap += np.exp(-0.5 * ((self.voltage_grid - center) / target_width) ** 2)
        
        return np.clip(heatmap, 0, 1)
    
    def generate_sample(self, n_peaks=None):
        """
        Generate one training sample.
        
        Returns:
            cv_signal: The CV-like curve (input to model)
            heatmap: Peak probability target
            spectra: The original Gaussians (ground truth)
            params: Gaussian parameters
        """
        # Step 1: Generate pure Gaussians
        spectra, params = self.generate_gaussians(n_peaks)
        
        # Step 2: Transform to CV curve
        cv_signal, baseline = self.add_cv_artifacts(spectra)
        
        # Step 3: Create heatmap target
        heatmap = self.create_heatmap_target(params)
        
        return cv_signal, heatmap, spectra, params
    
    def generate_dataset(self, n_samples=20000, save_path=None):
        """
        Generate full training dataset.
        """
        cv_signals = []
        heatmaps = []
        spectras = []
        all_params = []
        
        print(f"Generating {n_samples} Option C samples (Gaussian → CV)...")
        for i in range(n_samples):
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{n_samples}")
            
            cv_signal, heatmap, spectra, params = self.generate_sample()
            cv_signals.append(cv_signal)
            heatmaps.append(heatmap)
            spectras.append(spectra)
            all_params.append(params)
        
        cv_signals = np.array(cv_signals)
        heatmaps = np.array(heatmaps)
        spectras = np.array(spectras)
        
        if save_path:
            np.savez(save_path,
                     signals=cv_signals,  # Input to model
                     heatmaps=heatmaps,   # Target
                     spectras=spectras,   # Ground truth Gaussians
                     voltage_grid=self.voltage_grid)
            params_path = save_path.replace('.npz', '_params.json')
            with open(params_path, 'w') as f:
                json.dump(all_params, f)
            print(f"Saved to {save_path}")
        
        return cv_signals, heatmaps, spectras, all_params


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    gen = OptionCGenerator()
    
    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    fig.suptitle("Option C: Gaussian → CV Curve → Recover Gaussian", fontweight='bold', fontsize=14)
    
    for row in range(4):
        n_peaks = np.random.randint(1, 4)
        cv_signal, heatmap, spectra, params = gen.generate_sample(n_peaks=n_peaks)
        
        # Column 1: Original Gaussians (ground truth)
        ax1 = axes[row, 0]
        ax1.plot(gen.voltage_grid, spectra, 'g-', lw=2, label='True Spectra')
        for p in params:
            ax1.axvline(p['center'], color='red', ls='--', alpha=0.5)
        ax1.set_title(f"Step 1: Gaussians ({n_peaks} peaks)")
        ax1.set_ylabel("µA")
        ax1.grid(True, alpha=0.3)
        
        # Column 2: CV Curve (after adding baseline)
        ax2 = axes[row, 1]
        ax2.plot(gen.voltage_grid, cv_signal, 'b-', lw=1)
        ax2.set_title("Step 2: CV Curve (+ baseline)")
        ax2.grid(True, alpha=0.3)
        
        # Column 3: Target Heatmap
        ax3 = axes[row, 2]
        ax3.fill_between(gen.voltage_grid, heatmap, alpha=0.5, color='orange')
        ax3.set_title("Step 3: Target Heatmap")
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("option_c_samples.png", dpi=150)
    print("Saved option_c_samples.png")
    plt.show()
