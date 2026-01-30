"""
VER 2.0 Option B - CV-Based Training Data Generator
====================================================
Uses MechanisticSimulator to generate realistic CV curves with known peak parameters.
This creates more realistic training data than pure Gaussian summation.
"""
import numpy as np
import sys
import os
import json

# Add parent src to path for MechanisticSimulator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from cv_simulator import MechanisticSimulator


class CVDataGenerator:
    """
    Generate realistic CV training data using MechanisticSimulator.
    
    Output format:
    - signal: Oxidation scan only (after baseline subtraction)
    - heatmap: Peak probability at each voltage
    - params: Ground truth {E0, height} for each species
    """
    
    def __init__(self, n_points=500, v_start=0.5, v_end=-0.5):
        self.n_points = n_points
        self.v_start = v_start
        self.v_end = v_end
        self.voltage_grid = np.linspace(v_start, v_end, n_points)
        
    def generate_random_species(self, n_peaks):
        """Generate random species configs for MechanisticSimulator."""
        species = []
        used_e0 = []
        
        for _ in range(n_peaks):
            # Find a valid E0 position
            for attempt in range(100):
                e0 = np.random.uniform(-0.4, 0.4)
                # Minimum separation of 50mV to avoid complete overlap
                if all(abs(e0 - used) > 0.05 for used in used_e0):
                    used_e0.append(e0)
                    break
            else:
                continue
            
            # Random height (current magnitude)
            height = np.random.uniform(10.0, 100.0)
            species.append({'E0': e0, 'height': height})
        
        return species
    
    def extract_oxidation_signal(self, v_total, i_total):
        """
        Extract just the forward (oxidation) scan and apply simple baseline correction.
        Returns the signal on a fixed voltage grid.
        """
        # First half is forward scan
        n_half = len(v_total) // 2
        ox_v = v_total[:n_half]
        ox_i = i_total[:n_half]
        
        # Simple linear baseline: connect endpoints
        baseline = np.linspace(ox_i[0], ox_i[-1], len(ox_i))
        ox_signal = ox_i - baseline
        
        # Interpolate to fixed grid
        ox_signal_interp = np.interp(self.voltage_grid, ox_v[::-1], ox_signal[::-1])
        
        return ox_signal_interp
    
    def create_heatmap_target(self, species):
        """
        Create heatmap target for training.
        Peak center at each E0 position.
        """
        heatmap = np.zeros(self.n_points)
        target_width = 0.02  # Narrow for precise localization
        
        for sp in species:
            e0 = sp['E0']
            heatmap += np.exp(-0.5 * ((self.voltage_grid - e0) / target_width) ** 2)
        
        heatmap = np.clip(heatmap, 0, 1)
        return heatmap
    
    def generate_sample(self, n_peaks=None):
        """
        Generate one training sample.
        
        Returns:
            signal: Baseline-subtracted oxidation signal
            heatmap: Peak probability target
            params: List of {E0, height} dicts
        """
        if n_peaks is None:
            n_peaks = np.random.randint(1, 4)  # 1, 2, or 3 peaks
        
        species = self.generate_random_species(n_peaks)
        
        if len(species) == 0:
            # Fallback: generate at least one peak
            species = [{'E0': 0.0, 'height': 50.0}]
        
        # Generate CV curve
        sim = MechanisticSimulator(
            v_start=self.v_start,
            v_end=self.v_end,
            scan_rate=0.1
        )
        noise_std = np.random.uniform(0.05, 0.5)
        v, i = sim.run_simulation(species, noise_std=noise_std)
        
        # Extract oxidation signal
        signal = self.extract_oxidation_signal(v, i)
        
        # Create heatmap target
        heatmap = self.create_heatmap_target(species)
        
        # Convert species to simple format
        params = [{'E0': sp['E0'], 'height': sp['height']} for sp in species]
        
        return signal, heatmap, params
    
    def generate_dataset(self, n_samples=20000, save_path=None):
        """
        Generate full training dataset.
        """
        signals = []
        heatmaps = []
        all_params = []
        
        print(f"Generating {n_samples} CV-based samples...")
        for i in range(n_samples):
            if (i + 1) % 2000 == 0:
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
            params_path = save_path.replace('.npz', '_params.json')
            with open(params_path, 'w') as f:
                json.dump(all_params, f)
            print(f"Saved to {save_path}")
        
        return signals, heatmaps, all_params


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    gen = CVDataGenerator()
    
    # Show comparison: input CV-style signal vs target heatmap
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle("Option B: CV-Based Training Data (MechanisticSimulator)", fontweight='bold')
    
    for row in range(4):
        n_peaks = np.random.randint(1, 4)
        signal, heatmap, params = gen.generate_sample(n_peaks=n_peaks)
        
        # Left: Input signal (CV oxidation scan after baseline sub)
        ax1 = axes[row, 0]
        ax1.plot(gen.voltage_grid, signal, 'b-', lw=1)
        for p in params:
            ax1.axvline(p['E0'], color='red', ls='--', alpha=0.7)
        ax1.set_title(f"INPUT: CV Signal ({n_peaks} peaks)")
        ax1.set_xlabel("V")
        ax1.set_ylabel("µA")
        ax1.grid(True, alpha=0.3)
        
        # Right: Target heatmap
        ax2 = axes[row, 1]
        ax2.fill_between(gen.voltage_grid, heatmap, alpha=0.5, color='green')
        ax2.set_title("TARGET: Peak Heatmap")
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("cv_generator_samples.png", dpi=150)
    print("Saved cv_generator_samples.png")
    plt.show()
