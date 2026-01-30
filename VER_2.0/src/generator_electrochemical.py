"""
VER 2.0 Training Data Generator (Electrochemically Accurate)
=============================================================
Uses Randles-Ševčík simulator for proper oxidation/reduction behavior.
"""
import numpy as np
import sys
sys.path.insert(0, 'src')
from randles_sevcik_simulator import ImprovedCVSimulator as ReversibleCVSimulator


class ElectrochemicalDataGenerator:
    """
    Generate training data for VER 2.0 using proper electrochemistry.
    
    Key improvements over generator_cv.py:
    - Proper oxidation AND reduction peaks (not mirrored)
    - Correct ΔEp = 59/n mV separation
    - Based on Randles-Ševčík equation
    """
    
    def __init__(self, 
                 v_start: float = 0.5,
                 v_end: float = -0.5,
                 n_points: int = 500):
        """
        Parameters:
        -----------
        v_start : Starting voltage (V)
        v_end : Ending voltage (V) 
        n_points : Points per direction (total will be 2*n_points)
        """
        self.v_start = v_start
        self.v_end = v_end
        self.n_points = n_points
        
        # Initialize simulator
        self.sim = ReversibleCVSimulator(
            v_start=v_start,
            v_end=v_end,
            scan_rate=0.1,  # V/s
            n_points=n_points
        )
        
        # Standard voltage grid for interpolation
        self.voltage_grid = np.linspace(v_start, v_end, n_points)
        
    def extract_oxidation_signal(self, V: np.ndarray, I: np.ndarray) -> np.ndarray:
        """
        Extract and interpolate the oxidation (forward) scan.
        
        Parameters:
        -----------
        V : Full voltage array (forward + reverse)
        I : Full current array (forward + reverse)
        
        Returns:
        --------
        Oxidation signal interpolated to standard voltage grid
        """
        # Split forward scan
        n_half = len(V) // 2
        v_fwd = V[:n_half]
        i_fwd = I[:n_half]
        
        # Interpolate to standard grid
        signal = np.interp(self.voltage_grid, v_fwd, i_fwd)
        
        # Subtract linear baseline
        baseline = np.linspace(signal[0], signal[-1], len(signal))
        signal_sub = signal - baseline
        
        return signal_sub
    
    def create_heatmap(self, species: list, width: float = 0.05) -> np.ndarray:
        """
        Create target heatmap from species E0 positions.
        
        Parameters:
        -----------
        species : List of species dicts with 'E0' key
        width : Gaussian width for heatmap peaks (V)
        
        Returns:
        --------
        Heatmap array (0-1 normalized)
        """
        heatmap = np.zeros(self.n_points)
        
        for sp in species:
            e0 = sp['E0']
            # Create Gaussian peak at E0
            peak = np.exp(-((self.voltage_grid - e0) ** 2) / (2 * width ** 2))
            heatmap += peak
        
        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def generate_sample(self, n_peaks: int = None):
        """
        Generate one training sample.
        
        Parameters:
        -----------
        n_peaks : Number of peaks (1-3 if None)
        
        Returns:
        --------
        signal : Oxidation signal (baseline-subtracted)
        heatmap : Target heatmap
        params : Species parameters
        """
        # Generate random species
        species = self.sim.generate_random_species(n_species=n_peaks)
        
        # Simulate full CV
        V, I = self.sim.simulate_species(species)
        
        # Extract oxidation signal
        signal = self.extract_oxidation_signal(V, I)
        
        # Create heatmap
        heatmap = self.create_heatmap(species)
        
        # Extract parameters for ground truth
        params = []
        for sp in species:
            params.append({
                'E0': sp['E0'],
                'C': sp['C'],
                'n': sp.get('n', 1)
            })
        
        return signal, heatmap, params
    
    def generate_dataset(self, n_samples: int = 20000, save_path: str = None):
        """
        Generate full dataset for training.
        
        Parameters:
        -----------
        n_samples : Number of samples
        save_path : Path to save .npz file
        
        Returns:
        --------
        signals, heatmaps, all_params
        """
        signals = []
        heatmaps = []
        all_params = []
        
        print(f"Generating {n_samples} samples with Randles-Ševčík simulator...")
        
        for i in range(n_samples):
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{n_samples}")
            
            signal, heatmap, params = self.generate_sample()
            signals.append(signal)
            heatmaps.append(heatmap)
            all_params.append(params)
        
        signals = np.array(signals)
        heatmaps = np.array(heatmaps)
        
        if save_path:
            np.savez_compressed(
                save_path,
                signals=signals,
                heatmaps=heatmaps,
                voltage_grid=self.voltage_grid
            )
            print(f"\nSaved to: {save_path}")
            
            # Save parameters separately (JSON)
            import json
            param_file = save_path.replace('.npz', '_params.json')
            with open(param_file, 'w') as f:
                json.dump(all_params, f, indent=2)
            print(f"Saved parameters to: {param_file}")
        
        return signals, heatmaps, all_params


# =============================================================================
# Demo / Test
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    gen = ElectrochemicalDataGenerator()
    
    # Generate a few samples
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle("Training Data: Randles-Ševčík Based (Oxidation Scan Only)", 
                 fontweight='bold', fontsize=14)
    
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            
            # Generate sample
            signal, heatmap, params = gen.generate_sample(n_peaks=np.random.randint(1, 4))
            
            # Plot signal and heatmap
            ax2 = ax.twinx()
            ax.plot(gen.voltage_grid, signal, 'b-', lw=1.5, label='Signal (ox)')
            ax2.fill_between(gen.voltage_grid, heatmap, alpha=0.3, color='orange', label='Heatmap')
            
            # Mark E0
            for p in params:
                ax.axvline(p['E0'], color='red', ls='--', alpha=0.6, lw=1)
            
            e0_str = ', '.join([f"{p['E0']:.2f}V" for p in params])
            ax.set_title(f"{len(params)} peaks: {e0_str}", fontsize=9)
            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("Current (µA)", color='b')
            ax2.set_ylabel("Heatmap", color='orange')
            ax.grid(True, alpha=0.3)
            
            if row == 0 and col == 0:
                ax.legend(loc='upper left', fontsize=8)
                ax2.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("outputs/images/electrochemical_training_samples.png", dpi=200)
    print("Saved: outputs/images/electrochemical_training_samples.png")
    print("\nReady to generate 20k training samples!")
    plt.show()
