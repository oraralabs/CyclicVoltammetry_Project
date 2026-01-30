"""
Visualize Training Data: 5x5 Grid of Generated CV Curves
=========================================================
Show what the MechanisticSimulator generates for training.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')
from generator_cv import CVDataGenerator

# Create generator
gen = CVDataGenerator()

# Create 5x5 grid
fig, axes = plt.subplots(5, 5, figsize=(20, 16))
fig.suptitle("Training Data: Generated CV Curves (MechanisticSimulator)", 
             fontweight='bold', fontsize=16)

for row in range(5):
    for col in range(5):
        ax = axes[row, col]
        
        # Generate random sample
        n_peaks = np.random.randint(1, 4)
        signal, heatmap, params = gen.generate_sample(n_peaks=n_peaks)
        
        # Plot the CV signal
        ax.plot(gen.voltage_grid, signal, 'b-', lw=1.5)
        
        # Mark true peak positions
        for p in params:
            ax.axvline(p['E0'], color='red', ls='--', alpha=0.6, lw=1)
        
        # Title with peak count
        ax.set_title(f"{n_peaks} peak(s)", fontsize=10)
        ax.set_xlim(gen.voltage_grid.max(), gen.voltage_grid.min())  # Reverse x-axis
        ax.set_ylim(-20, 120)
        ax.grid(True, alpha=0.3)
        
        # Labels only on edges
        if col == 0:
            ax.set_ylabel("Current (µA)", fontsize=9)
        if row == 4:
            ax.set_xlabel("Voltage (V)", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/images/training_data_grid_5x5.png", dpi=200, bbox_inches='tight')
print("Saved: outputs/images/training_data_grid_5x5.png")
plt.show()
