"""
Show FULL CV Output: Oxidation + Reduction Scans
=================================================
Visualize what MechanisticSimulator actually generates before we extract
just the oxidation scan for training.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from cv_simulator import MechanisticSimulator

# Create simulator
sim = MechanisticSimulator(v_start=0.5, v_end=-0.5, scan_rate=0.1)

# Create 3x3 grid showing full CV curves
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle("Full CV Output: Oxidation (forward) + Reduction (reverse)", 
             fontweight='bold', fontsize=16)

for idx, ax in enumerate(axes.flat):
    # Random species
    n_peaks = np.random.randint(1, 3)
    species = []
    for _ in range(n_peaks):
        e0 = np.random.uniform(-0.3, 0.3)
        height = np.random.uniform(20, 80)
        species.append({'E0': e0, 'height': height})
    
    # Generate FULL CV curve
    v, i = sim.run_simulation(species, c_dl=5.0, noise_std=0.3)
    
    # Split into forward and reverse
    n_half = len(v) // 2
    fwd_v, fwd_i = v[:n_half], i[:n_half]
    rev_v, rev_i = v[n_half:], i[n_half:]
    
    # Plot
    ax.plot(fwd_v, fwd_i, 'b-', lw=1.5, label='Oxidation (fwd)', alpha=0.8)
    ax.plot(rev_v, rev_i, 'r-', lw=1.5, label='Reduction (rev)', alpha=0.8)
    
    # Mark E0 positions
    for sp in species:
        ax.axvline(sp['E0'], color='green', ls='--', alpha=0.5, lw=1)
        ax.axvline(sp['E0'] - 0.06, color='orange', ls='--', alpha=0.5, lw=1)  # Shifted for reduction
    
    ax.set_title(f"Sample {idx+1}: {n_peaks} species", fontsize=11)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (µA)")
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig("outputs/images/full_cv_with_reduction.png", dpi=200, bbox_inches='tight')
print("Saved: outputs/images/full_cv_with_reduction.png")
plt.show()
