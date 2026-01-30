"""
Compare Generated vs Real CV Curves
====================================
Analyze differences to refine the simulator.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import sys
sys.path.insert(0, 'src')
from randles_sevcik_simulator import ImprovedCVSimulator

# Parse real CV data
def parse_real_cv(filepath):
    result = subprocess.run(['iconv', '-f', 'UTF-16LE', '-t', 'UTF-8', filepath], 
                           capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('0.') or line.strip().startswith('-0.'):
            data_start = i
            break
    
    v_data, i_data = [], []
    for line in lines[data_start:]:
        parts = line.split(',')
        if len(parts) >= 2:
            try:
                v_data.append(float(parts[0]))
                i_data.append(float(parts[1]))
            except:
                continue
    
    return np.array(v_data), np.array(i_data)

# Load real data
v_real, i_real = parse_real_cv('data/PureHydration_9Jan.csv')

# Split into oxidation scan (first half)
n_half = len(v_real) // 2
v_real_ox = v_real[:n_half]
i_real_ox = i_real[:n_half]

# Generate synthetic data
sim = ImprovedCVSimulator(v_start=0.5, v_end=-0.5, scan_rate=0.1, n_points=500)

# Similar complexity to real data (looks like 3-4 peaks)
species = [
    {'E0': -0.35, 'C': 2.0, 'D': 1e-5, 'n': 1},
    {'E0': -0.33, 'C': 2.0, 'D': 1e-5, 'n': 1},
    {'E0': 0.02, 'C': 1.5, 'D': 1e-5, 'n': 1},
]
V_syn, I_syn = sim.simulate_species(species)
v_syn_ox = V_syn[:len(V_syn)//2]
i_syn_ox = I_syn[:len(I_syn)//2]

# Compare
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Real vs Generated CV Curves - Comparison", fontweight='bold', fontsize=14)

# Real data - full
axes[0, 0].plot(v_real_ox, i_real_ox, 'b-', lw=1.5, label='Real (PureHydration)')
axes[0, 0].set_title("Real CV Data (Oxidation Scan)")
axes[0, 0].set_xlabel("Voltage (V)")
axes[0, 0].set_ylabel("Current (µA)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(0.6, -0.6)

# Generated data - full
axes[0, 1].plot(v_syn_ox, i_syn_ox, 'r-', lw=1.5, label='Generated (Randles-Ševčík)')
axes[0, 1].set_title("Generated CV Data (Oxidation Scan)")
axes[0, 1].set_xlabel("Voltage (V)")
axes[0, 1].set_ylabel("Current (µA)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(0.6, -0.6)

# Overlay
axes[1, 0].plot(v_real_ox, i_real_ox, 'b-', lw=1.5, alpha=0.7, label='Real')
axes[1, 0].plot(v_syn_ox, i_syn_ox, 'r--', lw=1.5, alpha=0.7, label='Generated')
axes[1, 0].set_title("Overlay Comparison")
axes[1, 0].set_xlabel("Voltage (V)")
axes[1, 0].set_ylabel("Current (µA)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(0.6, -0.6)

# Normalized comparison (to see shapes)
i_real_norm = (i_real_ox - i_real_ox.min()) / (i_real_ox.max() - i_real_ox.min())
i_syn_norm = (i_syn_ox - i_syn_ox.min()) / (i_syn_ox.max() - i_syn_ox.min())

axes[1, 1].plot(v_real_ox, i_real_norm, 'b-', lw=1.5, alpha=0.7, label='Real (normalized)')
axes[1, 1].plot(v_syn_ox, i_syn_norm, 'r--', lw=1.5, alpha=0.7, label='Generated (normalized)')
axes[1, 1].set_title("Shape Comparison (Normalized)")
axes[1, 1].set_xlabel("Voltage (V)")
axes[1, 1].set_ylabel("Normalized Current")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim(0.6, -0.6)

plt.tight_layout()
plt.savefig("outputs/images/real_vs_generated_comparison.png", dpi=200)
print("Saved: outputs/images/real_vs_generated_comparison.png")

print("\n=== Analysis ===")
print(f"Real data range: {i_real_ox.min():.1f} to {i_real_ox.max():.1f} µA")
print(f"Generated data range: {i_syn_ox.min():.1f} to {i_syn_ox.max():.1f} µA")
print("\nObservations needed:")
print("- Peak shape (sharp rise vs gradual decay)")
print("- Baseline curvature")
print("- Current magnitude scaling")
print("- Asymmetry of peaks")
plt.show()
