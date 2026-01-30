"""
Analyze Real CV Data Structure
===============================
Plot the actual CV curve to understand what it should look like.
"""
import numpy as np
import matplotlib.pyplot as plt
import subprocess

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

# Load real CV data
v, i = parse_real_cv('data/PureHydration_9Jan.csv')

print(f"Total points: {len(v)}")
print(f"Voltage range: {v.min():.3f} to {v.max():.3f} V")
print(f"Current range: {i.min():.1f} to {i.max():.1f} µA")

# Find the turning point (where scan reverses)
# Look for where voltage direction changes
dv = np.diff(v)
turn_idx = None
for idx in range(1, len(dv)-1):
    if dv[idx-1] * dv[idx] < 0:  # Sign change
        turn_idx = idx
        break

if turn_idx is None:
    # No clear turning point, assume midpoint
    turn_idx = len(v) // 2

print(f"Turning point at index: {turn_idx}, V = {v[turn_idx]:.3f} V")

# Split into forward and reverse
v_fwd = v[:turn_idx]
i_fwd = i[:turn_idx]
v_rev = v[turn_idx:]
i_rev = i[turn_idx:]

print(f"Forward scan: {len(v_fwd)} points")
print(f"Reverse scan: {len(v_rev)} points")

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))

# Full CV curve (classic view)
ax1 = plt.subplot(2, 3, 1)
ax1.plot(v, i, 'b-', lw=1.5)
ax1.set_xlabel("Voltage (V)")
ax1.set_ylabel("Current (µA)")
ax1.set_title("Full CV Curve (as recorded)")
ax1.grid(True, alpha=0.3)
ax1.axvline(v[turn_idx], color='red', ls='--', alpha=0.5, label='Turn point')
ax1.legend()

# Forward scan only
ax2 = plt.subplot(2, 3, 2)
ax2.plot(v_fwd, i_fwd, 'b-', lw=1.5)
ax2.set_xlabel("Voltage (V)")
ax2.set_ylabel("Current (µA)")
ax2.set_title(f"Forward Scan (Oxidation): {v_fwd[0]:.2f}V → {v_fwd[-1]:.2f}V")
ax2.grid(True, alpha=0.3)

# Reverse scan only
ax3 = plt.subplot(2, 3, 3)
ax3.plot(v_rev, i_rev, 'r-', lw=1.5)
ax3.set_xlabel("Voltage (V)")
ax3.set_ylabel("Current (µA)")
ax3.set_title(f"Reverse Scan (Reduction): {v_rev[0]:.2f}V → {v_rev[-1]:.2f}V")
ax3.grid(True, alpha=0.3)

# Overlay forward and reverse
ax4 = plt.subplot(2, 3, 4)
ax4.plot(v_fwd, i_fwd, 'b-', lw=1.5, alpha=0.7, label='Forward (ox)')
ax4.plot(v_rev, i_rev, 'r-', lw=1.5, alpha=0.7, label='Reverse (red)')
ax4.set_xlabel("Voltage (V)")
ax4.set_ylabel("Current (µA)")
ax4.set_title("Overlay: Forward vs Reverse")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Time series view
ax5 = plt.subplot(2, 3, 5)
time = np.arange(len(v))
ax5.plot(time, v, 'k-', lw=1, label='Voltage')
ax5.set_xlabel("Time (point index)")
ax5.set_ylabel("Voltage (V)", color='k')
ax5.tick_params(axis='y', labelcolor='k')
ax5.grid(True, alpha=0.3)

ax5b = ax5.twinx()
ax5b.plot(time, i, 'b-', lw=1, alpha=0.7, label='Current')
ax5b.set_ylabel("Current (µA)", color='b')
ax5b.tick_params(axis='y', labelcolor='b')
ax5.set_title("Time Series: V and I vs Time")

# Current vs Voltage (hysteresis loop)
ax6 = plt.subplot(2, 3, 6)
# Plot as a continuous line showing the hysteresis
ax6.plot(v_fwd, i_fwd, 'b-', lw=2, label='Forward', alpha=0.8)
ax6.plot(v_rev, i_rev, 'r-', lw=2, label='Reverse', alpha=0.8)
# Add arrows to show direction
n_arrows = 5
for idx in np.linspace(10, len(v_fwd)-10, n_arrows).astype(int):
    ax6.annotate('', xy=(v_fwd[idx], i_fwd[idx]), 
                xytext=(v_fwd[idx-5], i_fwd[idx-5]),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
for idx in np.linspace(10, len(v_rev)-10, n_arrows).astype(int):
    ax6.annotate('', xy=(v_rev[idx], i_rev[idx]), 
                xytext=(v_rev[idx-5], i_rev[idx-5]),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

ax6.set_xlabel("Voltage (V)")
ax6.set_ylabel("Current (µA)")
ax6.set_title("CV Hysteresis Loop (I vs V)")
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/images/real_cv_detailed_analysis.png", dpi=200)
print("\nSaved: outputs/images/real_cv_detailed_analysis.png")
print("\n=== Key Observations ===")
print("This is what a REAL CV curve looks like!")
plt.show()
