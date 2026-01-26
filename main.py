import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import glob
import os
import numpy as np

from src.ingestion import process_file
from src.ai_brain import PeakTuner

def main():
    try:
        tuner = PeakTuner()
    except Exception as e:
        print(e)
        return

    data_files = glob.glob("data/*.csv")
    print(f"Found {len(data_files)} files.")

    for fp in data_files:
        filename = os.path.basename(fp)
        print(f"\nProcessing {filename}...")
        
        try:
            data = process_file(fp)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        # Extract all necessary signals
        x_ox = data['x_ox']
        sig_ox = data['sig_ox']
        x_red = data['x_red']
        sig_red = data['sig_red']

        # --- AI Prediction for BOTH scans ---
        # The AI was trained on positive peaks, so we use abs() for the reduction signal
        prom_ox = tuner.predict_prominence(sig_ox)
        prom_red = tuner.predict_prominence(np.abs(sig_red))
        
        print(f"  🤖 AI Prominence -> Ox: {prom_ox:.2f} | Red: {prom_red:.2f}")

        # --- Peak Detection for BOTH scans ---
        # Find peaks on the positive oxidation signal
        peaks_ox, _ = find_peaks(sig_ox, prominence=prom_ox, width=10)
        
        # Find troughs (negative peaks) on the reduction signal by inverting it
        peaks_red, _ = find_peaks(-sig_red, prominence=prom_red, width=10)
        
        print(f"  ✅ Found {len(peaks_ox)} Ox peaks, {len(peaks_red)} Red peaks.")
        
        # --- VISUALIZATION (Restoring the 3-Panel Concatenated View) ---
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        # Panel 1: Baseline Fitting
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(x_ox, data['ox_scan']['I_uA'], color='blue', alpha=0.3, label='Raw Ox')
        ax1.plot(x_ox, data['base_ox'], color='blue', linestyle='--', label='Baseline Ox')
        ax1.plot(x_red, data['red_scan']['I_uA'], color='green', alpha=0.3, label='Raw Red')
        ax1.plot(x_red, data['base_red'], color='green', linestyle='--', label='Baseline Red')
        ax1.set_title("1. Baseline Fitting")
        ax1.set_xlabel("Voltage (V)")
        ax1.set_ylabel("Current (uA)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Extracted Signals (Overlaid)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(x_ox, sig_ox, color='blue', label='Oxidation')
        ax2.fill_between(x_ox, sig_ox, 0, color='blue', alpha=0.1)
        ax2.plot(x_red, sig_red, color='green', label='Reduction')
        ax2.fill_between(x_red, sig_red, 0, color='green', alpha=0.1)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_title("2. Extracted Signals")
        ax2.set_xlabel("Voltage (V)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Concatenated Full Cycle
        ax3 = fig.add_subplot(gs[1, :]) # Spans both columns
        
        # Create a continuous index for plotting
        idx_ox = np.arange(len(x_ox))
        idx_red = np.arange(len(x_ox), len(x_ox) + len(x_red))
        
        ax3.plot(idx_ox, sig_ox, color='blue', label='Forward Scan')
        ax3.fill_between(idx_ox, sig_ox, 0, color='blue', alpha=0.2)
        
        ax3.plot(idx_red, sig_red, color='green', label='Reverse Scan')
        ax3.fill_between(idx_red, sig_red, 0, color='green', alpha=0.2)
        
        # Plot the detected peaks on the concatenated axis
        ax3.plot(idx_ox[peaks_ox], sig_ox[peaks_ox], "rx", markersize=12, markeredgewidth=2)
        ax3.plot(idx_red[peaks_red], sig_red[peaks_red], "rx", markersize=12, markeredgewidth=2)
        
        ax3.axvline(len(x_ox), color='red', linestyle='--', label='Switching Point')
        ax3.set_title(f"3. Concatenated Full Cycle (Peaks: {len(peaks_ox)} Ox, {len(peaks_red)} Red)")
        ax3.set_xlabel("Data Point Index (Time)")
        ax3.set_ylabel("Corrected Current (uA)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f"Full Pipeline Analysis: {filename}", fontsize=16)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()