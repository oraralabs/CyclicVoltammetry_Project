import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import glob
import os
import numpy as np

from src.ingestion import process_file
from src.ai_brain import PeakTuner

def main():
    tuner = PeakTuner()
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

        # Focus on Oxidation for the 3-panel view
        x = data['x_ox']
        y = data['sig_ox']

        # 1. AI
        prom = tuner.predict_prominence(y)
        print(f"  🤖 AI Prominence: {prom:.4f}")
        
        # 2. Peak Find (Simple)
        peaks, _ = find_peaks(y, prominence=prom, width=10)

        # 3. Visualization (The 3 Charts)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Chart 1: Raw + Baseline
        ax1.plot(x, data['ox_scan']['I_uA'], color='gray', alpha=0.5, label='Raw')
        ax1.plot(x, data['base_ox'], color='blue', linestyle='--', label='Linear 1-99%')
        ax1.set_title("1. Baseline Fit")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Extracted Signal
        ax2.plot(x, y, 'b-', label='Extracted')
        ax2.axhline(0, color='k', linewidth=0.5)
        ax2.set_title("2. Extracted Signal")
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: AI Detection
        ax3.plot(x, y, 'k-', alpha=0.5)
        ax3.plot(x[peaks], y[peaks], "rx", markersize=12, label=f'Peaks (P={prom:.2f})')
        ax3.set_title(f"3. AI Detection (Peaks: {len(peaks)})")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.suptitle(f"File: {filename}", fontsize=14)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()