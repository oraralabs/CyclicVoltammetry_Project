import matplotlib.pyplot as plt
import glob
import os
import numpy as np

from src.ingestion import process_file
from src.ai_brain import PeakTuner
from src.deconvolution import PeakResolver

def main():
    try:
        tuner = PeakTuner()
        resolver = PeakResolver()
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

        # --- PROCESS OXIDATION (Forward) ---
        x = data['x_ox']
        y = data['sig_ox']

        # 1. AI
        prom = tuner.predict_prominence(y)
        print(f"  🤖 AI Prominence: {prom:.4f}")

        # 2. Deconvolution
        peaks, result = resolver.fit_peaks(x, y, prom)
        print(f"  ✅ Resolved {len(peaks)} Peaks")

        # 3. Visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Raw + Baseline
        ax1.plot(x, data['ox_scan']['I_uA'], color='gray', alpha=0.5, label='Raw')
        ax1.plot(x, data['base_ox'], color='blue', linestyle='--', label='Poly Baseline')
        ax1.set_title("1. Baseline Fit (Poly deg=2)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Extracted Signal
        ax2.plot(x, y, 'b-', label='Extracted')
        ax2.axhline(0, color='k', linewidth=0.5)
        ax2.set_title("2. Signal (Flat?)")
        ax2.grid(True, alpha=0.3)
        
        # Deconvolution
        ax3.plot(x, y, 'k-', alpha=0.3)
        if result:
            ax3.plot(x, result.best_fit, 'r--', label='Fit')
            comps = result.eval_components(x=x)
            for name, val in comps.items():
                if 'g' in name:
                    ax3.fill_between(x, val, alpha=0.4, label=name)
        
        ax3.set_title(f"3. Deconvolution (Peaks: {len(peaks)})")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.suptitle(f"File: {filename}", fontsize=14)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()