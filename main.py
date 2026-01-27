import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from src.ingestion import process_file
from src.ai_brain import PeakTuner
from src.deconvolution import SparseDeconvolution, consolidate_peaks

def main():
    tuner = PeakTuner()
    solver = SparseDeconvolution(basis_width=0.10)
    data_files = glob.glob("data/*.csv")

    for fp in data_files:
        filename = os.path.basename(fp)
        print(f"\n--- Analyzing Full Cycle: {filename} ---")
        try:
            data = process_file(fp)
        except: continue

        ox_x, ox_sig = data['ox_x'], data['ox_sig']
        red_x, red_sig = data['red_x'], data['red_sig']

        # AI Prediction & Solver for Oxidation
        prom_ox = tuner.predict_prominence(ox_sig)
        dist_ox, _, centers_ox = solver.solve(ox_x, ox_sig, alpha=prom_ox * 0.1)
        peaks_ox = consolidate_peaks(dist_ox, centers_ox, ox_x)

        # AI Prediction & Solver for Reduction (Inverted for Lasso)
        prom_red = tuner.predict_prominence(np.abs(red_sig))
        dist_red, _, centers_red = solver.solve(red_x, -red_sig, alpha=prom_red * 0.1)
        peaks_red = consolidate_peaks(dist_red, centers_red, red_x)

        # --- PLOT 1: 3-PANEL PIPELINE (OX + RED) ---
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 1)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(ox_x, data['ox_y_raw'], 'b', alpha=0.3, label='Raw Ox')
        ax1.plot(ox_x, data['ox_base'], 'b--', label='Base Ox')
        ax1.plot(red_x, data['red_y_raw'], 'g', alpha=0.3, label='Raw Red')
        ax1.plot(red_x, data['red_base'], 'g--', label='Base Red')
        ax1.set_title("1. Independent Baseline Fitting")
        ax1.legend()

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(ox_x, ox_sig, 'b', label='Oxidation')
        ax2.plot(red_x, red_sig, 'g', label='Reduction')
        ax2.axhline(0, color='k', lw=0.5)
        ax2.set_title("2. Extracted Signals (Overlaid)")

        ax3 = fig.add_subplot(gs[2, 0])
        idx_ox = np.arange(len(ox_sig))
        idx_red = np.arange(len(ox_sig), len(ox_sig) + len(red_sig))
        ax3.plot(idx_ox, ox_sig, 'b', label='Forward Scan')
        ax3.plot(idx_red, red_sig, 'g', label='Reverse Scan')
        
        # Mark Ox Peaks
        for p in peaks_ox:
            p_idx = np.argmin(np.abs(ox_x - p['v']))
            ax3.plot(idx_ox[p_idx], ox_sig[p_idx], 'rx', ms=12, mew=2)
            ax3.text(idx_ox[p_idx], ox_sig[p_idx]+1, f"{p['v']:.2f}V", color='red', ha='center')
        
        # Mark Red Peaks (sig_red is negative)
        for p in peaks_red:
            p_idx = np.argmin(np.abs(red_x - p['v']))
            ax3.plot(idx_red[p_idx], red_sig[p_idx], 'rx', ms=12, mew=2)
            ax3.text(idx_red[p_idx], red_sig[p_idx]-2, f"{p['v']:.2f}V", color='red', ha='center')

        ax3.axvline(len(ox_sig), color='red', ls='--')
        ax3.set_title(f"3. Concatenated Full Cycle (Total Peaks: {len(peaks_ox) + len(peaks_red)})")
        plt.tight_layout()

        # --- PLOT 2: DECONVOLUTED SPECTRUM (OXIDATION) ---
        fig_recon, ax_recon = plt.subplots(figsize=(10, 6))
        sigma = 0.10 / 2.355
        colors = plt.cm.viridis(np.linspace(0, 1, len(peaks_ox)))
        for i, p in enumerate(peaks_ox):
            curve = p['mag'] * np.exp(-0.5 * ((ox_x - p['v']) / sigma)**2)
            ax_recon.fill_between(ox_x, curve, alpha=0.4, color=colors[i], label=f"Species @ {p['v']:.2f}V")
            ax_recon.plot(ox_x, curve, color=colors[i], lw=1)
        ax_recon.set_title(f"Quantified Species Spectrum: {filename}")
        ax_recon.legend()
        plt.show()

if __name__ == "__main__":
    main()