import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

from src.cv_simulator import CVPhysicsSimulator
from src.ingestion import process_file
from src.ai_brain import PeakTuner

def run_pipeline_on_file(filepath):
    tuner = PeakTuner()
    filename = os.path.basename(filepath)
    print(f"\n--- Running Pipeline on {filename} ---")
    
    try:
        data = process_file(filepath)
    except Exception as e:
        print(f"  Error: {e}")
        return

    # 1. Extract Individual Signals
    x_ox, sig_ox = data['ox_x'], data['ox_sig']
    x_red, sig_red = data['red_x'], data['red_sig']

    # 2. AI Prediction
    prom_ox = tuner.predict_prominence(sig_ox)
    prom_red = tuner.predict_prominence(np.abs(sig_red))

    # 3. Peak Finding
    peaks_ox, _ = find_peaks(sig_ox, prominence=prom_ox, width=10)
    peaks_red, _ = find_peaks(-sig_red, prominence=prom_red, width=10)

    # 4. Concatenation (For the 3rd Panel)
    # This joins the two halves into one continuous array
    full_signal = np.concatenate([sig_ox, sig_red])
    full_indices = np.arange(len(full_signal))
    
    # Adjust peak indices for the concatenated array
    # Ox peaks stay the same, Red peaks are shifted by the length of the Ox array
    concat_peaks_ox = peaks_ox
    concat_peaks_red = peaks_red + len(sig_ox)

    # --- VISUALIZATION ---
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 1)

    # Panel 1: Independent Baseline Fitting
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x_ox, data['ox_y_raw'], color='blue', alpha=0.3, label='Raw Ox')
    ax1.plot(x_ox, data['ox_base'], 'b--', label='Baseline Ox')
    ax1.plot(x_red, data['red_y_raw'], color='green', alpha=0.3, label='Raw Red')
    ax1.plot(x_red, data['red_base'], 'g--', label='Baseline Red')
    ax1.set_title("1. Independent Baseline Fitting")
    ax1.legend()

    # Panel 2: Extracted Signals (Overlaid)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x_ox, sig_ox, label='Oxidation Signal', color='blue')
    ax2.plot(x_red, sig_red, label='Reduction Signal', color='green')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_title("2. Extracted Signals (Non-Mirrored)")
    ax2.legend()

    # Panel 3: Concatenated Full Cycle (Input for ML)
    ax3 = fig.add_subplot(gs[2, 0])
    # Plotting against index (Time) instead of Voltage to show concatenation
    ax3.plot(full_indices[:len(sig_ox)], sig_ox, color='blue', label='Forward Scan')
    ax3.plot(full_indices[len(sig_ox):], sig_red, color='green', label='Reverse Scan')
    
    # Plot all detected peaks
    ax3.plot(concat_peaks_ox, full_signal[concat_peaks_ox], "rx", markersize=12, label='Detected Peaks')
    ax3.plot(concat_peaks_red, full_signal[concat_peaks_red], "rx", markersize=12)
    
    ax3.axvline(len(sig_ox), color='red', linestyle='--', label='Switching Point')
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.set_title(f"3. Concatenated Full Cycle (Found {len(peaks_ox)} Ox, {len(peaks_red)} Red)")
    ax3.set_xlabel("Data Point Index (Time)")
    ax3.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure simulator exists and species are defined
    simulator = CVPhysicsSimulator()
    species_definitions = [
        { 'E0': -0.4, 'height': 25.0, 'width': 0.08 },
        { 'E0': -0.2, 'height': 15.0, 'width': 0.12 },
    ]
    df_simulated = simulator.generate_complex_cv(species_definitions, noise_level=0.3)

    temp_filepath = "temp_simulated_data.csv"
    with open(temp_filepath, 'w') as f:
        f.write("Simulator Output\n" * 10)
        f.write("V,uA\n")
        df_simulated.to_csv(f, index=False, header=False)

    run_pipeline_on_file(temp_filepath)
    if os.path.exists(temp_filepath):
        os.remove(temp_filepath)