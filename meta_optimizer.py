import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.cv_simulator import CVPhysicsSimulator
from src.deconvolution import SparseDeconvolution, consolidate_peaks
from src.ai_brain import PeakTuner
import os

def calculate_accuracy(truth_peaks, detected_peaks):
    """
    Compares detected peaks to ground truth.
    Returns: Average Position Error (mV) and Peak Count Error.
    """
    if len(detected_peaks) == 0:
        return 500, len(truth_peaks) # High penalty for finding nothing

    pos_errors = []
    # Match each truth peak to the closest detected peak
    for t in truth_peaks:
        dists = [abs(p['v'] - t['E0']) for p in detected_peaks]
        min_dist = min(dists)
        if min_dist < 0.1: # Only count if within 100mV
            pos_errors.append(min_dist * 1000) # Store in mV
    
    count_error = abs(len(truth_peaks) - len(detected_peaks))
    avg_pos_error = np.mean(pos_errors) if pos_errors else 200
    
    return avg_pos_error, count_error

def run_parameter_sweep(n_trials=20):
    simulator = CVPhysicsSimulator()
    tuner = PeakTuner()
    
    # --- The Search Grid ---
    # We test different Alpha multipliers and Basis Widths
    alpha_multipliers = [0.01, 0.05, 0.1, 0.2, 0.5]
    basis_widths = [0.05, 0.08, 0.10, 0.12, 0.15]
    
    results = []

    print(f"🚀 Starting Meta-Optimization Sweep ({len(alpha_multipliers) * len(basis_widths)} combos)...")

    for bw in basis_widths:
        solver = SparseDeconvolution(basis_width=bw)
        for am in alpha_multipliers:
            trial_pos_errors = []
            trial_count_errors = []
            
            # Run multiple random trials for this specific setting to get an average
            for i in range(n_trials):
                # 1. Generate known truth (2 or 3 overlapping peaks)
                truth = [
                    {'E0': np.random.uniform(-0.5, -0.2), 'height': 20.0, 'width': 0.1},
                    {'E0': np.random.uniform(-0.1, 0.2), 'height': 15.0, 'width': 0.1}
                ]
                df_sim = simulator.generate_complex_cv(truth, noise_level=0.3)
                
                # 2. Run Pipeline (Extraction is skipped here to focus on Decon accuracy)
                x = df_sim['E_V'].values[:1000] # Forward scan
                y = df_sim['I_uA'].values[:1000] 
                # (Normally we would subtract baseline, but simulator gives us pure signal + noise)
                
                # 3. Predict & Solve
                prom = tuner.predict_prominence(y)
                dist, _, centers = solver.solve(x, y, alpha=prom * am)
                detected = consolidate_peaks(dist, centers)
                
                # 4. Score
                p_err, c_err = calculate_accuracy(truth, detected)
                trial_pos_errors.append(p_err)
                trial_count_errors.append(c_err)
            
            results.append({
                'Basis_Width': bw,
                'Alpha_Mult': am,
                'Avg_Pos_Error': np.mean(trial_pos_errors),
                'Avg_Count_Error': np.mean(trial_count_errors),
                'Total_Score': np.mean(trial_pos_errors) + (np.mean(trial_count_errors) * 50)
            })
            print(f"  Tested: Width={bw}, AlphaMult={am} -> Score: {results[-1]['Total_Score']:.1f}")

    return pd.DataFrame(results)

# --- EXECUTION ---
if __name__ == "__main__":
    df_results = run_parameter_sweep(n_trials=10)
    
    # Visualize results to find the 'Safe Zone'
    pivot = df_results.pivot(index="Alpha_Mult", columns="Basis_Width", values="Total_Score")
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, cmap="viridis_r")
    plt.title("Meta-Optimization: Finding lowest error settings")
    plt.show()

    best_idx = df_results['Total_Score'].idxmin()
    best = df_results.iloc[best_idx]
    print("\n--- WINNING SETTINGS FOUND ---")
    print(f"Optimal Basis Width:    {best['Basis_Width']}")
    print(f"Optimal Alpha Mult:     {best['Alpha_Mult']}")
    print(f"Resulting Avg Error:    {best['Avg_Pos_Error']:.2f} mV")