import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from src.cv_simulator import CVPhysicsSimulator
from src.ingestion import process_file
from src.ai_brain import PeakTuner
from src.deconvolution import SparseDeconvolution, consolidate_peaks

def run_validation():
    # 1. Define the Ground Truth (The Answer Key)
    # We place Species 1 and 2 very close together (-0.40 and -0.28)
    # and Species 3 further away (+0.10)
    truth = [
        {'E0': -0.40, 'height': 20.0, 'width': 0.10},
        {'E0': -0.28, 'height': 15.0, 'width': 0.10},
        {'E0':  0.10, 'height': 10.0, 'width': 0.10}
    ]

    # 2. Generate Simulated Data
    sim = CVPhysicsSimulator()
    df = sim.generate_complex_cv(truth, noise_level=0.2)
    
    temp_file = "validation_test.csv"
    with open(temp_file, 'w') as f:
        f.write("Validation Test\n"*10 + "V,uA\n")
        df.to_csv(f, index=False, header=False)

    # 3. Run Pipeline
    tuner = PeakTuner()
    solver = SparseDeconvolution(basis_width=0.10)
    data = process_file(temp_file)
    x, y = data['ox_x'], data['ox_sig']

    # 4. Deconvolute
    prom = tuner.predict_prominence(y)
    dist, _, centers = solver.solve(x, y, alpha=prom * 0.1)
    detected = consolidate_peaks(dist, centers)

    # 5. Calculate Accuracy
    print("\n--- PERFORMANCE METRICS ---")
    for i, t_peak in enumerate(truth):
        # Find the closest detected peak
        dists = [abs(p['v'] - t_peak['E0']) for p in detected]
        if dists and min(dists) < 0.08:
            idx = np.argmin(dists)
            pos_err = dists[idx] * 1000 # convert to mV
            mag_err = abs(detected[idx]['mag'] - t_peak['height']) / t_peak['height'] * 100
            print(f"Species {i+1} ({t_peak['E0']}V): FOUND | Pos Error: {pos_err:.1f} mV | Mag Error: {mag_err:.1f}%")
        else:
            print(f"Species {i+1} ({t_peak['E0']}V): MISSED")

    # Cleanup
    os.remove(temp_file)

if __name__ == "__main__":
    run_validation()