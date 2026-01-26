import numpy as np
import pandas as pd
from src.cv_simulator import CVPhysicsSimulator
from src.ai_brain import PeakTuner

def generate_physics_dataset(n_samples=500):
    simulator = CVPhysicsSimulator()
    tuner = PeakTuner()
    
    X_list = []
    y_list = []

    print(f"Generating {n_samples} training samples based on Randles-Sevcik physics...")

    for i in range(n_samples):
        # Randomize the number of peaks (1 to 3)
        num_peaks = np.random.randint(1, 4)
        species = []
        for _ in range(num_peaks):
            species.append({
                'E0': np.random.uniform(-0.6, 0.6),
                'height': np.random.uniform(5.0, 30.0),
                'width': np.random.uniform(0.05, 0.15)
            })

        # Generate the raw data with random noise
        noise_lvl = np.random.uniform(0.1, 0.5)
        df = simulator.generate_complex_cv(species, noise_level=noise_lvl)
        
        # We focus on the Oxidation scan for training
        # In a real device, we would train separate models for POS and NEG scans
        half = len(df) // 2
        ox_signal = df['I_uA'].values[:half]
        
        # Feature extraction
        feats = tuner.extract_features(ox_signal)
        
        # The 'True' target: mathematically, the prominence needs to be
        # slightly higher than the maximum noise spike to avoid false positives.
        # We 'cheat' here by knowing the true noise level from the simulator.
        target_prominence = noise_lvl * 4.5 

        X_list.append(feats)
        y_list.append(target_prominence)

    return pd.DataFrame(X_list), np.array(y_list)

if __name__ == "__main__":
    X, y = generate_physics_dataset(1000)
    tuner = PeakTuner()
    tuner.train_on_physics(X, y)