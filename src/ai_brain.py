import numpy as np
import pandas as pd
import joblib
import os

class PeakTuner:
    def __init__(self, model_path='models/neotrient_brain_precision.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing Model! Put it here: {model_path}")
        
        self.model = joblib.load(model_path)
        print(f"🧠 AI Brain loaded.")

    def extract_features(self, current_array):
        """
        MUST match the logic used in Colab training exactly.
        """
        diffs = np.diff(current_array)
        mad = np.median(np.abs(diffs - np.median(diffs)))
        
        return pd.DataFrame([{
            'Feature_StdDev': np.std(current_array),
            'Feature_MAD': mad,
            'Feature_Max': np.max(current_array)
        }])

    def predict_prominence(self, current_array):
        feats = self.extract_features(current_array)
        prediction = self.model.predict(feats)[0]
        return prediction