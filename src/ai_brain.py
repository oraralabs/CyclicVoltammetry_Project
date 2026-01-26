import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

class PeakTuner:
    def __init__(self, model_path='models/neotrient_brain_physics.pkl'):
        # Check for the physics model first, fall back to precision model if needed
        if not os.path.exists(model_path):
            model_path = 'models/neotrient_brain_precision.pkl'
            
        try:
            self.model = joblib.load(model_path)
            print(f"AI Brain loaded from {model_path}")
        except:
            self.model = RandomForestRegressor(n_estimators=100)
            print("New AI Brain initialized (requires training).")

    def extract_features(self, current_array):
        """
        Calculates descriptors for the signal. 
        Note: Standardized to 3 features to maintain model compatibility.
        """
        diffs = np.diff(current_array)
        mad = np.median(np.abs(diffs - np.median(diffs)))
        
        return {
            'Feature_StdDev': np.std(current_array),
            'Feature_MAD': mad,
            'Feature_Max': np.max(current_array)
        }

    def train_on_physics(self, X_data, y_labels):
        """
        Trains the model on realistic Randles-Sevcik data.
        """
        print("Training AI on physical simulation data...")
        self.model.fit(X_data, y_labels)
        # Ensure models folder exists
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump(self.model, 'models/neotrient_brain_physics.pkl')
        print("Training complete. Physics-aware model saved.")

    def predict_prominence(self, current_array):
        """
        Infers the best prominence threshold for the signal.
        """
        feats = pd.DataFrame([self.extract_features(current_array)])
        return self.model.predict(feats)[0]