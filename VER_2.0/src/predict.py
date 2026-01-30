"""
VER 2.0 - Local Inference
=========================
Load trained model and predict peaks from new signals.
"""
import numpy as np
from typing import List, Dict, Tuple
from scipy.signal import find_peaks


class PeakPredictor:
    """
    Load trained U-Net model and predict Gaussian peak parameters.
    """
    
    def __init__(self, model_path: str, norm_path: str):
        """
        Args:
            model_path: Path to trained .keras model
            norm_path: Path to normalization_params.npz
        """
        import tensorflow as tf
        
        self.model = tf.keras.models.load_model(model_path)
        
        norm_data = np.load(norm_path)
        self.signal_mean = norm_data['signal_mean']
        self.signal_std = norm_data['signal_std']
        self.voltage_grid = norm_data['voltage_grid']
        
        print(f"Model loaded: {model_path}")
    
    def preprocess(self, signal: np.ndarray) -> np.ndarray:
        """Normalize and reshape signal for model input."""
        signal_norm = (signal - self.signal_mean) / self.signal_std
        return signal_norm.reshape(1, -1, 1)
    
    def predict_heatmap(self, signal: np.ndarray) -> np.ndarray:
        """Predict peak probability heatmap from signal."""
        x = self.preprocess(signal)
        heatmap = self.model.predict(x, verbose=0).squeeze()
        return heatmap
    
    def extract_peaks(self, heatmap: np.ndarray, 
                      threshold: float = 0.3,
                      min_distance: int = 10) -> List[Dict]:
        """
        Extract peak locations from predicted heatmap.
        
        Args:
            heatmap: Predicted peak probability array
            threshold: Minimum probability to consider a peak
            min_distance: Minimum separation between peaks (in points)
            
        Returns:
            List of peak dicts with 'center' and 'confidence'
        """
        # Find local maxima above threshold
        peaks_idx, properties = find_peaks(heatmap, 
                                           height=threshold,
                                           distance=min_distance)
        
        peaks = []
        for idx in peaks_idx:
            peaks.append({
                'center': self.voltage_grid[idx],
                'confidence': heatmap[idx],
                'index': idx
            })
        
        # Sort by voltage
        peaks.sort(key=lambda x: x['center'])
        return peaks
    
    def predict(self, signal: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Full prediction pipeline.
        
        Args:
            signal: Input signal array (500 points)
            
        Returns:
            (heatmap, peaks_list)
        """
        heatmap = self.predict_heatmap(signal)
        peaks = self.extract_peaks(heatmap)
        return heatmap, peaks


# =============================================================================
# Test / Demo
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, 'src')
    from generator import SignalGenerator
    
    # Check if model exists
    import os
    model_path = 'models/peak_detector.keras'
    norm_path = 'models/normalization_params.npz'
    
    if not os.path.exists(model_path):
        print("=" * 60)
        print("MODEL NOT FOUND")
        print("=" * 60)
        print(f"\nExpected: {model_path}")
        print("\nTo get the trained model:")
        print("1. Open VER_2.0/notebooks/train_unet.ipynb in Google Colab")
        print("2. Upload VER_2.0/data/training_data.npz to Colab")
        print("3. Run all cells")
        print("4. Download peak_detector.keras and normalization_params.npz")
        print("5. Place them in VER_2.0/models/")
        print("\n" + "=" * 60)
    else:
        # Run inference demo
        predictor = PeakPredictor(model_path, norm_path)
        gen = SignalGenerator()
        
        fig, axes = plt.subplots(3, 3, figsize=(14, 10))
        fig.suptitle("VER 2.0 - Inference Demo", fontweight='bold')
        
        for row in range(3):
            signal, _, true_params = gen.generate_sample(n_peaks=row+1)
            heatmap, detected = predictor.predict(signal)
            
            # Plot signal
            axes[row, 0].plot(gen.voltage_grid, signal, 'b-')
            axes[row, 0].set_title(f"Input ({row+1} peaks)")
            axes[row, 0].grid(True, alpha=0.3)
            
            # Plot heatmap
            axes[row, 1].fill_between(gen.voltage_grid, heatmap, alpha=0.5, color='orange')
            axes[row, 1].set_title("Predicted Heatmap")
            axes[row, 1].set_ylim(0, 1.1)
            axes[row, 1].grid(True, alpha=0.3)
            
            # Plot comparison
            for p in true_params:
                axes[row, 2].axvline(p['center'], color='green', ls='--', lw=2, alpha=0.7, label='True')
            for p in detected:
                axes[row, 2].axvline(p['center'], color='red', ls='-', lw=2, alpha=0.7, label='Detected')
            axes[row, 2].set_title(f"True (green) vs Detected (red)")
            axes[row, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("inference_demo.png", dpi=150)
        print("Saved inference_demo.png")
        plt.show()
