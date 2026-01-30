"""
VER 2.0 Pipeline Validation Demo
==================================
Generate 9x9 grid of synthetic CV curves, run full pipeline, and validate
peak detection accuracy.

This demonstrates:
1. CV generation (MechanisticSimulator)
2. U-Net peak detection (heatmap prediction)
3. Gaussian fitting (parameter extraction)
4. Accuracy assessment (detected vs ground truth)
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, 'src')

from generator_cv import CVDataGenerator
from gaussian_fitter import GaussianFitter
import tensorflow as tf

# Suppress TF warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class PipelineValidator:
    """
    End-to-end validation of VER 2.0 pipeline.
    """
    
    def __init__(self):
        # Load trained model
        self.model = tf.keras.models.load_model('models/peak_detector_cv.keras')
        norm = np.load('models/normalization_params_cv.npz')
        self.signal_mean = norm['signal_mean']
        self.signal_std = norm['signal_std']
        self.voltage_grid = norm['voltage_grid']
        
        # Initialize components
        self.generator = CVDataGenerator()
        self.fitter = GaussianFitter(self.voltage_grid)
        
        print("✓ Loaded model and components")
    
    def run_single_analysis(self, n_peaks=None):
        """
        Generate one CV curve and run full pipeline.
        
        Returns:
        --------
        results : dict with keys:
            - signal: Generated signal
            - heatmap_true: Ground truth heatmap
            - heatmap_pred: Predicted heatmap
            - params_true: True peak parameters
            - params_detected: Detected peak parameters
            - accuracy: Detection accuracy metrics
        """
        # Generate synthetic CV
        signal, heatmap_true, params_true = self.generator.generate_sample(n_peaks=n_peaks)
        
        # Normalize for model
        signal_norm = (signal - self.signal_mean) / self.signal_std
        
        # Predict heatmap
        heatmap_pred = self.model.predict(signal_norm.reshape(1, -1, 1), verbose=0).squeeze()
        
        # Fit Gaussians
        params_detected = self.fitter.fit_all_peaks(signal, heatmap_pred, threshold=0.2)
        
        # Calculate accuracy
        accuracy = self._calculate_accuracy(params_true, params_detected)
        
        return {
            'signal': signal,
            'heatmap_true': heatmap_true,
            'heatmap_pred': heatmap_pred,
            'params_true': params_true,
            'params_detected': params_detected,
            'accuracy': accuracy
        }
    
    def _calculate_accuracy(self, true_params, detected_params):
        """
        Calculate detection accuracy metrics.
        """
        n_true = len(true_params)
        n_detected = len(detected_params)
        
        # Peak count accuracy
        count_correct = (n_true == n_detected)
        
        # Position accuracy (if same count)
        position_errors = []
        if count_correct and n_true > 0:
            # Match detected to true (closest pairs)
            true_positions = [p['E0'] for p in true_params]
            det_positions = [p['center'] for p in detected_params]
            
            for t_pos in true_positions:
                closest_det = min(det_positions, key=lambda d: abs(d - t_pos))
                error = abs(closest_det - t_pos)
                position_errors.append(error)
        
        return {
            'count_correct': count_correct,
            'n_true': n_true,
            'n_detected': n_detected,
            'position_errors': position_errors,
            'mean_position_error': np.mean(position_errors) if position_errors else None
        }
    
    def validate_grid(self, grid_size=9, save_path='outputs/images/validation_grid_9x9.png'):
        """
        Generate grid of test cases and validate.
        
        Parameters:
        -----------
        grid_size : Size of grid (default 9x9 = 81 test cases)
        save_path : Where to save visualization
        """
        print(f"\nRunning {grid_size}x{grid_size} validation grid...")
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        fig.suptitle(f"VER 2.0 Pipeline Validation: {grid_size}×{grid_size} Test Cases", 
                     fontweight='bold', fontsize=16)
        
        # Statistics
        results_summary = {
            'total': 0,
            'count_correct': 0,
            'position_errors': [],
            'by_n_peaks': {1: {'total': 0, 'correct': 0}, 
                          2: {'total': 0, 'correct': 0},
                          3: {'total': 0, 'correct': 0}}
        }
        
        for row in range(grid_size):
            for col in range(grid_size):
                ax = axes[row, col]
                
                # Deterministic peak count cycling
                n_peaks = (row * grid_size + col) % 3 + 1
                
                # Run analysis
                result = self.run_single_analysis(n_peaks=n_peaks)
                
                # Update statistics
                results_summary['total'] += 1
                acc = result['accuracy']
                if acc['count_correct']:
                    results_summary['count_correct'] += 1
                    results_summary['by_n_peaks'][n_peaks]['correct'] += 1
                results_summary['by_n_peaks'][n_peaks]['total'] += 1
                
                if acc['position_errors']:
                    results_summary['position_errors'].extend(acc['position_errors'])
                
                # Plot
                ax.plot(self.voltage_grid, result['signal'], 'b-', lw=0.8, alpha=0.7)
                ax.fill_between(self.voltage_grid, result['heatmap_pred'], 
                               alpha=0.3, color='orange')
                
                # Mark peaks
                for p in result['params_detected']:
                    ax.axvline(p['center'], color='red', ls='--', alpha=0.6, lw=0.8)
                
                # Title with accuracy
                color = 'green' if acc['count_correct'] else 'red'
                ax.set_title(f"T:{acc['n_true']} D:{acc['n_detected']}", 
                           fontsize=7, color=color)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(self.voltage_grid.max(), self.voltage_grid.min())
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved visualization: {save_path}")
        
        # Print summary
        self._print_summary(results_summary)
        
        return results_summary
    
    def _print_summary(self, results):
        """
        Print validation summary statistics.
        """
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        
        total = results['total']
        correct = results['count_correct']
        accuracy = 100 * correct / total if total > 0 else 0
        
        print(f"\nOverall Peak Count Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        
        print("\nBreakdown by Number of Peaks:")
        for n in [1, 2, 3]:
            stats = results['by_n_peaks'][n]
            if stats['total'] > 0:
                acc = 100 * stats['correct'] / stats['total']
                print(f"  {n} peak(s): {stats['correct']}/{stats['total']} ({acc:.1f}%)")
        
        if results['position_errors']:
            errors = np.array(results['position_errors'])
            print(f"\nPosition Accuracy (when count correct):")
            print(f"  Mean error: {errors.mean()*1000:.1f} mV")
            print(f"  Std error:  {errors.std()*1000:.1f} mV")
            print(f"  Max error:  {errors.max()*1000:.1f} mV")
        
        print("="*60 + "\n")


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("VER 2.0 Pipeline Validation Demo")
    print("=" * 60)
    
    # Initialize validator
    validator = PipelineValidator()
    
    # Run 9x9 grid validation
    results = validator.validate_grid(grid_size=9)
    
    print("\n✓ Validation complete!")
    print("  Check outputs/images/validation_grid_9x9.png")
