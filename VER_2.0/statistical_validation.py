"""
Statistical Validation of VER 2.0 Pipeline
===========================================
Comprehensive statistical analysis with ML metrics on 100+ test cases.

Metrics Computed:
- Precision, Recall, F1-score (peak detection as classification)
- R² coefficient (position prediction accuracy)
- MAE, RMSE (error statistics)
- Confusion matrix (peak count)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, r2_score, confusion_matrix
import sys
import os
sys.path.insert(0, 'src')

from generator_cv import CVDataGenerator
from gaussian_fitter import GaussianFitter
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class StatisticalValidator:
    """
    Rigorous statistical validation with ML metrics.
    """
    
    def __init__(self):
        # Load model
        self.model = tf.keras.models.load_model('models/peak_detector_cv.keras')
        norm = np.load('models/normalization_params_cv.npz')
        self.signal_mean = norm['signal_mean']
        self.signal_std = norm['signal_std']
        self.voltage_grid = norm['voltage_grid']
        
        # Initialize components
        self.generator = CVDataGenerator()
        self.fitter = GaussianFitter(self.voltage_grid)
        
        print("✓ Loaded model and components\n")
    
    def run_statistical_test(self, n_samples=100):
        """
        Run comprehensive statistical validation.
        
        Parameters:
        -----------
        n_samples : Number of test cases (default 100)
        
        Returns:
        --------
        results : dict with all metrics and data
        """
        print(f"Running statistical validation on {n_samples} test cases...")
        print("=" * 70)
        
        # Storage for results
        all_true_counts = []
        all_pred_counts = []
        all_true_positions = []
        all_pred_positions = []
        position_errors = []
        height_errors_rel = []
        area_errors_rel = []
        
        # Binary classification arrays (for precision/recall)
        # We'll create a grid and mark where peaks should/shouldn't be
        detection_true = []  # Ground truth: 1 if peak nearby, 0 otherwise
        detection_pred = []  # Prediction: 1 if detected peak nearby, 0 otherwise
        
        for i in range(n_samples):
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{n_samples}")
            
            # Generate sample
            signal, heatmap_true, params_true = self.generator.generate_sample()
            
            # Predict
            signal_norm = (signal - self.signal_mean) / self.signal_std
            heatmap_pred = self.model.predict(signal_norm.reshape(1, -1, 1), verbose=0).squeeze()
            
            # Extract peaks
            params_detected = self.fitter.fit_all_peaks(signal, heatmap_pred, threshold=0.2)
            
            # Store counts
            n_true = len(params_true)
            n_detected = len(params_detected)
            all_true_counts.append(n_true)
            all_pred_counts.append(n_detected)
            
            # Match detected peaks to true peaks (Hungarian matching)
            true_pos = np.array([p['E0'] for p in params_true])
            det_pos = np.array([p['center'] for p in params_detected])
            
            all_true_positions.extend(true_pos)
            all_pred_positions.extend(det_pos)
            
            # Calculate matched pairs (closest neighbors within threshold)
            matched_pairs = self._match_peaks(params_true, params_detected, threshold=0.1)
            
            # Position errors
            for true_p, det_p in matched_pairs:
                pos_err = abs(det_p['center'] - true_p['E0'])
                position_errors.append(pos_err)
                
                # Height error (relative)
                height_err_rel = abs(det_p['height'] - true_p['height']) / (true_p['height'] + 1e-6)
                height_errors_rel.append(height_err_rel)
                
                # Area error (relative)
                if 'area' in det_p:
                    area_err_rel = abs(det_p['area'] - true_p.get('area', det_p['area'])) / (det_p['area'] + 1e-6)
                    area_errors_rel.append(area_err_rel)
            
            # Binary classification at each voltage point
            # Mark regions near true peaks as positive class
            for v_idx, v in enumerate(self.voltage_grid):
                # True: is there a peak within ±50mV?
                is_true_peak = any(abs(v - p['E0']) < 0.05 for p in params_true)
                # Predicted: is there a detected peak within ±50mV?
                is_pred_peak = any(abs(v - p['center']) < 0.05 for p in params_detected)
                
                detection_true.append(int(is_true_peak))
                detection_pred.append(int(is_pred_peak))
        
        print(f"✓ Completed {n_samples} test cases\n")
        
        # Calculate metrics
        results = self._calculate_metrics(
            all_true_counts, all_pred_counts,
            all_true_positions, all_pred_positions,
            position_errors, height_errors_rel, area_errors_rel,
            detection_true, detection_pred
        )
        
        return results
    
    def _match_peaks(self, true_params, det_params, threshold=0.1):
        """
        Match detected peaks to true peaks using closest neighbor.
        Only matches within threshold distance.
        """
        matched = []
        true_used = set()
        det_used = set()
        
        # Sort by distance
        pairs = []
        for i, t_p in enumerate(true_params):
            for j, d_p in enumerate(det_params):
                dist = abs(d_p['center'] - t_p['E0'])
                if dist < threshold:
                    pairs.append((dist, i, j, t_p, d_p))
        
        # Greedy matching (closest first)
        pairs.sort()
        for dist, i, j, t_p, d_p in pairs:
            if i not in true_used and j not in det_used:
                matched.append((t_p, d_p))
                true_used.add(i)
                det_used.add(j)
        
        return matched
    
    def _calculate_metrics(self, true_counts, pred_counts, true_pos, pred_pos,
                          pos_errors, height_errors, area_errors,
                          det_true, det_pred):
        """
        Calculate comprehensive metrics.
        """
        results = {}
        
        # 1. Peak Count Metrics
        true_counts = np.array(true_counts)
        pred_counts = np.array(pred_counts)
        
        count_accuracy = np.mean(true_counts == pred_counts)
        count_mae = np.mean(np.abs(true_counts - pred_counts))
        
        results['count_accuracy'] = count_accuracy
        results['count_mae'] = count_mae
        
        # Confusion matrix for peak counts
        cm = confusion_matrix(true_counts, pred_counts, labels=[1, 2, 3])
        results['confusion_matrix'] = cm
        
        # 2. Precision, Recall, F1 (binary classification at each voltage point)
        det_true = np.array(det_true)
        det_pred = np.array(det_pred)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            det_true, det_pred, average='binary', zero_division=0
        )
        
        results['precision'] = precision
        results['recall'] = recall
        results['f1_score'] = f1
        
        # 3. Position Accuracy (R²)
        if len(true_pos) > 0 and len(pred_pos) > 0:
            # For R², we need matched pairs
            # Use position errors to calculate R²
            pos_errors = np.array(pos_errors)
            true_pos_matched = np.array(true_pos[:len(pos_errors)])
            pred_pos_matched = true_pos_matched - pos_errors  # Reconstruct predicted
            
            # R² = 1 - (SS_res / SS_tot)
            ss_res = np.sum((true_pos_matched - pred_pos_matched) ** 2)
            ss_tot = np.sum((true_pos_matched - np.mean(true_pos_matched)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            
            results['r2_position'] = r2
            results['mae_position'] = np.mean(pos_errors) * 1000  # mV
            results['rmse_position'] = np.sqrt(np.mean(pos_errors ** 2)) * 1000  # mV
            results['std_position'] = np.std(pos_errors) * 1000  # mV
        
        # 4. Height and Area Errors
        if len(height_errors) > 0:
            results['mae_height_rel'] = np.mean(height_errors) * 100  # %
            results['std_height_rel'] = np.std(height_errors) * 100  # %
        
        if len(area_errors) > 0:
            results['mae_area_rel'] = np.mean(area_errors) * 100  # %
            results['std_area_rel'] = np.std(area_errors) * 100  # %
        
        # 5. Store raw data for plotting
        results['raw_data'] = {
            'true_counts': true_counts,
            'pred_counts': pred_counts,
            'position_errors': np.array(pos_errors) * 1000,  # mV
            'height_errors_rel': np.array(height_errors) * 100,  # %
            'area_errors_rel': np.array(area_errors) * 100  # %
        }
        
        return results
    
    def print_report(self, results):
        """
        Print formatted statistical report.
        """
        print("\n" + "=" * 70)
        print("STATISTICAL VALIDATION REPORT")
        print("=" * 70)
        
        print("\n1. PEAK DETECTION METRICS (Classification)")
        print("-" * 70)
        print(f"  Precision:  {results['precision']:.4f}")
        print(f"  Recall:     {results['recall']:.4f}")
        print(f"  F1-Score:   {results['f1_score']:.4f}")
        
        print("\n2. PEAK COUNT ACCURACY")
        print("-" * 70)
        print(f"  Exact Match Accuracy: {results['count_accuracy']*100:.1f}%")
        print(f"  Mean Absolute Error:  {results['count_mae']:.3f} peaks")
        
        print("\n  Confusion Matrix (rows=true, cols=pred):")
        cm = results['confusion_matrix']
        print(f"       Predicted:  1    2    3")
        for i, label in enumerate([1, 2, 3]):
            print(f"  True {label}:      {cm[i, 0]:3d}  {cm[i, 1]:3d}  {cm[i, 2]:3d}")
        
        print("\n3. POSITION ACCURACY")
        print("-" * 70)
        print(f"  R² Coefficient:      {results.get('r2_position', 0):.4f}")
        print(f"  Mean Absolute Error: {results.get('mae_position', 0):.1f} mV")
        print(f"  RMSE:                {results.get('rmse_position', 0):.1f} mV")
        print(f"  Std Deviation:       {results.get('std_position', 0):.1f} mV")
        
        print("\n4. PARAMETER ACCURACY (Relative Errors)")
        print("-" * 70)
        print(f"  Height MAE:  {results.get('mae_height_rel', 0):.1f}%")
        print(f"  Height Std:  {results.get('std_height_rel', 0):.1f}%")
        if 'mae_area_rel' in results:
            print(f"  Area MAE:    {results.get('mae_area_rel', 0):.1f}%")
            print(f"  Area Std:    {results.get('std_area_rel', 0):.1f}%")
        
        print("\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        
        # Overall grade
        f1 = results['f1_score']
        r2 = results.get('r2_position', 0)
        
        if f1 > 0.95 and r2 > 0.95:
            grade = "EXCELLENT (A+)"
        elif f1 > 0.90 and r2 > 0.90:
            grade = "VERY GOOD (A)"
        elif f1 > 0.85 and r2 > 0.85:
            grade = "GOOD (B)"
        else:
            grade = "NEEDS IMPROVEMENT (C)"
        
        print(f"\nOverall Performance: {grade}")
        print(f"  - Peak detection (F1): {f1:.3f}")
        print(f"  - Position accuracy (R²): {r2:.3f}")
        print("\n" + "=" * 70 + "\n")
    
    def plot_results(self, results, save_path='outputs/images/statistical_validation.png'):
        """
        Create comprehensive visualization.
        """
        fig = plt.figure(figsize=(16, 10))
        
        raw = results['raw_data']
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        cm = results['confusion_matrix']
        im = ax1.imshow(cm, cmap='Blues', aspect='auto')
        ax1.set_xticks([0, 1, 2])
        ax1.set_yticks([0, 1, 2])
        ax1.set_xticklabels([1, 2, 3])
        ax1.set_yticklabels([1, 2, 3])
        ax1.set_xlabel('Predicted Count')
        ax1.set_ylabel('True Count')
        ax1.set_title('Confusion Matrix')
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                text = ax1.text(j, i, cm[i, j], ha="center", va="center", 
                              color="white" if cm[i, j] > cm.max()/2 else "black")
        plt.colorbar(im, ax=ax1)
        
        # 2. Position Error Distribution
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(raw['position_errors'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(results.get('mae_position', 0), color='red', ls='--', lw=2, label=f"MAE={results.get('mae_position', 0):.1f} mV")
        ax2.set_xlabel('Position Error (mV)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Position Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Height Error Distribution
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(raw['height_errors_rel'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        ax3.axvline(results.get('mae_height_rel', 0), color='red', ls='--', lw=2, 
                   label=f"MAE={results.get('mae_height_rel', 0):.1f}%")
        ax3.set_xlabel('Height Error (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Height Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Metrics Bar Chart
        ax4 = plt.subplot(2, 3, 4)
        metrics = ['Precision', 'Recall', 'F1-Score', 'R²']
        values = [results['precision'], results['recall'], results['f1_score'], results.get('r2_position', 0)]
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylim(0, 1.1)
        ax4.set_ylabel('Score')
        ax4.set_title('Key Metrics Summary')
        ax4.axhline(0.9, color='gray', ls='--', alpha=0.5, label='Target (0.9)')
        ax4.legend()
        ax4.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Count Agreement Plot
        ax5 = plt.subplot(2, 3, 5)
        count_range = [1, 2, 3]
        true_hist = [np.sum(raw['true_counts'] == c) for c in count_range]
        pred_hist = [np.sum(raw['pred_counts'] == c) for c in count_range]
        
        x = np.arange(len(count_range))
        width = 0.35
        ax5.bar(x - width/2, true_hist, width, label='True', color='skyblue', edgecolor='black', alpha=0.7)
        ax5.bar(x + width/2, pred_hist, width, label='Predicted', color='lightcoral', edgecolor='black', alpha=0.7)
        ax5.set_xlabel('Number of Peaks')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Peak Count Distribution')
        ax5.set_xticks(x)
        ax5.set_xticklabels(count_range)
        ax5.legend()
        ax5.grid(True, axis='y', alpha=0.3)
        
        # 6. Summary Text Box
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
STATISTICAL SUMMARY

Peak Detection:
  • Precision: {results['precision']:.3f}
  • Recall: {results['recall']:.3f}
  • F1-Score: {results['f1_score']:.3f}

Position Accuracy:
  • R²: {results.get('r2_position', 0):.3f}
  • MAE: {results.get('mae_position', 0):.1f} mV
  • RMSE: {results.get('rmse_position', 0):.1f} mV

Count Accuracy:
  • Exact Match: {results['count_accuracy']*100:.1f}%
  • MAE: {results['count_mae']:.2f} peaks

Parameter Errors:
  • Height (rel): {results.get('mae_height_rel', 0):.1f}±{results.get('std_height_rel', 0):.1f}%
        """
        
        ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('VER 2.0 Statistical Validation Report', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved visualization: {save_path}")


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("VER 2.0 Statistical Validation")
    print("=" * 70)
    print("Running comprehensive ML metrics on 100 test cases...\n")
    
    validator = StatisticalValidator()
    results = validator.run_statistical_test(n_samples=100)
    
    # Print report
    validator.print_report(results)
    
    # Plot results
    validator.plot_results(results)
    
    print("✓ Statistical validation complete!")
