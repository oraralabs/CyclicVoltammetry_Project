import numpy as np
from lmfit.models import GaussianModel, LinearModel
from scipy.signal import find_peaks

class PeakResolver:
    def fit_peaks(self, x, y, ai_prominence):
        """
        Iterative Deconvolution:
        1. Finds obvious peaks using AI threshold.
        2. Fits.
        3. Checks residuals (error). If error is high, looks for peaks in the residuals (shoulders).
        4. Refits with added peaks.
        """
        # 1. First Pass: Obvious Peaks
        # We use a lower width to catch sharper features
        peaks_idx, _ = find_peaks(y, prominence=ai_prominence, width=5)
        
        # If nothing found, try a very loose search just to get started
        if len(peaks_idx) == 0:
            peaks_idx, _ = find_peaks(y, prominence=ai_prominence*0.5)
            
        if len(peaks_idx) == 0: return [], None

        # 2. Build Initial Model
        model = LinearModel(prefix='base_')
        params = model.make_params(slope=0, intercept=0)
        
        for i, idx in enumerate(peaks_idx):
            self._add_gaussian(model, params, x[idx], y[idx], i)
            
        # 3. Initial Fit
        result = model.fit(y, params, x=x)
        
        # 4. RESIDUAL CHECK (The "Smart" Step)
        # Calculate where the model failed
        residuals = y - result.best_fit
        
        # Look for peaks in the residuals (Hidden Shoulders)
        # We look for lumps that are at least 20% of the main signal height
        res_peaks, _ = find_peaks(residuals, height=np.max(y)*0.1, width=5)
        
        if len(res_peaks) > 0:
            print(f"    ... Found {len(res_peaks)} hidden shoulders. Refitting.")
            
            # Add these new peaks to the existing model
            current_peak_count = len(peaks_idx)
            for i, idx in enumerate(res_peaks):
                # Add new gaussian
                pid = current_peak_count + i
                self._add_gaussian(model, params, x[idx], residuals[idx], pid)
            
            # Refit with all peaks together
            result = model.fit(y, params, x=x)

        # 5. Extract Results
        final_peaks = []
        for name in result.params:
            if 'center' in name:
                prefix = name.split('center')[0]
                final_peaks.append({
                    'loc': result.params[f'{prefix}center'].value,
                    'height': result.params[f'{prefix}height'].value,
                    'width': result.params[f'{prefix}sigma'].value * 2.355
                })
                
        return final_peaks, result

    def _add_gaussian(self, model, params, center, height, idx):
        prefix = f'g{idx}_'
        peak = GaussianModel(prefix=prefix)
        params.update(peak.make_params(center=center, amplitude=height*0.1, sigma=0.05))
        model += peak