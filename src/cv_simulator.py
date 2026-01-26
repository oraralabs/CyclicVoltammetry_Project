import numpy as np
import pandas as pd

class CVPhysicsSimulator:
    """
    Refined simulator using a diffusive-wave approximation to match 
    Randles-Sevcik physics (sharp rise, long tail).
    """
    def __init__(self, v_start=-1.0, v_end=1.0, points_per_scan=1000):
        self.v_start = v_start
        self.v_end = v_end
        self.points = points_per_scan

    def _randles_sevcik_shape(self, v, e0, height, width, direction):
        """
        Approximates the physical response of a diffusion-limited reaction.
        """
        # Normalized potential relative to formal potential
        # (F/RT) * (V - E0)
        f_rt = 38.92  # Constant at 25°C
        xi = f_rt * (v - e0) * direction
        
        # This function approximates the current profile of a Nernstian wave
        # It creates the characteristic sharp rise and the decay tail
        with np.errstate(over='ignore', invalid='ignore'):
            # The 'Diffusion Tail' physics
            term = np.exp(-xi)
            shape = 1.0 / (1.0 + term + 0.4 * np.sqrt(np.abs(xi) + 1e-9))
            
            # Mask values before the peak to ensure clean baseline
            shape = np.where(xi < -5, 0, shape)
            
        return shape * height

    def generate_complex_cv(self, species_list, noise_level=0.2):
        fwd_v = np.linspace(self.v_start, self.v_end, self.points)
        rev_v = np.linspace(self.v_end, self.v_start, self.points)
        v_total = np.concatenate([fwd_v, rev_v])
        
        i_total = np.zeros_like(v_total)
        
        for species in species_list:
            e0 = species['E0']
            h = species['height']
            w = species['width'] # Used here as a scaling factor for kinetics
            
            # Forward Scan (Oxidation)
            i_total[:self.points] += self._randles_sevcik_shape(fwd_v, e0, h, w, 1)
            
            # Reverse Scan (Reduction) - Physically shifted by ~59mV/n
            i_total[self.points:] += self._randles_sevcik_shape(rev_v, e0 - 0.06, -h * 0.95, w, -1)

        # Realistic Backgrounds
        i_total += v_total * 3.5  # Resistive Tilt
        
        phase = np.linspace(0, np.pi, self.points)
        hysteresis = 4.0 * np.sin(phase) 
        i_total[:self.points] += hysteresis
        i_total[self.points:] -= hysteresis
        
        # Instrument Noise
        i_total += np.random.normal(0, noise_level, len(i_total))

        return pd.DataFrame({'E_V': v_total, 'I_uA': i_total})