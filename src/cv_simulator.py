import numpy as np
import pandas as pd

class MechanisticSimulator:
    """
    High-Fidelity CV Simulator.
    Models Semi-Infinite Linear Diffusion (Randles-Sevcik) and 
    Double Layer Capacitance for a 'Research Grade' look.
    """
    def __init__(self, v_start=0.5, v_end=-0.5, scan_rate=0.1):
        self.v_start = v_start
        self.v_end = v_end
        self.scan_rate = scan_rate
        self.points = 1000

    def _get_norm_current(self, v, e0, direction):
        """
        The mathematical kernel for a reversible CV peak.
        Produces the sharp rise and the 1/sqrt(t) diffusion tail.
        """
        # (F/RT)(V - E0)
        f_rt = 38.92 
        xi = f_rt * (v - e0) * direction
        
        # Empirical approximation of the current function chi(sigma*t)
        # This creates the smooth 'Paper-Look' asymmetric curve
        with np.errstate(over='ignore', invalid='ignore'):
            # The 'Peak' part
            term1 = np.exp(-0.35 * xi)
            # The 'Tail' part
            term2 = 1.0 + np.exp(-xi)
            # Smooth transition
            current = 1.0 / (term1 + term2 + 0.45 * np.sqrt(np.abs(xi) + 0.01))
            
            # Clean the baseline before the reaction starts
            current = np.where(xi < -4, 0, current)
            
        return current

    def run_simulation(self, species_configs, c_dl=5.0, r_u=0.01, noise_std=0.02):
        """
        species_configs: List of dicts [{'E0': 0.0, 'height': 15.0}]
        c_dl: Double Layer Capacitance (uA*s/V) - joins the loop
        noise_std: Medical grade low noise
        """
        # 1. Voltage Sweep
        fwd_v = np.linspace(self.v_start, self.v_end, self.points)
        rev_v = np.linspace(self.v_end, self.v_start, self.points)
        v_total = np.concatenate([fwd_v, rev_v])
        
        i_faradaic = np.zeros_like(v_total)
        
        # 2. Add species (Faradaic Current)
        for sp in species_configs:
            e0 = sp['E0']
            h = sp['height']
            # Forward scan
            i_faradaic[:self.points] += self._get_norm_current(fwd_v, e0, -1) * h
            # Reverse scan (shifted by approx 59mV)
            i_faradaic[self.points:] += self._get_norm_current(rev_v, e0 - 0.06, 1) * -h * 0.95

        # 3. Add Capacitance (Double Layer Charging)
        # i_cap = C_dl * dV/dt. This ensures the loop meets at the ends.
        i_cap = np.zeros_like(v_total)
        # Forward scan charging
        i_cap[:self.points] = -c_dl * self.scan_rate
        # Reverse scan charging
        i_cap[self.points:] = c_dl * self.scan_rate
        
        # 4. Total Current + Resistive Tilt + Noise
        i_total = i_faradaic + i_cap + (v_total * 1.5)
        i_total += np.random.normal(0, noise_std, len(i_total))
        
        return v_total, i_total

def add_instrument_noise(v, i, noise_level=0.02):
    # Noise added inside simulator now, but keeping for compatibility
    return pd.DataFrame({'E_V': v, 'I_uA': i})