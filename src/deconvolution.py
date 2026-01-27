import numpy as np
from sklearn.linear_model import Lasso

class SparseDeconvolution:
    def __init__(self, basis_width=0.10):
        self.basis_width = basis_width

    def construct_basis_matrix(self, voltage_grid):
        n_points = len(voltage_grid)
        centers = voltage_grid[::5] 
        A = np.zeros((n_points, len(centers)))
        sigma = self.basis_width / 2.355
        for j, mu in enumerate(centers):
            A[:, j] = np.exp(-0.5 * ((voltage_grid - mu) / sigma)**2)
        return A, centers

    def solve(self, x, y, alpha=0.01):
        A, centers = self.construct_basis_matrix(x)
        solver = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=5000)
        solver.fit(A, y)
        return solver.coef_, solver.predict(A), centers

def consolidate_peaks(distribution, centers, voltage_grid):
    if np.max(distribution) == 0: return []
    v_min, v_max = np.min(voltage_grid), np.max(voltage_grid)
    margin = (v_max - v_min) * 0.05
    
    sig_idx = np.where(distribution > np.max(distribution) * 0.1)[0]
    final_peaks = []
    
    if len(sig_idx) > 0:
        clusters, current_cluster = [], [sig_idx[0]]
        for i in range(1, len(sig_idx)):
            if sig_idx[i] <= sig_idx[i-1] + 2:
                current_cluster.append(sig_idx[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sig_idx[i]]
        clusters.append(current_cluster)

        for cluster in clusters:
            best_idx = cluster[np.argmax(distribution[cluster])]
            peak_v = centers[best_idx]
            if (peak_v > v_min + margin) and (peak_v < v_max - margin):
                final_peaks.append({'v': peak_v, 'mag': distribution[best_idx]})
    return final_peaks