import pandas as pd
import numpy as np

# [Keep parse_cv_file exactly as it is - no changes needed there]
def parse_cv_file(filepath):
    # ... (Same code as before) ...
    encodings = ['utf-16', 'utf-8', 'latin1']
    df = None
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                lines = [f.readline() for _ in range(20)]
            header_idx = -1
            for i, line in enumerate(lines):
                if line.strip().startswith("V,") or "Potential" in line:
                    header_idx = i
                    break
            if header_idx != -1:
                df = pd.read_csv(filepath, encoding=enc, header=header_idx, skip_blank_lines=True)
                break
        except UnicodeError:
            continue
    if df is None: raise ValueError("Could not read file.")
    valid_cols = [c for c in df.columns if "Unnamed" not in str(c)]
    df = df[valid_cols]
    v_col = df.columns[-2]
    i_col = df.columns[-1]
    return pd.DataFrame({
        'E_V': pd.to_numeric(df[v_col], errors='coerce'),
        'I_uA': pd.to_numeric(df[i_col], errors='coerce')
    }).dropna()

def fit_anchor_baseline(x, y):
    """
    IMPROVED: Polynomial Anchor (Degree 2).
    Fits a gentle curve using the first 10% and last 10% of data points.
    This handles the capacitive curvature better than a straight line.
    """
    x = np.array(x)
    y = np.array(y)
    
    # Select anchor regions (Start and End)
    n_points = len(x)
    n_anchor = int(n_points * 0.05) # Use 5% edge points
    
    # Combine start and end data for fitting
    x_anchors = np.concatenate([x[:n_anchor], x[-n_anchor:]])
    y_anchors = np.concatenate([y[:n_anchor], y[-n_anchor:]])
    
    # Fit a 2nd degree polynomial (Parabola) to these anchors
    coeffs = np.polyfit(x_anchors, y_anchors, deg=2)
    baseline_poly = np.poly1d(coeffs)
    
    return baseline_poly(x)

def process_file(filepath):
    """
    INDEPENDENT BASELINES (Top and Bottom processed separately).
    """
    df_raw = parse_cv_file(filepath)
    
    # 1. Split
    start_v = df_raw['E_V'].iloc[0]
    turn_idx = (df_raw['E_V'] - start_v).abs().idxmax()
    
    scan_a = df_raw.iloc[:turn_idx].sort_values('E_V')
    scan_b = df_raw.iloc[turn_idx:].sort_values('E_V')
    
    # 2. Identify Top/Bottom
    if scan_a['I_uA'].mean() > scan_b['I_uA'].mean():
        ox_scan, red_scan = scan_a, scan_b
    else:
        ox_scan, red_scan = scan_b, scan_a
        
    # 3. Process Oxidation
    x_ox = ox_scan['E_V'].values
    y_ox_raw = ox_scan['I_uA'].values
    base_ox = fit_anchor_baseline(x_ox, y_ox_raw)
    sig_ox = y_ox_raw - base_ox
    
    # 4. Process Reduction
    x_red = red_scan['E_V'].values
    y_red_raw = red_scan['I_uA'].values
    base_red = fit_anchor_baseline(x_red, y_red_raw)
    sig_red = y_red_raw - base_red
    
    return {
        'ox_scan': ox_scan, 'red_scan': red_scan,
        'x_ox': x_ox, 'sig_ox': sig_ox, 'base_ox': base_ox,
        'x_red': x_red, 'sig_red': sig_red, 'base_red': base_red
    }