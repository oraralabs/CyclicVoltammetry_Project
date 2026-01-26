import pandas as pd
import numpy as np

def parse_cv_file(filepath):
    # Robust parsing logic
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
    """LINEAR ANCHOR (1% - 99%). The most stable method."""
    x = np.array(x)
    y = np.array(y)
    min_v, max_v = np.min(x), np.max(x)
    span = max_v - min_v
    
    anchor_1_v = min_v + (span * 0.01)
    anchor_2_v = min_v + (span * 0.99)
    
    sort_idx = np.argsort(x)
    anchor_1_i = np.interp(anchor_1_v, x[sort_idx], y[sort_idx])
    anchor_2_i = np.interp(anchor_2_v, x[sort_idx], y[sort_idx])
    
    slope = (anchor_2_i - anchor_1_i) / (anchor_2_v - anchor_1_v)
    intercept = anchor_1_i - (slope * anchor_1_v)
    
    return (slope * x) + intercept

def process_file(filepath):
    df_raw = parse_cv_file(filepath)
    start_v = df_raw['E_V'].iloc[0]
    turn_idx = (df_raw['E_V'] - start_v).abs().idxmax()
    
    scan_a = df_raw.iloc[:turn_idx].sort_values('E_V')
    scan_b = df_raw.iloc[turn_idx:].sort_values('E_V')
    
    if scan_a['I_uA'].mean() > scan_b['I_uA'].mean():
        ox_scan, red_scan = scan_a, scan_b
    else:
        ox_scan, red_scan = scan_b, scan_a
        
    x_ox = ox_scan['E_V'].values
    y_ox_raw = ox_scan['I_uA'].values
    base_ox = fit_anchor_baseline(x_ox, y_ox_raw)
    sig_ox = y_ox_raw - base_ox
    
    x_red = red_scan['E_V'].values
    y_red_raw = red_scan['I_uA'].values
    base_red = fit_anchor_baseline(x_red, y_red_raw)
    sig_red = y_red_raw - base_red
    
    return {
        'ox_scan': ox_scan, 'red_scan': red_scan,
        'x_ox': x_ox, 'sig_ox': sig_ox, 'base_ox': base_ox,
        'x_red': x_red, 'sig_red': sig_red, 'base_red': base_red
    }