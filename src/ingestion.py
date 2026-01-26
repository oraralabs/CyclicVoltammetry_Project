import pandas as pd
import numpy as np

def parse_cv_file(filepath):
    """
    Parses a wide CSV with robust encoding and dynamic header detection.
    """
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
            
    if df is None:
        raise ValueError("Could not read file. Unknown encoding or format.")

    valid_cols = [c for c in df.columns if "Unnamed" not in str(c)]
    df = df[valid_cols]
    v_col = df.columns[-2]
    i_col = df.columns[-1]
    
    return pd.DataFrame({
        'E_V': pd.to_numeric(df[v_col], errors='coerce'),
        'I_uA': pd.to_numeric(df[i_col], errors='coerce')
    }).dropna()

def fit_linear_baseline(x, y):
    """
    Fits a linear baseline connecting the 1% and 99% marks of an individual scan.
    """
    x = np.array(x)
    y = np.array(y)
    min_v, max_v = np.min(x), np.max(x)
    span = max_v - min_v
    
    # Anchors at the extreme ends
    anchor_1_v = min_v + (span * 0.01)
    anchor_2_v = min_v + (span * 0.99)
    
    # Interpolate to find current at those voltages
    sort_idx = np.argsort(x)
    a1_i = np.interp(anchor_1_v, x[sort_idx], y[sort_idx])
    a2_i = np.interp(anchor_2_v, x[sort_idx], y[sort_idx])
    
    slope = (a2_i - a1_i) / (anchor_2_v - anchor_1_v)
    intercept = a1_i - (slope * anchor_1_v)
    return (slope * x) + intercept

def process_file(filepath):
    """
    PHASE 1 RESTORED: Independent baseline logic for unique peak shapes.
    """
    df_raw = parse_cv_file(filepath)
    
    # 1. Turn Detection
    start_v = df_raw['E_V'].iloc[0]
    turn_idx = (df_raw['E_V'] - start_v).abs().idxmax()
    
    scan_a = df_raw.iloc[:turn_idx]
    scan_b = df_raw.iloc[turn_idx:]
    
    if scan_a['I_uA'].mean() > scan_b['I_uA'].mean():
        top_scan, bot_scan = scan_a, scan_b
    else:
        top_scan, bot_scan = scan_b, scan_a
        
    # 2. Independent Baseline Subtraction
    # Oxidation (Top)
    x_ox = top_scan['E_V'].values
    y_ox_raw = top_scan['I_uA'].values
    base_ox = fit_linear_baseline(x_ox, y_ox_raw)
    sig_ox = y_ox_raw - base_ox
    
    # Reduction (Bottom)
    x_red = bot_scan['E_V'].values
    y_red_raw = bot_scan['I_uA'].values
    base_red = fit_linear_baseline(x_red, y_red_raw)
    sig_red = y_red_raw - base_red
    
    return {
        'ox_x': x_ox, 
        'ox_y_raw': y_ox_raw, 
        'ox_base': base_ox, 
        'ox_sig': sig_ox,
        'red_x': x_red, 
        'red_y_raw': y_red_raw, 
        'red_base': base_red, 
        'red_sig': sig_red
    }