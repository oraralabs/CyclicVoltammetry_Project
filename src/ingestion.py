import pandas as pd
import numpy as np

def parse_cv_file(filepath):
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
        except: continue
    if df is None: raise ValueError("File format error.")
    valid_cols = [c for c in df.columns if "Unnamed" not in str(c)]
    df = df[valid_cols]
    v_col, i_col = df.columns[-2], df.columns[-1]
    return pd.DataFrame({'E_V': pd.to_numeric(df[v_col], errors='coerce'), 'I_uA': pd.to_numeric(df[i_col], errors='coerce')}).dropna()

def fit_linear_baseline(x, y):
    x, y = np.array(x), np.array(y)
    span = np.max(x) - np.min(x)
    a1_v, a2_v = np.min(x) + (span * 0.01), np.min(x) + (span * 0.99)
    sort_idx = np.argsort(x)
    a1_i = np.interp(a1_v, x[sort_idx], y[sort_idx])
    a2_i = np.interp(a2_v, x[sort_idx], y[sort_idx])
    slope = (a2_i - a1_i) / (a2_v - a1_v)
    intercept = a1_i - (slope * a1_v)
    return (slope * x) + intercept

def process_file(filepath):
    df_raw = parse_cv_file(filepath)
    start_v = df_raw['E_V'].iloc[0]
    turn_idx = (df_raw['E_V'] - start_v).abs().idxmax()
    scan_a = df_raw.iloc[:turn_idx].sort_values('E_V')
    scan_b = df_raw.iloc[turn_idx:].sort_values('E_V')
    
    if scan_a['I_uA'].mean() > scan_b['I_uA'].mean():
        top, bot = scan_a, scan_b
    else:
        top, bot = scan_b, scan_a
        
    ox_x, ox_y = top['E_V'].values, top['I_uA'].values
    ox_base = fit_linear_baseline(ox_x, ox_y)
    
    red_x, red_y = bot['E_V'].values, bot['I_uA'].values
    red_base = fit_linear_baseline(red_x, red_y)
    
    return {
        'ox_x': ox_x, 'ox_y_raw': ox_y, 'ox_base': ox_base, 'ox_sig': ox_y - ox_base,
        'red_x': red_x, 'red_y_raw': red_y, 'red_base': red_base, 'red_sig': red_y - red_base
    }