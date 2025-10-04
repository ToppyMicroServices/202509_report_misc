"""proc_code06_build_jp_jhf_rmbs_vs_gdp

Reconstruct a quarterly JHF RMBS vs GDP dataset from the semiannual JSDA components and JP GDP.

Outputs (prefixed):
  data_processed/proc_code06_JP_JHF_RMBS_vs_GDP_quarterly.csv
  data_processed/proc_code06_JP_JHF_RMBS_vs_GDP_annual_summary.csv (year-end level + annual avg ratio)

Logic:
  - Read components: expects columns containing 'JHF' (stock, hundred million yen) and a date column.
  - Convert 100m yen -> billion yen (*0.1) for JHF.
  - Semiannual to quarterly: forward-fill within each half-year to quarter boundaries (Q1/Q2/Q3/Q4).
  - Load GDP (JPNNGDP.csv or IMF alternative) and create quarterly NGDP in billion yen.
  - Compute ratio_pct = JHF_RMBS_bny / NGDP_bny * 100.
  - Annual summary: year-end JHF stock, annual NGDP sum, annual average ratio (mean of quarterly ratio).

Fallbacks:
  - GDP search list: JPNNGDP.csv, NGDPSAXDCJPQ.csv, fetch_code07_JP_IMF_GDP_quarterly.csv.
  - If GDP missing -> skip gracefully.

CLI:
  python code/proc_code06_build_jp_jhf_rmbs_vs_gdp.py [--force]
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
import argparse
from util_code01_lib_io import safe_to_csv, find

PROC = Path('data_processed')
RAW = Path('data_raw')
COMP_FILE = 'JP_RMBS_components_semiannual_JSDA_2012_2025.csv'
# Fallback actual fetch output name present in repo
COMP_FALLBACKS = [
    'fetch_code03_JSDA_RMBS_components.csv'
]
GDP_FILES = [
    'JPNNGDP.csv',
    'NGDPSAXDCJPQ.csv',
    'fetch_code07_JP_IMF_GDP_quarterly.csv',
    'JP_IMF_GDP_quarterly.csv'
]

def load_components():
    # Try canonical expected name
    if (PROC/COMP_FILE).exists():
        p = PROC/COMP_FILE
    else:
        # fallback search
        found = None
        for name in COMP_FALLBACKS:
            cand = PROC / name
            if cand.exists():
                found = cand; break
        if found is None:
            return pd.DataFrame()
        p = found
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
    # If file has only one column without dates, synthesize semiannual dates
    if df.shape[1] == 1:
        col = df.columns[0]
        # assume values sorted; generate semiannual sequence starting 2012H1 if length fits
        start = pd.Timestamp('2012-06-30')
        dates = [start + pd.DateOffset(months=6*i) for i in range(len(df))]
        df.insert(0,'Date', dates)
    dcol = None
    for c in df.columns:
        if 'date' in c.lower():
            dcol = c; break
    if dcol is None:
        dcol = df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol], errors='coerce')
    df = df.dropna(subset=[dcol]).sort_values(dcol)
    jhf_col = None
    for c in df.columns:
        low = c.lower()
        if ('jhf' in low and ('rmbs' in low or 'mortgage' in low)) or ('rmbs' in low and 'jhf' not in low and jhf_col is None):
            jhf_col = c; break
    if jhf_col is None:
        return pd.DataFrame()
    out = df[[dcol, jhf_col]].rename(columns={jhf_col:'JHF_RMBS_100m_yen'})
    out['JHF_RMBS_bny'] = out['JHF_RMBS_100m_yen'] * 0.1
    out = out.drop(columns=['JHF_RMBS_100m_yen']).set_index(dcol).sort_index()
    return out

def load_gdp():
    for name in GDP_FILES:
        # search processed/raw
        candidates = find([name])
        if not candidates:
            continue
        path = candidates[0]
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        # detect date column
        dcol=None
        for c in df.columns:
            if c.lower() in ('date','observation_date'):
                dcol=c; break
        if dcol is None:
            dcol=df.columns[0]
        df[dcol]=pd.to_datetime(df[dcol], errors='coerce')
        df=df.dropna(subset=[dcol]).set_index(dcol).sort_index()
        # first numeric
        val=None
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                val=c; break
        if val is None:
            continue
        g = df[[val]].rename(columns={val:'NGDP_raw'})
        # Heuristic scale: if max>1e6 assume million yen; keep raw (already in 1000?)/ simply convert to billion by /1000 if >1e6?
        # For NGDP (JPNNGDP.csv) values ~500000 so representing billions of yen (approx). We'll treat as billions already.
        g['NGDP_bny'] = g['NGDP_raw']
        return g[['NGDP_bny']]
    return pd.DataFrame()

def to_quarterly_from_semi(df: pd.DataFrame):
    # Expand semiannual points to quarter periods: assign each date to its quarter, ffill
    q = df.copy()
    q.index = q.index.to_period('Q').to_timestamp()
    # Reindex to full quarterly range
    full = pd.date_range(q.index.min(), q.index.max(), freq='QS')
    out = q.reindex(full).ffill()
    return out

def build():
    comp = load_components()
    if comp.empty:
        print('[WARN] components missing/empty')
        return 1
    gdp = load_gdp()
    if gdp.empty:
        print('[WARN] gdp missing/empty')
        return 1
    comp_q = to_quarterly_from_semi(comp)
    gdp_q = gdp.copy()
    gdp_q.index = gdp_q.index.to_period('Q').to_timestamp()
    merged = comp_q.join(gdp_q, how='inner')
    if merged.empty:
        print('[WARN] merged empty after join')
        return 1
    merged['ratio_pct'] = 100.0 * merged['JHF_RMBS_bny'] / merged['NGDP_bny']
    quarterly_path = PROC/'proc_code06_JP_JHF_RMBS_vs_GDP_quarterly.csv'
    safe_to_csv(merged.reset_index().rename(columns={'index':'DATE'}), quarterly_path, index=False, overwrite=True)
    # Annual summary
    ann_end = merged.resample('YE-DEC').last()[['JHF_RMBS_bny']]
    ann_gdp = merged['NGDP_bny'].resample('YE-DEC').sum().to_frame('NGDP_bny')
    ann_avg_ratio = merged['ratio_pct'].resample('YE-DEC').mean().to_frame('ratio_pct_avg')
    annual = ann_end.join(ann_gdp).join(ann_avg_ratio)
    annual['ratio_pct_year_end'] = 100.0 * ann_end['JHF_RMBS_bny'] / ann_gdp['NGDP_bny']
    annual_path = PROC/'proc_code06_JP_JHF_RMBS_vs_GDP_annual_summary.csv'
    safe_to_csv(annual.reset_index().rename(columns={'index':'DATE'}), annual_path, index=False, overwrite=True)
    print('[OK] wrote', quarterly_path)
    print('[OK] wrote', annual_path)
    return 0

def main(argv=None):
    ap = argparse.ArgumentParser(description='Build JP JHF RMBS vs GDP quarterly + annual summary (proc_code06).')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)
    PROC.mkdir(parents=True, exist_ok=True)
    return build()

if __name__ == '__main__':  # pragma: no cover
    import sys
    sys.exit(main())
