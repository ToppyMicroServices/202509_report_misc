"""proc_code05_build_annual_mbs_rmbs_ratios

Build unified annual MBS (US) and RMBS (Japan) to nominal GDP ratios with prefixed
processed outputs replacing legacy unprefixed CSVs:
  - US_MBS_to_AnnualGDP.csv
  - JP_RMBS_to_AnnualGDP.csv
  - JP_US_MBS_to_AnnualGDP_2012_2021.csv (comparison slice)

Outputs (always written, overwriting only via safe unique naming avoided):
  data_processed/proc_code05_US_MBS_to_NGDP_annual.csv
  data_processed/proc_code05_JP_RMBS_to_NGDP_annual.csv
  data_processed/proc_code05_JP_US_MBS_RMBS_to_NGDP_compare_2012_2021.csv

Logic:
  * US: Use quarterly FRED series AGSEBMPTCMAHDFS (Agency & GSE MBS outstanding, $Mil)
         + NGDPSAXDCUSQ (Nominal GDP, national currency, quarterly) if available locally.
         Convert MBS to $Bn, GDP quarterly sum to annual level. Ratio = MBS/GDP *100.
    Fallback: if quarterly inputs missing, attempt legacy annual ratio CSV.
  * JP: Use JSDA semiannual components CSV (fetch_code03) + JP nominal GDP quarterly
         (IMF or alternative JPNNGDP) – convert hundred million yen to billion yen (×0.1),
         resample to annual (Dec last for stock; GDP annual sum). Ratio *100.
    Fallback: legacy annual ratio CSV.
  * Join for comparison (intersection years) and slice 2012–2021 inclusive for reproducibility.

Assumptions / Tolerances:
  - Missing one side results in skipping comparison output but still writes the other side if built.
  - Date indices are year-end (Dec 31) timestamps for both series.
  - All numeric coercions drop invalid rows (warn if drops occur).

CLI:
  python code/proc_code05_build_annual_mbs_rmbs_ratios.py [--force]

"""
from __future__ import annotations
import pandas as pd, argparse, sys
from pathlib import Path
from util_code01_lib_io import find, read_fred_two_col, to_annual_from_q4, to_annual_from_quarterly_sum, safe_to_csv

PROC_DIR = Path('data_processed')
RAW_DIR = Path('data_raw')

US_MBS_SERIES = 'AGSEBMPTCMAHDFS.csv'
# Allow alternate local filename(s) if primary series name differs.
# MORTGAGE30US.csv is sometimes downloaded manually by users assuming it is the
# needed outstanding stock series; include as a fallback pattern so the script
# emits a clearer diagnostic instead of silent miss.
US_MBS_ALT_PATTERNS = ['MORTGAGE30US.csv', 'MORTGAGE30US*.csv', 'WMBS*.csv']
US_GDP_SERIES = 'NGDPSAXDCUSQ.csv'
JP_JSDA_COMPONENTS = 'JP_RMBS_components_semiannual_JSDA_2012_2025.csv'
JP_GDP_FILES = ['JPNNGDP.csv', 'fetch_code07_JP_IMF_GDP_quarterly.csv', 'JP_IMF_GDP_quarterly.csv']

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_us() -> pd.DataFrame|None:
    # First try canonical filename, then alternates.
    mbs_files = find([US_MBS_SERIES])
    if not mbs_files:
        alt = find(US_MBS_ALT_PATTERNS)
        if alt:
            print(f"[INFO] US MBS: using alternate file {alt[0].name}")
            mbs_files = alt
    gdp_files = find([US_GDP_SERIES, US_GDP_SERIES.replace('.csv','*.csv')])
    if not mbs_files or not gdp_files:
        if not mbs_files:
            print(f"[WARN] US MBS series not found (looked for {US_MBS_SERIES} or alternates {US_MBS_ALT_PATTERNS})")
        if not gdp_files:
            print(f"[WARN] US GDP series not found (expected {US_GDP_SERIES})")
        return None
    mbs = read_fred_two_col(mbs_files[0])
    col_m = mbs.columns[0]
    mbs[col_m] = mbs[col_m] / 1_000.0  # $Mil -> $Bn
    gdp = read_fred_two_col(gdp_files[0])
    col_g = gdp.columns[0]
    gdp_a = to_annual_from_quarterly_sum(gdp, col_g).rename(columns={col_g: 'US_NGDP_$Bn'})
    mbs_a = to_annual_from_q4(mbs, col_m).rename(columns={col_m: 'US_MBS_$Bn'})
    df = mbs_a.join(gdp_a, how='inner')
    if df.empty:
        return None
    df['US_MBS_to_NGDP_%'] = 100.0 * df['US_MBS_$Bn'] / df['US_NGDP_$Bn']
    return df

def build_us_fallback() -> pd.DataFrame|None:
    legacy = find(['US_MBS_to_AnnualGDP.csv'])
    if not legacy:
        return None
    try:
        df = pd.read_csv(legacy[0])
    except Exception:
        return None
    # Heuristic: first date-like column
    dcol = df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol], errors='coerce')
    df = df.dropna(subset=[dcol]).set_index(dcol).sort_index()
    # Find ratio column
    ratio_col = None
    for c in df.columns:
        if 'to' in c.lower() and 'gdp' in c.lower():
            ratio_col = c; break
    if ratio_col is None:
        # fallback numeric
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not nums:
            return None
        ratio_col = nums[-1]
    out = df[[ratio_col]].rename(columns={ratio_col:'US_MBS_to_NGDP_%'})
    return out

def build_jp() -> pd.DataFrame|None:
    jsda_files = find([JP_JSDA_COMPONENTS])
    gdp_files = find(JP_GDP_FILES)
    if not jsda_files or not gdp_files:
        return None
    jsda = pd.read_csv(jsda_files[0])
    # date column detection
    dcols = [c for c in jsda.columns if 'date' in c.lower() or 'month' in c.lower()]
    if not dcols:
        return None
    dcol = dcols[0]
    jsda[dcol] = pd.to_datetime(jsda[dcol], errors='coerce')
    jsda = jsda.dropna(subset=[dcol]).sort_values(dcol)
    cols_lower = {c.lower(): c for c in jsda.columns}
    total_col = None
    for k,v in cols_lower.items():
        if 'total' in k and ('100' in k or 'yen' in k or 'rmbs' in k):
            total_col = v; break
    if total_col is None:
        # sum JHF + private
        jhf = [c for c in jsda.columns if 'jhf' in c.lower()]
        prv = [c for c in jsda.columns if 'priv' in c.lower()]
        if not jhf or not prv:
            return None
        jsda['rmbs_total_100m_yen'] = jsda[jhf[0]] + jsda[prv[0]]
        total_col = 'rmbs_total_100m_yen'
    else:
        jsda = jsda.rename(columns={total_col: 'rmbs_total_100m_yen'})
    rmbs = jsda[[dcol, 'rmbs_total_100m_yen']].copy()
    rmbs[dcol] = pd.to_datetime(rmbs[dcol], errors='coerce')
    rmbs = rmbs.dropna(subset=[dcol]).set_index(dcol).sort_index()
    # Convert hundred million yen → billion yen
    rmbs_bny = (rmbs['rmbs_total_100m_yen'] * 0.1).to_frame('JP_RMBS_bny')
    # Semiannual → forward fill quarterly edges then annual last
    rmbs_q = rmbs_bny.resample('QE-DEC').ffill()
    # GDP
    gpath = gdp_files[0]
    gdp = pd.read_csv(gpath)
    gdp_dcol = None
    for c in gdp.columns:
        if 'date' in c.lower():
            gdp_dcol = c; break
    if gdp_dcol is None:
        gdp_dcol = gdp.columns[0]
    gdp[gdp_dcol] = pd.to_datetime(gdp[gdp_dcol], errors='coerce')
    # Value column
    val_candidates = [c for c in gdp.columns if c != gdp_dcol]
    val = val_candidates[0]
    gdp = gdp.dropna(subset=[gdp_dcol]).rename(columns={gdp_dcol:'DATE', val:'JP_NGDP_bny'}).set_index('DATE').sort_index()
    gdp_a = gdp['JP_NGDP_bny'].resample('YE-DEC').sum(min_count=1).to_frame()
    rmbs_a = rmbs_q.resample('YE-DEC').last()
    df = rmbs_a.join(gdp_a, how='inner')
    if df.empty:
        return None
    df['JP_RMBS_to_NGDP_%'] = 100.0 * df['JP_RMBS_bny'] / df['JP_NGDP_bny']
    return df

def build_jp_fallback() -> pd.DataFrame|None:
    legacy = find(['JP_RMBS_to_AnnualGDP.csv'])
    if not legacy:
        return None
    try:
        df = pd.read_csv(legacy[0])
    except Exception:
        return None
    dcol = df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol], errors='coerce')
    df = df.dropna(subset=[dcol]).set_index(dcol).sort_index()
    # heuristics
    ratio_col = None
    for c in df.columns:
        if 'to' in c.lower() and 'gdp' in c.lower():
            ratio_col = c; break
    if ratio_col is None:
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not nums:
            return None
        ratio_col = nums[-1]
    out = df[[ratio_col]].rename(columns={ratio_col:'JP_RMBS_to_NGDP_%'})
    return out

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Build annual US MBS & JP RMBS ratios (prefixed outputs).')
    ap.add_argument('--force', action='store_true', help='Ignore empty build failures and still write fallbacks if present.')
    args = ap.parse_args(argv)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    us = build_us()
    if us is None:
        us = build_us_fallback()
        if us is not None:
            print('[INFO] US ratio: used legacy fallback')
    if us is None:
        print('[WARN] US ratio unavailable')
    else:
        path_us = PROC_DIR/'proc_code05_US_MBS_to_NGDP_annual.csv'
        safe_to_csv(us.reset_index(), path_us, index=False)
        print('[OK] wrote', path_us)

    jp = build_jp()
    if jp is None:
        jp = build_jp_fallback()
        if jp is not None:
            print('[INFO] JP ratio: used legacy fallback')
    if jp is None:
        print('[WARN] JP ratio unavailable')
    else:
        path_jp = PROC_DIR/'proc_code05_JP_RMBS_to_NGDP_annual.csv'
        safe_to_csv(jp.reset_index(), path_jp, index=False)
        print('[OK] wrote', path_jp)

    if us is not None and jp is not None:
        compare = us.join(jp, how='inner')
        if not compare.empty:
            sl = compare[(compare.index.year>=2012) & (compare.index.year<=2021)]
            if sl.empty:
                print('[WARN] comparison slice 2012–2021 empty')
            else:
                path_cmp = PROC_DIR/'proc_code05_JP_US_MBS_RMBS_to_NGDP_compare_2012_2021.csv'
                safe_to_csv(sl.reset_index(), path_cmp, index=False)
                print('[OK] wrote', path_cmp)
        else:
            print('[WARN] no US/JP overlap for comparison')
    return 0

if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
