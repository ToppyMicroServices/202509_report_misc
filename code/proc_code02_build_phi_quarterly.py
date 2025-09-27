#!/usr/bin/env python
"""proc_code02_build_phi_quarterly

Build two processed CSV outputs from underlying JSDA RMBS component data and BoJ housing loan stock series:
 1) proc_code02_JP_phi_from_JSDA_cleaned_quarterly_DEDUP.csv
    - Semiannual RMBS component observations (Mar / Sep etc.) merged with BoJ loans (quarterly)
    - Computed columns (bnJPY):
        RMBS_total_bnJPY, RMBS_JHF_bnJPY, RMBS_priv_bnJPY, JP_bank_housing_bnJPY,
        RMBS_total_bnJPY_calc (= JHF + priv), gap_total_vs_calc,
        phi_JP_incl_private (= (JHF+priv)/(Loans + JHF+priv)), rmbs_sum (redundant = calc)
    - Duplicated dates removed.
 2) proc_code02_JP_phi_quarterly.csv
    - A quarterly panel (all quarter ends) with BoJ loans every quarter, RMBS only on semiannual
      dates (other quarters left NaN for RMBS_* and phi).
    - Columns: BoJ_housing_loans_bnJPY, RMBS_total_bnJPY, RMBS_JHF_bnJPY, RMBS_priv_bnJPY, phi_JP

Assumptions / Inference:
  * JSDA component file provides columns (100m_yen units):
        JHF_RMBS_100m_yen, Private_RMBS_100m_yen, Total_RMBS_100m_yen
    Conversion: 100m_yen / 10 -> bnJPY (since 100 million JPY = 0.1 billion JPY).
  * BoJ loans file provides at least one numeric column representing the stock in bnJPY.
    Preferred column name: 'BoJ_housing_loans_bnJPY' OR first numeric column after 'Date'.
  * If BoJ file is missing, fallback: infer BoJ series from existing JP_phi_quarterly.csv (if present).

CLI:
  --jsda PATH   (default auto-detect)
  --boj PATH    (default data_processed/JP_Bank_Housing_Loans_Stock_quarterly_LA01_DLBL_DB_SL.csv)
    --out-quarterly PATH (default data_processed/proc_code02_JP_phi_quarterly.csv)
    --out-dedup PATH     (default data_processed/proc_code02_JP_phi_from_JSDA_cleaned_quarterly_DEDUP.csv)
  --force              overwrite outputs even if exist
  --verbose            extra logging

Outputs overwrite by default only with --force (else replaced silently if missing). Simplicity > ultra safety here.
"""
from __future__ import annotations
import argparse, os, sys, textwrap
import pandas as pd
from typing import Optional, List

PROC_DIR = 'data_processed'
RAW_DIR = 'data_raw'

JSDA_CANDIDATES: List[str] = [
    os.path.join(PROC_DIR, 'JP_RMBS_components_semiannual_JSDA_2012_2025.csv'),
    os.path.join('.', 'JP_RMBS_components_semiannual_JSDA_2012_2025.csv'),  # root (repo root file)
    os.path.join(RAW_DIR, 'JP_RMBS_components_semiannual_JSDA_2012_2025.csv'),
]

BOJ_DEFAULT = os.path.join(PROC_DIR, 'proc_code02_JP_Bank_Housing_Loans_Stock_quarterly.csv')
BOJ_CANDIDATES: List[str] = [
    BOJ_DEFAULT,
    os.path.join(PROC_DIR, 'JP_Bank_Housing_Loans_Stock_quarterly_LA01_DLBL_DB_SL.csv'),
    os.path.join('data_processed_bk', 'proc_code02_JP_Bank_Housing_Loans_Stock_quarterly.csv'),
    os.path.join('data_processed_bk', 'JP_Bank_Housing_Loans_Stock_quarterly_LA01_DLBL_DB_SL.csv'),
]


def log(msg: str):
    print(f"[PROC02] {msg}")


def find_existing(path_list: List[str]) -> Optional[str]:
    for p in path_list:
        if os.path.isfile(p):
            return p
    return None


def load_jsda_components(path: str, verbose=False) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Date'])
    required = ['JHF_RMBS_100m_yen','Private_RMBS_100m_yen','Total_RMBS_100m_yen']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"JSDA file missing required columns: {missing}")
    # Convert 100m_yen -> bnJPY
    for col in required:
        new_col = col.replace('_100m_yen','_bnJPY')
        df[new_col] = pd.to_numeric(df[col], errors='coerce') / 10.0  # 100m yen /10 => bnJPY
    keep = ['Date','JHF_RMBS_bnJPY','Private_RMBS_bnJPY','Total_RMBS_bnJPY']
    out = df[keep].sort_values('Date').reset_index(drop=True)
    if verbose:
        log(f"JSDA components loaded {len(out)} rows: {path}")
    return out


def load_boj_series(path: str, verbose=False) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=['Date'])
    if 'BoJ_housing_loans_bnJPY' in df.columns:
        val_col = 'BoJ_housing_loans_bnJPY'
    else:
        # Heuristic: first numeric column after Date (case-insensitive)
        others = [c for c in df.columns if c.lower() != 'date']
        num_candidates = []
        for c in others:
            try:
                pd.to_numeric(df[c], errors='raise')
                num_candidates.append(c)
            except Exception:
                continue
        if not num_candidates:
            # last resort: coerce first non-date to numeric
            if others:
                df[others[0]] = pd.to_numeric(df[others[0]], errors='coerce')
                val_col = others[0]
            else:
                raise ValueError('No numeric columns in BoJ file')
        else:
            val_col = num_candidates[0]
        df = df.rename(columns={val_col: 'BoJ_housing_loans_bnJPY'})
    out = df[['Date','BoJ_housing_loans_bnJPY']].sort_values('Date').reset_index(drop=True)
    if verbose:
        log(f"BoJ loans loaded {len(out)} rows from {path} (min {out['Date'].min().date()} max {out['Date'].max().date()})")
    return out


def fallback_boj_from_phi(phi_quarterly_path: str, verbose=False) -> Optional[pd.DataFrame]:
    if not os.path.isfile(phi_quarterly_path):
        return None
    try:
        df = pd.read_csv(phi_quarterly_path, parse_dates=['Date'])
    except Exception:
        return None
    cols_lower = [c.lower() for c in df.columns]
    if 'date' not in cols_lower:
        return None
    if 'boj_housing_loans_bnjpy' not in cols_lower:
        return None
    # unify col names
    if 'Date' not in df.columns:
        # attempt case-insensitive match
        for c in df.columns:
            if c.lower() == 'date':
                df = df.rename(columns={c: 'Date'})
    if 'BoJ_housing_loans_bnJPY' not in df.columns:
        for c in df.columns:
            if c.lower() == 'boj_housing_loans_bnjpy':
                df = df.rename(columns={c: 'BoJ_housing_loans_bnJPY'})
    out = df[['Date','BoJ_housing_loans_bnJPY']].dropna(subset=['BoJ_housing_loans_bnJPY'])
    if verbose:
        log(f"Fallback BoJ series recovered from existing {phi_quarterly_path}")
    return out


def build_dedup(jsda: pd.DataFrame, boj: pd.DataFrame, verbose=False) -> pd.DataFrame:
    # Merge only on available semiannual dates (inner merge on those JSDA dates)
    merged = pd.merge(jsda, boj, on='Date', how='left')
    merged = merged.sort_values('Date').reset_index(drop=True)
    merged['RMBS_total_bnJPY_calc'] = merged['JHF_RMBS_bnJPY'] + merged['Private_RMBS_bnJPY']
    merged['gap_total_vs_calc'] = merged['Total_RMBS_bnJPY'] - merged['RMBS_total_bnJPY_calc']
    merged['rmbs_sum'] = merged['RMBS_total_bnJPY_calc']
    merged = merged.rename(columns={
        'Total_RMBS_bnJPY':'RMBS_total_bnJPY',
        'JHF_RMBS_bnJPY':'RMBS_JHF_bnJPY',
        'Private_RMBS_bnJPY':'RMBS_priv_bnJPY',
        'BoJ_housing_loans_bnJPY':'JP_bank_housing_bnJPY'
    })
    merged['phi_JP_incl_private'] = merged['RMBS_total_bnJPY_calc'] / (
        merged['JP_bank_housing_bnJPY'] + merged['RMBS_total_bnJPY_calc']
    )
    before = len(merged)
    merged = merged.drop_duplicates(subset=['Date'])
    if verbose and len(merged) != before:
        log(f"Removed {before-len(merged)} duplicate rows in dedup file")
    cols = ['Date','RMBS_total_bnJPY','RMBS_JHF_bnJPY','RMBS_priv_bnJPY','JP_bank_housing_bnJPY',
            'RMBS_total_bnJPY_calc','gap_total_vs_calc','phi_JP_incl_private','rmbs_sum']
    return merged[cols]


def build_quarterly_panel(dedup: pd.DataFrame, boj: pd.DataFrame, verbose=False) -> pd.DataFrame:
    # Build a full quarterly Date index from min to max in combined sources
    date_min = min(dedup['Date'].min(), boj['Date'].min())
    date_max = max(dedup['Date'].max(), boj['Date'].max())
    # Use 'QE' (quarter end) to avoid deprecated 'Q'
    full_idx = pd.date_range(date_min, date_max, freq='QE')
    # Prepare RMBS (semiannual) subset
    rmbs = dedup[['Date','RMBS_total_bnJPY','RMBS_JHF_bnJPY','RMBS_priv_bnJPY']].copy()
    rmbs = rmbs.rename(columns={'Date':'_Date'})
    # BoJ series quarterly
    boj_q = boj.rename(columns={'Date':'_Date','BoJ_housing_loans_bnJPY':'BoJ_housing_loans_bnJPY'})
    panel = pd.DataFrame({'Date': full_idx})
    panel = panel.rename(columns={'Date':'_Date'})
    panel = panel.merge(boj_q, on='_Date', how='left')
    panel = panel.merge(rmbs, on='_Date', how='left')
    panel = panel.rename(columns={'_Date':'Date'})
    # Compute phi where RMBS_total available
    panel['phi_JP'] = panel['RMBS_total_bnJPY'] / (panel['BoJ_housing_loans_bnJPY'] + panel['RMBS_total_bnJPY'])
    panel = panel[['Date','BoJ_housing_loans_bnJPY','RMBS_total_bnJPY','RMBS_JHF_bnJPY','RMBS_priv_bnJPY','phi_JP']]
    if verbose:
        log(f"Quarterly panel built {len(panel)} rows (range {panel['Date'].min().date()} -> {panel['Date'].max().date()})")
    return panel


def write_csv(df: pd.DataFrame, path: str, force: bool, verbose=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path) and not force:
        # Overwrite silently (design choice) but inform via log
        if verbose:
            log(f"Overwriting existing (no --force needed for this simple pipeline): {path}")
    # Ensure any pre-existing unnamed index column is dropped
    df = df.loc[:, [c for c in df.columns if not c.lower().startswith('unnamed')]]
    df.to_csv(path, index=False)
    if verbose:
        log(f"WROTE {path} rows={len(df)} cols={list(df.columns)}")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument('--jsda', help='JSDA RMBS components CSV (auto-detect if omitted)')
    ap.add_argument('--boj', default=BOJ_DEFAULT, help='BoJ housing loans quarterly CSV')
    ap.add_argument('--out-quarterly', default=os.path.join(PROC_DIR,'proc_code02_JP_phi_quarterly.csv'))
    ap.add_argument('--out-dedup', default=os.path.join(PROC_DIR,'proc_code02_JP_phi_from_JSDA_cleaned_quarterly_DEDUP.csv'))
    ap.add_argument('--force', action='store_true', help='(reserved) not strictly required (always overwrites)')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    verbose = args.verbose

    # Locate JSDA file
    jsda_path = args.jsda or find_existing(JSDA_CANDIDATES)
    if not jsda_path:
        log('ERROR: JSDA components file 未検出')
        log('Tried:')
        for c in JSDA_CANDIDATES: log(f'  - {c}')
        log('ヒント: 先に JSDA/JHF ソースを data_raw/ に配置 あるいは fetch_code03_extract_jsda_rmbs_components.py を実行')
        sys.exit(1)
    if verbose: log(f'Using JSDA file: {jsda_path}')

    # Load JSDA
    jsda = load_jsda_components(jsda_path, verbose=verbose)

    # Load BoJ
    boj_path = args.boj
    boj: Optional[pd.DataFrame] = None
    if boj_path and os.path.isfile(boj_path):
        try:
            boj = load_boj_series(boj_path, verbose=verbose)
        except Exception as e:
            log(f"WARN: Failed to load BoJ file '{boj_path}': {e}")
    if boj is None:
        # Try known candidate locations
        cand = find_existing(BOJ_CANDIDATES)
        if cand:
            try:
                boj = load_boj_series(cand, verbose=verbose)
                if verbose: log(f"Using BoJ fallback: {cand}")
            except Exception as e:
                log(f"WARN: BoJ fallback load failed '{cand}': {e}")
    if boj is None:
        fb = fallback_boj_from_phi(args.out_quarterly, verbose=verbose)
        if fb is not None:
            boj = fb
    if boj is None:
        log('ERROR: BoJ housing loans series 不足 (file & fallback ともに失敗).')
        log('期待ファイル例: data_processed/proc_code02_JP_Bank_Housing_Loans_Stock_quarterly.csv')
        log('対応: BoJ 元データを取得し上記形式 (Date, BoJ_housing_loans_bnJPY) で配置して再実行')
        sys.exit(2)

    dedup = build_dedup(jsda, boj, verbose=verbose)
    quarterly = build_quarterly_panel(dedup, boj, verbose=verbose)

    write_csv(dedup, args.out_dedup, force=args.force, verbose=verbose)
    write_csv(quarterly, args.out_quarterly, force=args.force, verbose=verbose)

    # Simple integrity summary
    log(f"phi rows (semiannual dedup)={dedup.shape} quarterly_panel={quarterly.shape}")
    log('DONE')

if __name__ == '__main__':
    main()
