"""fig_code10_verify_us_phi_sources

End-to-end verification for US off-balance sheet share phi (φ):

Goals:
    - Fetch or load two FRED series (pools outstanding & household mortgages) and compute φ = Pools / HH.
    - Detect a potential structural break (largest absolute quarterly diff post-1990) or use a user-specified date.
    - If a break is detected, scale the pre-break segment to align levels, producing an adjusted series.

Outputs:
    data_processed/US_phi_series_raw.csv           (raw pools, hh, phi_raw)
    data_processed/US_phi_series_adjusted.csv      (includes scaled pre-break series if detected)
    data_processed/US_phi_verification_report.txt  (metadata & break diagnostics)
    figures/fig_code10_US_phi_break_verification.png (annotated figure)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import io  # legacy (kept for potential future text buffer usage)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from util_code01_lib_io import safe_savefig, safe_to_csv, figure_name_with_code
from fetch_code08_fred_series import fetch_fred_series

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POOL_ID = "AGSEBMPTCMAHDFS"   # Agency- and GSE-Backed Mortgage Pools (Millions USD)
HH_ID   = "HHMSDODNS"         # Household Mortgages Outstanding (Trillions or Billions)
DATA_RAW = Path("data_raw")
DATA_PROCESSED = Path("data_processed")
FIG_DIR = Path("figures")

# ---------------------------------------------------------------------------
# Fetch & Load Helpers
# ---------------------------------------------------------------------------
def fetch_series(series_id: str, out_dir: Path, do_download: bool=True):
    """Wrapper around fetch_fred_series to keep original signature.

    Returns (df, saved_path|None) with index normalized to quarter end.
    """
    df, saved = fetch_fred_series(series_id, out_dir, do_download=do_download)
    return df, saved


def load_local(series_id: str) -> pd.DataFrame:
    path = DATA_RAW / f"{series_id}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] local load failed {series_id}: {e}")
        return pd.DataFrame()
    # FRED sometimes uses 'DATE', sometimes 'observation_date'. Accept any column containing 'date'.
    if 'DATE' not in df.columns:
        date_candidates = [c for c in df.columns if 'date' in c.lower()]
        if date_candidates:
            df = df.rename(columns={date_candidates[0]: 'DATE'})
        else:
            print(f"[WARN] no DATE-like column in {path.name}; columns={list(df.columns)[:5]}")
            return pd.DataFrame()
    if series_id not in df.columns:
        val_cols = [c for c in df.columns if c != 'DATE']
        if not val_cols:
            return pd.DataFrame()
        df = df.rename(columns={val_cols[0]: series_id})
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['DATE']).set_index('DATE').sort_index()
    df.index = df.index.to_period('Q').to_timestamp('Q')  # already quarter-end; OK
    return df


# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------
def harmonize_units(pools: pd.Series, hh: pd.Series) -> Tuple[pd.Series, pd.Series, str, str]:
    pools_bn = pools / 1000.0  # millions -> billions
    pools_note = "converted from millions to billions (÷1000)"
    hh_max = hh.max()
    if hh_max < 500:  # implies trillions
        hh_bn = hh * 1000.0
        hh_note = "assumed trillions; scaled *1000 to billions"
    else:
        hh_bn = hh
        hh_note = "assumed already billions"
    return pools_bn, hh_bn, pools_note, hh_note


def detect_break(phi: pd.Series, forced_date: Optional[pd.Timestamp]=None):
    if phi.empty:
        return None
    if forced_date is not None and forced_date in phi.index:
        bk = forced_date
        jump = phi.loc[bk] - phi.shift().loc[bk]
    else:
        diffs = phi.diff().dropna()
        if diffs.empty:
            return None
        recent = diffs[diffs.index >= pd.Timestamp('1990-01-01')]
        target = recent if not recent.empty else diffs
        bk = target.abs().idxmax()
        jump = target.loc[bk]
    loc = phi.index.get_loc(bk)
    if isinstance(loc, slice) or loc == 0:
        return None
    prev = phi.index[loc-1]
    med_prev = phi.diff().abs().loc[:prev].tail(8).median()
    if med_prev is None or np.isnan(med_prev):
        med_prev = 0.0
    if abs(jump) > max(0.05, 5*med_prev):
        scale = phi.loc[bk] / phi.loc[prev] if phi.loc[prev] != 0 else np.nan
        return {
            'break_date': bk,
            'prev_date': prev,
            'jump': float(jump),
            'scale_pre_to_post': float(scale),
            'median_prev_abs_diff': float(med_prev)
        }
    return None


def build_adjusted(phi: pd.Series, info: dict) -> pd.Series:
    if not info:
        return pd.Series(dtype=float)
    prev = info['prev_date']
    scale = info['scale_pre_to_post']
    if not np.isfinite(scale) or scale <= 0:
        return pd.Series(dtype=float)
    adj = phi.copy()
    adj.loc[adj.index <= prev] = adj.loc[adj.index <= prev] * scale
    return adj


def write_report(path: Path, meta: dict):
    lines = ["US φ Verification Report", "==========================", ""]
    for k,v in meta.items():
        lines.append(f"{k}: {v}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding='utf-8')
    print(f"[INFO] Report written: {path}")


def plot_series(raw: pd.Series, adj: Optional[pd.Series], info: Optional[dict]):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(raw.index, raw, label='φ raw', marker='o', ms=3)
    if adj is not None and not adj.empty:
        ax.plot(adj.index, adj, label='φ adjusted (pre scaled)', linestyle='--')
    if info:
        bk = info['break_date']
        ax.axvline(bk, color='k', lw=1, alpha=0.6)
        ax.text(bk, ax.get_ylim()[1]*0.95, f"break\n{bk.date()}", ha='center', va='top', fontsize=8)
    ax.set_title('US φ structural break verification')
    ax.set_ylabel('share')
    ax.grid(True, alpha=0.3)
    ax.legend()
    outp = figure_name_with_code(__file__, FIG_DIR / 'US_phi_break_verification.png')
    saved = safe_savefig(fig, outp, overwrite=True)
    print(f"[INFO] Figure written: {saved}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='US phi (off-balance share) source fetch & structural break verification')
    ap.add_argument('--no-download', action='store_true', help='Skip network fetch; rely on local CSVs only')
    ap.add_argument('--break-date', type=str, default=None, help='Force break date (YYYY-MM-DD); skip auto detection if provided')
    args = ap.parse_args()

    forced_break = pd.Timestamp(args.break_date) if args.break_date else None

    # load local (before possible overwrite by download)
    local_pool = load_local(POOL_ID)
    local_hh = load_local(HH_ID)

    dl_pool, dl_pool_path = fetch_series(POOL_ID, DATA_RAW, not args.no_download)
    dl_hh,   dl_hh_path   = fetch_series(HH_ID, DATA_RAW, not args.no_download)

    pool = dl_pool if not dl_pool.empty else local_pool
    hh   = dl_hh   if not dl_hh.empty   else local_hh
    if pool.empty or hh.empty:
        if args.no_download:
            print('[SKIP] required series missing locally (pool or hh empty) and --no-download specified')
            print(f'  EXPECTED FILES: data_raw/{POOL_ID}.csv data_raw/{HH_ID}.csv')
            return 0  # treat as graceful skip so orchestrator does not fail
        print('[FATAL] required series missing (pool or hh empty)')
        print(f'  HINT: run without --no-download or ensure data_raw/{POOL_ID}.csv and {HH_ID}.csv exist')
        return 1

    pool = pool[~pool.index.duplicated()]
    hh   = hh[~hh.index.duplicated()]

    pools_bn, hh_bn, pool_note, hh_note = harmonize_units(pool.iloc[:,0], hh.iloc[:,0])
    df = pd.concat([pools_bn, hh_bn], axis=1)
    df.columns = ['pools_bn', 'hh_bn']
    df['phi_raw'] = (df['pools_bn'] / df['hh_bn']).clip(lower=0)

    info = detect_break(df['phi_raw'], forced_break)
    if info:
        phi_adj = build_adjusted(df['phi_raw'], info)
        df_adj = df.copy()
        df_adj['phi_adj_prebreak_scaled'] = phi_adj
    else:
        phi_adj = pd.Series(dtype=float)
        df_adj = df.copy()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pref_raw = DATA_PROCESSED / 'proc_code04_US_phi_series_raw.csv'
    pref_adj = DATA_PROCESSED / 'proc_code04_US_phi_series_adjusted.csv'
    # Decide legacy vs pref target (if pref already present, append new fig_code10_ version only if content differs)
    def _maybe_save(df_out, legacy_name, pref_path):
        if pref_path.exists():
            # do not regenerate another copy; just inform
            print(f"[INFO] Prefixed exists; skipping legacy duplicate: {pref_path.name}")
            return pref_path
        legacy_target = figure_name_with_code(__file__, DATA_PROCESSED / legacy_name)
        return safe_to_csv(df_out, legacy_target, index=False, overwrite=True)
    raw_final = _maybe_save(df.reset_index().rename(columns={'index':'DATE'}), 'US_phi_series_raw.csv', pref_raw)
    print(f"[INFO] Saved raw series: {raw_final}")
    adj_final = _maybe_save(df_adj.reset_index().rename(columns={'index':'DATE'}), 'US_phi_series_adjusted.csv', pref_adj)
    print(f"[INFO] Saved adjusted series: {adj_final}")

    meta = {
        'pool_source_used': 'download' if not dl_pool.empty else 'local',
        'hh_source_used': 'download' if not dl_hh.empty else 'local',
        'pool_rows': len(pool),
        'hh_rows': len(hh),
        'pool_unit_note': pool_note,
        'hh_unit_note': hh_note,
        'phi_obs': len(df),
        'download_pool_path': str(dl_pool_path) if dl_pool_path else 'NA',
        'download_hh_path': str(dl_hh_path) if dl_hh_path else 'NA',
    }
    if info:
        meta.update({
            'detected_break_date': info['break_date'],
            'previous_quarter': info['prev_date'],
            'jump': info['jump'],
            'scale_pre_to_post': info['scale_pre_to_post'],
            'median_prev_abs_diff': info['median_prev_abs_diff'],
        })
    pref_rep = DATA_PROCESSED / 'proc_code04_US_phi_verification_report.txt'
    # Always apply figure_name_with_code so report gets fig_code10_ prefix if new
    rep_target = pref_rep if pref_rep.exists() else (DATA_PROCESSED / 'US_phi_verification_report.txt')
    write_report(figure_name_with_code(__file__, rep_target), meta)
    plot_series(df['phi_raw'], (phi_adj if not phi_adj.empty else None), info)
    print('[DONE] Verification complete')
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
