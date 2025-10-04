#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
proc_code04_build_us_phi_series_observed.py

Builds US observed φ (pre/post anchored) under data_processed/ from FRED-like raw CSVs in data_raw/.

Inputs (required):
  - data_raw/AGSEBMPTCMAHDFS.csv   (Agency/GSE Pools)
  - data_raw/HHMSDODNS.csv         (Home mortgages; Z.1)
Input (optional):
  - data_raw/BOGZ1FL673065105Q.csv (ABS Issuers; proxy for PLMBS). If missing, uses 0 (agency-only lower bound).

Outputs (CSV):
  - data_processed/proc_code04_US_phi_series_observed_pre.csv
  - data_processed/proc_code04_US_phi_series_observed_post.csv

Notes:
  - Resample to quarter end (Q-DEC) with last observation. Units normalized to USD billions via heuristic.
  - Strict splice at boundary (default 2010-03-31) on numerator (Pools + PLMBS) with windowed mean alignment.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / 'data_raw'
OUTD = ROOT / 'data_processed'
POOLS = RAW / 'AGSEBMPTCMAHDFS.csv'
HOME  = RAW / 'HHMSDODNS.csv'
PLMBS = RAW / 'BOGZ1FL673065105Q.csv'
OUT_PRE  = OUTD / 'proc_code04_US_phi_series_observed_pre.csv'
OUT_POST = OUTD / 'proc_code04_US_phi_series_observed_post.csv'

def read_fred_like(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    dcol = None
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ('date','observation_date'):
            dcol = c; break
    if dcol is None:
        dcol = df.columns[0]
    vcol = None
    for c in df.columns:
        if c != dcol:
            vcol = c; break
    if vcol is None:
        vcol = df.columns[-1]
    t = pd.to_datetime(df[dcol], errors='coerce')
    v = pd.to_numeric(df[vcol], errors='coerce')
    s = pd.Series(v.values, index=t).sort_index()
    s = s[s.index.notna()]
    return s

def to_quarter_bil(s: pd.Series) -> pd.Series:
    sq = s.resample('QE-DEC').last()
    med = float(pd.Series(sq).dropna().median()) if sq.notna().any() else 0.0
    if med > 1e4:  # millions -> billions
        sq = sq / 1000.0
    elif 0 < med < 10:  # trillions -> billions
        sq = sq * 1000.0
    return sq

def splice_strict(series: pd.Series, boundary: str, window: int, anchor: str) -> tuple[pd.Series,float]:
    b = pd.Timestamp(boundary)
    pre  = series.loc[series.index <  b].dropna()
    post = series.loc[series.index >= b].dropna()
    if pre.empty or post.empty:
        return series, 1.0
    pre_vals  = pre.iloc[-window:]  if len(pre)  >= window else pre
    post_vals = post.iloc[:window] if len(post) >= window else post
    pre_mean, post_mean = float(pre_vals.mean()), float(post_vals.mean())
    if not np.isfinite(pre_mean) or not np.isfinite(post_mean) or pre_mean==0 or post_mean==0:
        return series, 1.0
    if anchor == 'pre':
        k = pre_mean / post_mean
        post_adj = post * k
        combined = pd.concat([pre, post_adj]).sort_index()
    else:
        k = post_mean / pre_mean
        pre_adj = pre * k
        combined = pd.concat([pre_adj, post]).sort_index()
    return combined, float(k)

def build(anchor: str, boundary: str, window: int) -> pd.DataFrame:
    pools = to_quarter_bil(read_fred_like(POOLS))
    home  = to_quarter_bil(read_fred_like(HOME))
    if PLMBS.exists():
        plmbs = to_quarter_bil(read_fred_like(PLMBS))
        plmbs_src = PLMBS.name
    else:
        plmbs = pd.Series(0.0, index=home.index)
        plmbs_src = 'missing->zero'
    idx = home.index
    pools_al = pools.reindex(idx).ffill()
    plmbs_al = plmbs.reindex(idx).ffill()
    num = pools_al + plmbs_al
    num_s, k = splice_strict(num, boundary=boundary, window=window, anchor=anchor)
    phi = (num_s / home.reindex(num_s.index)).rename('phi_US_observed_proxy')
    phi_agency_only = (pools_al / home.reindex(pools_al.index)).reindex(phi.index)
    df = pd.DataFrame({
        'Date': phi.index,
        'pools_agency_bilUSD': pools_al.reindex(phi.index),
        'plmbs_bilUSD': plmbs_al.reindex(phi.index),
        'pools_total_bilUSD': num_s.reindex(phi.index),
        'home_mortgages_bilUSD': home.reindex(phi.index),
        'phi_US_observed_proxy': phi,
        'phi_US_agency_only_lower_bound': phi_agency_only,
    })
    df.attrs['splice_k'] = k
    df.attrs['plmbs_src'] = plmbs_src
    df.attrs['boundary'] = boundary
    df.attrs['window'] = window
    df.attrs['anchor'] = anchor
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boundary', default='2010-03-31')
    ap.add_argument('--window', type=int, default=2)
    args = ap.parse_args()
    for p in (POOLS, HOME):
        if not p.exists():
            raise SystemExit(f"[ERR] missing required raw file: {p}")
    OUTD.mkdir(parents=True, exist_ok=True)
    df_pre = build('pre', boundary=args.boundary, window=args.window)
    df_post = build('post', boundary=args.boundary, window=args.window)
    df_pre.to_csv(OUT_PRE, index=False)
    df_post.to_csv(OUT_POST, index=False)
    with open(OUT_PRE.with_suffix(OUT_PRE.suffix+'.audit.txt'), 'w', encoding='utf-8') as f:
        f.write(f"splice: on, anchor=pre, boundary={df_pre.attrs['boundary']}, window={df_pre.attrs['window']}, k={df_pre.attrs['splice_k']:.6g}\n")
        f.write(f"plmbs source: {df_pre.attrs['plmbs_src']}\n")
        f.write(f"rows: {len(df_pre)}\n")
    with open(OUT_POST.with_suffix(OUT_POST.suffix+'.audit.txt'), 'w', encoding='utf-8') as f:
        f.write(f"splice: on, anchor=post, boundary={df_post.attrs['boundary']}, window={df_post.attrs['window']}, k={df_post.attrs['splice_k']:.6g}\n")
        f.write(f"plmbs source: {df_post.attrs['plmbs_src']}\n")
        f.write(f"rows: {len(df_post)}\n")
    print(f"[OK] wrote {OUT_PRE} and {OUT_POST}")

if __name__ == '__main__':
    main()
