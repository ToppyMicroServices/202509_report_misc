#!/usr/bin/env python3
"""
proc_code01_build_k_decomposition.py

... (existing docstring content) ...
"""

from __future__ import annotations
import io, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / 'data_raw'
PROC = ROOT / 'data_processed'

# Outputs expected by downstream code
OUT_US_FULL     = PROC / 'proc_code01_k_decomposition_US_quarterly.csv'           # rate+term (new, required)
OUT_US_RATEONLY = PROC / 'proc_code01_k_decomposition_US_quarterly_RATE_ONLY.csv' # legacy (compat)
OUT_JP          = PROC / 'proc_code01_k_decomposition_JP_quarterly.csv'

# FRED mortgage 30y (monthly CSV)
FRED_MORTGAGE30US_CSV = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US'

# FRB DDP (Z.1)
DDP_BASE = 'https://www.federalreserve.gov/datadownload/Output.aspx'
DDP_PARAMS = {
    'rel': 'Z1', 'lastobs': '', 'from': '', 'to': '',
    'filetype': 'csv', 'label': 'omit', 'layout': 'seriescolumn',
}

# Z.1 series (DDP style IDs). Minimal set to build WAM proxy.
Z1_SERIES = {
    'HH_HOME_MORT_LVL': ('FL153165105.Q', 'HH_home_mort_level'),
    'HH_HOME_MORT_FLO': ('FA153165105.Q', 'HH_home_mort_flow'),
    # Agency pools (level) sometimes fails on mirrors; keep flow only as optional
    'AGENCY_POOLS_FLO': ('FA413065005.Q', 'AGY_pools_mort_asset_flow'),
}

FRED_ALIASES = {
    'FL153165105.Q': 'HMLBSHNO',
    'FA153165105.Q': 'BOGZ1FA153165105Q',
    'FL413065005.Q': 'BOGZ1FL413065005Q',
    'FA413065005.Q': 'BOGZ1FA413065005Q',
}

DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36',
    'Accept': 'text/csv,application/zip,application/octet-stream,application/xml;q=0.9,*/*;q=0.1',
}

def _make_session(timeout: float = 60.0) -> requests.Session:
    sess = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount('https://', adapter)
    sess.mount('http://', adapter)
    sess.headers.update(DEFAULT_HEADERS)
    sess.request_timeout = timeout
    return sess

def _normalize_ddp_series_id(series_id: str) -> str:
    s = series_id.strip().upper()
    if s.startswith('BOGZ1'):
        s = s.replace('BOGZ1', '', 1)
    if not s.endswith('.Q') and s != 'MORTGAGE30US':
        s = s + '.Q'
    return s

def _ddp_to_fred_id(ddp_id: str) -> str:
    s = _normalize_ddp_series_id(ddp_id)
    if s in FRED_ALIASES:
        return FRED_ALIASES[s]
    return 'BOGZ1' + s.replace('.', '')

def _fetch_fred_series(ddp_series_id: str, sess: requests.Session, timeout: float = 60.0) -> pd.DataFrame:
    fred_id = _ddp_to_fred_id(ddp_series_id)
    if ddp_series_id == 'MORTGAGE30US':
        url = FRED_MORTGAGE30US_CSV
    else:
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}'
    r = sess.get(url, timeout=timeout)
    r.raise_for_status()
    text = r.text
    if text.lstrip().lower().startswith('<!doctype html') or '<html' in text[:200].lower():
        raise RuntimeError(f'FRED returned HTML for {fred_id}')
    df = pd.read_csv(io.StringIO(text))
    lower_cols = {c.lower(): c for c in df.columns}
    date_col = lower_cols.get('date') or lower_cols.get('observation_date') or df.columns[0]
    value_col = next((c for c in df.columns if c != date_col), None)
    if value_col is None:
        raise RuntimeError(f'Unexpected FRED CSV for {fred_id}: cols={list(df.columns)}')
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.rename(columns={date_col: 'Date', value_col: _normalize_ddp_series_id(ddp_series_id)})
    q = df.set_index('Date').resample('QE-DEC')[_normalize_ddp_series_id(ddp_series_id)].mean().to_frame().reset_index()
    q[_normalize_ddp_series_id(ddp_series_id)] = pd.to_numeric(q[_normalize_ddp_series_id(ddp_series_id)], errors='coerce')
    return q

def _fetch_ddp_series(series_id: str, sess: requests.Session, timeout: float = 60.0) -> pd.DataFrame:
    sid = _normalize_ddp_series_id(series_id)
    params = DDP_PARAMS.copy(); params['series'] = sid
    try:
        r = sess.get(DDP_BASE, params=params, timeout=timeout)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        time_candidates = [c for c in df.columns if str(c).lower().startswith('time')]
        if not time_candidates:
            raise RuntimeError('no time col')
        time_col = time_candidates[0]
        val_col = [c for c in df.columns if c != time_col][0]
        df = df.rename(columns={time_col: 'Date', val_col: sid})
        if df['Date'].astype(str).str.contains('Q').any():
            df['Date'] = pd.PeriodIndex(df['Date'], freq='Q').to_timestamp(how='end')
        else:
            df['Date'] = pd.to_datetime(df['Date'])
        df[sid] = pd.to_numeric(df[sid], errors='coerce')
        return df
    except Exception:
        try:
            params2 = params.copy(); params2['series'] = f'Z1/Z1/{sid}'
            r2 = sess.get(DDP_BASE, params=params2, timeout=timeout)
            r2.raise_for_status()
            df2 = pd.read_csv(io.StringIO(r2.text))
            time_candidates = [c for c in df2.columns if str(c).lower().startswith('time')]
            if not time_candidates:
                raise RuntimeError('no time col 2')
            time_col = time_candidates[0]
            val_col = [c for c in df2.columns if c != time_col][0]
            df2 = df2.rename(columns={time_col: 'Date', val_col: sid})
            if df2['Date'].astype(str).str.contains('Q').any():
                df2['Date'] = pd.PeriodIndex(df2['Date'], freq='Q').to_timestamp(how='end')
            else:
                df2['Date'] = pd.to_datetime(df2['Date'])
            df2[sid] = pd.to_numeric(df2[sid], errors='coerce')
            return df2
        except Exception:
            return _fetch_fred_series(series_id, sess, timeout=timeout)

def _fetch_z1_inputs_for_us(timeout: float = 60.0) -> pd.DataFrame:
    sess = _make_session(timeout=timeout)
    dfs = []
    for key, (sid, cname) in Z1_SERIES.items():
        try:
            d = _fetch_ddp_series(sid, sess, timeout=timeout)
            d = d.rename(columns={_normalize_ddp_series_id(sid): cname})
            dfs.append(d)
            print(f"[proc01][Z1] OK {sid} -> {cname} rows={len(d)}")
        except Exception as e:
            print(f"[proc01][Z1][WARN] {sid} failed: {e}")
        time.sleep(0.3)
    merged = None
    for d in dfs:
        merged = d if merged is None else pd.merge(merged, d, on='Date', how='outer')
    if merged is None:
        merged = pd.DataFrame({'Date': []})
    # mortgage rate (monthly -> quarterly)
    try:
        fred = _fetch_fred_series('MORTGAGE30US', sess, timeout=timeout)
        valcol = _normalize_ddp_series_id('MORTGAGE30US')
        fred = fred.rename(columns={valcol: 'US_mortgage30yr_rate_pct'})
        fred['US_mortgage30yr_rate'] = fred['US_mortgage30yr_rate_pct'] / 100.0
        fred = fred.drop(columns=['US_mortgage30yr_rate_pct'])
        merged = pd.merge(merged, fred, on='Date', how='left')
        print(f"[proc01][FRED] OK MORTGAGE30US rows={len(fred)}")
    except Exception as e:
        print(f"[proc01][FRED][WARN] MORTGAGE30US failed: {e}")
    return merged.sort_values('Date').reset_index(drop=True)

def _annuity_k(annual_rate, wam_months):
    i = np.asarray(annual_rate, float) / 12.0
    n = np.asarray(wam_months, float)
    i = np.where(i <= -0.999999/12, -0.999999/12, i)
    n = np.where(n < 1, 1, n)
    with np.errstate(divide='ignore', invalid='ignore'):
        k = i / (1.0 - (1.0 + i) ** (-n))
    return np.where(np.isfinite(k), k, 1.0 / n)

def _dk_components_from_r_m(r, m):
    r = np.asarray(r, float); m = np.asarray(m, float)
    r_l, m_l = np.roll(r,1), np.roll(m,1)
    base = _annuity_k(r_l, m_l)
    rate = _annuity_k(r,   m_l) - base
    term = _annuity_k(r_l, m)   - base
    rate[0] = np.nan; term[0] = np.nan
    return rate, term

def build_us_k_from_z1_proxy(timeout: float = 90.0,
                             min_months: int = 60,
                             max_months: int = 420,
                             smooth: int = 4) -> bool:
    df = _fetch_z1_inputs_for_us(timeout=timeout)
    need = ['HH_home_mort_level','HH_home_mort_flow','US_mortgage30yr_rate']
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"[proc01][US][ERR] missing inputs: {missing}")
        return False
    lvl = df['HH_home_mort_level'].astype(float)
    flo = df['HH_home_mort_flow'].astype(float)
    # quarterly net repayment rate (use only when net flow negative)
    repay_rate_q = np.maximum(0.0, -flo) / np.where(lvl>0, lvl, np.nan)
    repay_rate_q = pd.Series(repay_rate_q).rolling(smooth, min_periods=1).mean().values
    repay_rate_y = 4.0 * repay_rate_q
    wam_months = 12.0 / np.maximum(repay_rate_y, 1e-6)
    wam_months = np.clip(wam_months, min_months, max_months)

    r_ann = df['US_mortgage30yr_rate'].astype(float).values
    dk_rate, dk_term = _dk_components_from_r_m(r_ann, wam_months)
    k_full = _annuity_k(r_ann, wam_months)

    out = pd.DataFrame({'Date': df['Date'], 'k': k_full, 'dk_rate': dk_rate, 'dk_term': dk_term})
    OUT_US_FULL.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_US_FULL, index=False)
    # legacy rate-only (for backward compatibility of downstream)
    out_rate_only = out[['Date','dk_rate']].rename(columns={'dk_rate':'US_dk_rate'})
    out_rate_only.to_csv(OUT_US_RATEONLY, index=False)
    print(f"[proc01][US] wrote {OUT_US_FULL.name} rows={len(out)} and {OUT_US_RATEONLY.name}")
    return True

def main() -> int:
    # （ここで既存の JP 側の処理を先に呼んでいてもOK）
    ok_us = build_us_k_from_z1_proxy(timeout=90.0)
    return 0 if ok_us else 1

if __name__ == '__main__':
    raise SystemExit(main())