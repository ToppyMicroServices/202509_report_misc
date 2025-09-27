#!/usr/bin/env python3
"""proc_code09_build_phi_capital_interaction_panel

Build panel for interaction scatter/regression of next‑quarter bank credit growth
on current off‑balance share (phi) and capital thinness.

Targets (written to data_processed/):
  - proc_code09_phi_capital_interaction_panel.csv

Schema (wide, one row per country×quarter):
  Date                (quarter end)
  country             ("US"|"JP")
  phi                 (US: phi_US_bank_off_raw; JP: phi_JP)
  CET1_RWA            (capital ratio, decimal or percent auto‑normalized)
  CET1_RWA_pct        (percent form)
  Thin_raw            (= 1 / CET1_RWA_pct  OR fallback (100 - CET1_RWA_pct)) used only if inversion chosen
  Thin                 final thinness metric (higher = thinner capital). Default: (100 - CET1_RWA_pct).
  credit_growth_ppYoY  (quarter t YoY pp change proxy)
  credit_growth_ppYoY_lead (lead 1 quarter, y variable)
  phi_x_Thin          (interaction regressor)
  Thin_tercile        (Low|Mid|High based on CET1_RWA_pct across both countries pooled)

Logic / Assumptions:
  * Uses existing processed outputs:
      proc_code04_US_phi_series_raw.csv (US phi)
      proc_code02_JP_phi_quarterly.csv  (JP phi)
      proc_code03_*_DSR_CreditGrowth_panel.csv for credit growth proxy columns (CreditGrowth_ppYoY)
  * CET1/RWA source CSVs expected under data_raw/ with names:
      CET1_RWA_US_WB.csv, CET1_RWA_JP_WB.csv (World Bank style)
    They may have headers like DATE,value or observation_date,value. Values may be in percent (e.g. 12.5) or ratio (0.125).
    Heuristic: if median<1.2 treat as ratio -> convert *100.
  * Quarter alignment: all series mapped to quarter end (Period('Q')->timestamp('Q')).
  * Leading credit growth: for each (country,Date) define credit_growth_ppYoY_lead = value at next quarter; last quarter per country has NaN lead and is dropped for regression convenience.
  * Thinness definition: Thin = (100 - CET1_RWA_pct). (Higher value means thinner capitalization.) Alternate inverse form left in Thin_raw when ratio form plausible.
  * Terciles computed on CET1_RWA_pct (NOT Thin) so that Low tercile = low capital ratio (thin), High tercile = strong capital. This matches intuitive labeling in plots.

CLI:
  python code/proc_code09_build_phi_capital_interaction_panel.py
  python code/proc_code09_build_phi_capital_interaction_panel.py --alt-thin-inverse

"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data_processed'
RAW  = ROOT / 'data_raw'
OUT  = PROC / 'proc_code09_phi_capital_interaction_panel.csv'

US_PHI_FILE = PROC / 'proc_code04_US_phi_series_raw.csv'
JP_PHI_FILE = PROC / 'proc_code02_JP_phi_quarterly.csv'
US_PANEL    = PROC / 'proc_code03_US_DSR_CreditGrowth_panel.csv'
JP_PANEL    = PROC / 'proc_code03_JP_DSR_CreditGrowth_panel.csv'
US_CET1     = RAW  / 'CET1_RWA_US_WB.csv'
JP_CET1     = RAW  / 'CET1_RWA_JP_WB.csv'


def _log(m: str):
    print(f"[PROC09] {m}")


def _load_cet1(path: Path, country: str) -> pd.DataFrame:
    if not path.exists():
        _log(f"WARN: CET1 file missing {path}")
        return pd.DataFrame(columns=["Date","CET1_RWA_pct","country"])
    df = pd.read_csv(path)
    # Detect date col
    dcol = None
    for c in df.columns:
        lc = c.lower()
        if lc in ('date','observation_date','time','period'):
            dcol = c; break
    if dcol is None:
        dcol = df.columns[0]
    vcol = [c for c in df.columns if c != dcol][0]
    df[dcol] = pd.to_datetime(df[dcol], errors='coerce')
    df[vcol] = pd.to_numeric(df[vcol], errors='coerce')
    df = df.dropna(subset=[dcol, vcol])[[dcol, vcol]].rename(columns={dcol:'Date', vcol:'val'})
    # Heuristic percent vs ratio
    med = df['val'].median()
    if med < 1.2:  # ratio (e.g. 0.12) -> convert to percent
        df['CET1_RWA_pct'] = df['val'] * 100.0
    else:
        df['CET1_RWA_pct'] = df['val']
    df['Date'] = df['Date'].dt.to_period('Q').dt.to_timestamp('Q')
    df = df[['Date','CET1_RWA_pct']].drop_duplicates(subset=['Date']).sort_values('Date')
    df['country'] = country
    return df


def _load_phi_us() -> pd.DataFrame:
    if not US_PHI_FILE.exists():
        return pd.DataFrame(columns=['Date','phi'])
    df = pd.read_csv(US_PHI_FILE, parse_dates=['Date'])
    if 'phi_US_bank_off_raw' not in df.columns:
        return pd.DataFrame(columns=['Date','phi'])
    out = df[['Date','phi_US_bank_off_raw']].rename(columns={'phi_US_bank_off_raw':'phi'}).copy()
    out['Date'] = out['Date'].dt.to_period('Q').dt.to_timestamp('Q')
    out = out.drop_duplicates(subset=['Date']).sort_values('Date')
    out['country'] = 'US'
    return out


def _load_phi_jp() -> pd.DataFrame:
    if not JP_PHI_FILE.exists():
        return pd.DataFrame(columns=['Date','phi'])
    df = pd.read_csv(JP_PHI_FILE, parse_dates=['Date'])
    if 'phi_JP' not in df.columns:
        return pd.DataFrame(columns=['Date','phi'])
    out = df[['Date','phi_JP']].rename(columns={'phi_JP':'phi'}).copy()
    out['Date'] = out['Date'].dt.to_period('Q').dt.to_timestamp('Q')
    out = out.drop_duplicates(subset=['Date']).sort_values('Date')
    out['country'] = 'JP'
    return out


def _load_credit_panel(panel_path: Path, country: str) -> pd.DataFrame:
    if not panel_path.exists():
        return pd.DataFrame(columns=['Date','CreditGrowth_ppYoY'])
    try:
        df = pd.read_csv(panel_path, index_col=0)
    except Exception:
        return pd.DataFrame(columns=['Date','CreditGrowth_ppYoY'])
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    # Accept either CreditGrowth_ppYoY or credit growth alt naming
    col = None
    for c in df.columns:
        if c.lower() == 'creditgrowth_ppyoy':
            col = c; break
    if col is None:
        return pd.DataFrame(columns=['Date','CreditGrowth_ppYoY'])
    sub = df[[col]].rename(columns={col:'CreditGrowth_ppYoY'})
    sub = sub.dropna().copy()
    sub['Date'] = sub.index.to_period('Q').to_timestamp('Q')
    sub = sub[['Date','CreditGrowth_ppYoY']].drop_duplicates(subset=['Date']).sort_values('Date')
    sub['country'] = country
    return sub


def build_panel(alt_thin_inverse: bool=False) -> pd.DataFrame:
    # Load components
    us_phi = _load_phi_us()
    jp_phi = _load_phi_jp()
    us_cap = _load_cet1(US_CET1, 'US')
    jp_cap = _load_cet1(JP_CET1, 'JP')
    us_cg  = _load_credit_panel(US_PANEL, 'US')
    jp_cg  = _load_credit_panel(JP_PANEL, 'JP')
    pieces = []
    for phi_df, cap_df, cg_df in ((us_phi, us_cap, us_cg),(jp_phi,jp_cap,jp_cg)):
        if phi_df.empty or cap_df.empty or cg_df.empty:
            continue
        # Merge on quarter end intersection
        m = phi_df.merge(cap_df, on=['Date','country'], how='inner')
        m = m.merge(cg_df, on=['Date','country'], how='inner')
        pieces.append(m)
    if not pieces:
        return pd.DataFrame()
    panel = pd.concat(pieces, ignore_index=True)
    # Thinness definitions
    if alt_thin_inverse:
        # Use inverse ratio style: Thin_raw = 1 / CET1_pct, scaled to mean 0 center
        panel['Thin_raw'] = 1.0 / panel['CET1_RWA_pct'].replace(0, np.nan)
        # Standardize (z-score) for stability
        panel['Thin'] = (panel['Thin_raw'] - panel['Thin_raw'].mean()) / panel['Thin_raw'].std(ddof=0)
    else:
        panel['Thin_raw'] = 100.0 - panel['CET1_RWA_pct']
        panel['Thin'] = panel['Thin_raw']  # already interpretable in pp
    # Lead credit growth (within country)
    panel = panel.sort_values(['country','Date'])
    panel['credit_growth_ppYoY'] = panel['CreditGrowth_ppYoY']
    panel['credit_growth_ppYoY_lead'] = panel.groupby('country')['credit_growth_ppYoY'].shift(-1)
    panel['phi_x_Thin'] = panel['phi'] * panel['Thin']
    # Drop last per country (lead NaN)
    panel = panel.dropna(subset=['credit_growth_ppYoY_lead'])
    # Terciles on CET1_RWA_pct pooled
    pct_vals = panel['CET1_RWA_pct']
    q1, q2 = pct_vals.quantile([1/3, 2/3])
    def tercile(v):
        if v <= q1: return 'Low'   # low capital (thin)
        if v <= q2: return 'Mid'
        return 'High'
    panel['Thin_tercile'] = panel['CET1_RWA_pct'].apply(tercile)
    panel = panel[['Date','country','phi','CET1_RWA_pct','Thin_raw','Thin','credit_growth_ppYoY','credit_growth_ppYoY_lead','phi_x_Thin','Thin_tercile']]
    panel = panel.sort_values(['country','Date']).reset_index(drop=True)
    return panel


def main():
    ap = argparse.ArgumentParser(description='Build phi×capital interaction panel (proc_code09)')
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--alt-thin-inverse', action='store_true', help='Use inverse CET1 (1/CET1_pct) standardized as Thin')
    args = ap.parse_args()
    panel = build_panel(alt_thin_inverse=args.alt_thin_inverse)
    if panel.empty:
        _log('No rows built (missing inputs).')
        return 1
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)
    _log(f'WROTE {out_path.name} rows={len(panel)} countries={panel.country.nunique()} date_range={panel.Date.min().date()}->{panel.Date.max().date()}')
    return 0

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
