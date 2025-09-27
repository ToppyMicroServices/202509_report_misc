#!/usr/bin/env python3
"""proc_code08_build_dsr_k_quantity_decomposition

DSR 変化を以下の一次近似で寄与分解 (四半期):

  DSR = k * (Debt/Income)  (概念式; 実際の BIS DSR は広義だが近似とする)
  ΔDSR ≈ k̄ * Δ(D/I) + (D/I)̄ * Δk
  さらに Δk ≈ dk_rate + dk_term (既存 proc_code01 の近似寄与) + resid_k

したがって DSR の変化 (pp) を:
  quantity_contrib_pp = k_mid * Δ(D/I) * 100
  k_rate_contrib_pp   = (D/I)_mid * dk_rate * 100
  k_term_contrib_pp   = (D/I)_mid * dk_term * 100
  residual_pp         = ΔDSR_pp - (quantity + k_rate + k_term)

入出力:
  入力:
    - US: proc_code03_US_DSR_CreditGrowth_panel.csv (DSR_pct 列)
           proc_code01_k_decomposition_US_quarterly_RATE_ONLY.csv (k, dk_rate, dk_term(=0))
    - JP: proc_code03_JP_DSR_CreditGrowth_panel.csv (任意, legacy由来) + proc_code01_k_decomposition_JP_quarterly.csv
  出力:
    - data_processed/proc_code08_US_DSR_k_quantity_decomposition.csv
    - data_processed/proc_code08_JP_DSR_k_quantity_decomposition.csv (JP両方揃えば)

注意:
  * DSR_pct は百分率→小数化 ( /100 )
  * (Debt/Income) 近似 = DSR_frac / k (k>0 の場合) (単純化)
  * 初期行の寄与は 0
  * 欠損/非現実値(k<=0) 行はスキップ (前後で寄与0、比率NaN)
  * k 差分は実際の Δk (k_t - k_{t-1}); dk_rate/dk_term は proc_code01 の近似寄与をそのまま使用

CLI 例:
  python code/proc_code08_build_dsr_k_quantity_decomposition.py
  python code/proc_code08_build_dsr_k_quantity_decomposition.py --skip-jp
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data_processed'

US_DSR_PANEL = PROC / 'proc_code03_US_DSR_CreditGrowth_panel.csv'
JP_DSR_PANEL = PROC / 'proc_code03_JP_DSR_CreditGrowth_panel.csv'
US_K_FILE    = PROC / 'proc_code01_k_decomposition_US_quarterly_RATE_ONLY.csv'
JP_K_FILE    = PROC / 'proc_code01_k_decomposition_JP_quarterly.csv'


def _log(m):
    print(f"[PROC08] {m}")


def load_panel(p: Path):
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, index_col=0)
    except Exception:
        return None
    df.index = pd.to_datetime(df.index)
    # Quarter-end normalize to align with k decomposition (which uses quarter end timestamps)
    df.index = df.index.to_period('Q').to_timestamp('Q')
    if 'DSR_pct' not in df.columns:
        return None
    out = df[['DSR_pct']].copy()
    out = out[~out.index.duplicated()].sort_index()
    return out


def load_k(p: Path):
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=['Date'])
    except Exception:
        return None
    req = {'Date','k','dk_rate'}
    if not req.issubset(df.columns):
        return None
    if 'dk_term' not in df.columns:
        df['dk_term'] = 0.0
    df = df.sort_values('Date').drop_duplicates(subset=['Date'])
    df = df.set_index('Date')
    # Normalize to quarter-end midnight (discard potential 23:59:59.999999999 artifacts)
    df.index = df.index.to_period('Q').to_timestamp('Q')
    df.index = pd.to_datetime(df.index)  # ensure standard datetime64[ns]
    return df[['k','dk_rate','dk_term']]


def build(country: str, panel: pd.DataFrame, kdf: pd.DataFrame, out_path: Path):
    # Align by quarter end dates present in both (inner join)
    df = panel.join(kdf, how='inner')
    if df.empty:
        _log(f"{country} no overlap between DSR panel and k data -> skip")
        return False
    df['DSR_frac'] = df['DSR_pct'] / 100.0
    # Debt/Income proxy (avoid division by zero/neg)
    df['debt_income_ratio'] = np.where(df['k'] > 0, df['DSR_frac'] / df['k'], np.nan)
    # Prepare contribution columns
    cols = ['delta_DSR_pp','quantity_contrib_pp','k_rate_contrib_pp','k_term_contrib_pp','residual_pp']
    for c in cols:
        df[c] = 0.0
    prev = None
    for idx, row in df.iterrows():
        if prev is None:
            prev = row
            continue
        d_curr = row['DSR_frac']; d_prev = prev['DSR_frac']
        k_curr = row['k']; k_prev = prev['k']
        di_curr = row['debt_income_ratio']; di_prev = prev['debt_income_ratio']
        if not all(np.isfinite([d_curr,d_prev,k_curr,k_prev])):
            prev = row; continue
        delta_dsr = (d_curr - d_prev)
        dk = k_curr - k_prev
        k_mid = 0.5 * (k_curr + k_prev)
        di_mid = 0.5 * (di_curr + di_prev) if np.isfinite(di_curr) and np.isfinite(di_prev) else np.nan
        delta_di = (di_curr - di_prev) if np.isfinite(di_curr) and np.isfinite(di_prev) else 0.0
        # Quantity contribution
        quantity_contrib = k_mid * delta_di if np.isfinite(k_mid) else 0.0
        # k contributions via rate/term dk_* (already Δk の各寄与) スケール: (D/I)_mid
        dk_rate = row['dk_rate']
        dk_term = row['dk_term']
        k_rate_contrib = (di_mid * dk_rate) if np.isfinite(di_mid) else 0.0
        k_term_contrib = (di_mid * dk_term) if np.isfinite(di_mid) else 0.0
        residual = delta_dsr - (quantity_contrib + k_rate_contrib + k_term_contrib)
        df.loc[idx, 'delta_DSR_pp'] = delta_dsr * 100.0
        df.loc[idx, 'quantity_contrib_pp'] = quantity_contrib * 100.0
        df.loc[idx, 'k_rate_contrib_pp'] = k_rate_contrib * 100.0
        df.loc[idx, 'k_term_contrib_pp'] = k_term_contrib * 100.0
        df.loc[idx, 'residual_pp'] = residual * 100.0
        prev = row
    # Output tidy
    out = df[['DSR_pct','k','debt_income_ratio',
              'delta_DSR_pp','quantity_contrib_pp','k_rate_contrib_pp','k_term_contrib_pp','residual_pp']].reset_index().rename(columns={'index':'Date'})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    _log(f"{country} wrote {out_path.name} rows={len(out)} residual_max_abs={out['residual_pp'].abs().max():.3e}pp")
    return True


def main():
    ap = argparse.ArgumentParser(description='Build DSR decomposition (quantity vs k: rate/term) – proc_code08')
    ap.add_argument('--skip-us', action='store_true')
    ap.add_argument('--skip-jp', action='store_true')
    args = ap.parse_args()

    any_ok = False
    if not args.skip_us:
        panel = load_panel(US_DSR_PANEL)
        kdf = load_k(US_K_FILE)
        if panel is None or kdf is None:
            _log('US missing panel or k -> skip')
        else:
            any_ok |= build('US', panel, kdf, PROC/'proc_code08_US_DSR_k_quantity_decomposition.csv')

    if not args.skip_jp:
        panel = load_panel(JP_DSR_PANEL)
        kdf = load_k(JP_K_FILE)
        if panel is None or kdf is None:
            _log('JP missing panel or k -> skip')
        else:
            any_ok |= build('JP', panel, kdf, PROC/'proc_code08_JP_DSR_k_quantity_decomposition.csv')

    if not any_ok:
        _log('No decomposition built.')
        return 1
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
