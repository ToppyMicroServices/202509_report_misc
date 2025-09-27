#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fig_code05_phi_offbalance_repro_strict_v2.py
#
# Plots US (required) and JP (optional) off-balance share φ with strict splice.
# US inputs (required):
#   data_processed/proc_code04_US_phi_series_observed_pre.csv
#   data_processed/proc_code04_US_phi_series_observed_post.csv
# JP inputs (optional, auto-detected):
#   data_processed/proc_code04_JP_phi_series_observed_pre.csv / _post.csv
#   or a single JP observed file matching *JP*observed*.csv

import pandas as pd, matplotlib.pyplot as plt, re
from pathlib import Path
from glob import glob

def _read_csv_flex(path):
    df = pd.read_csv(path)
    # Date column candidates
    for c in ['Date','date','observation_date','DATE']:
        if c in df.columns:
            x = pd.to_datetime(df[c].astype(str), errors='coerce')
            break
    else:
        # fallback: first column is date
        first = df.columns[0]
        x = pd.to_datetime(df[first].astype(str), errors='coerce')
    return x, df

def _find_phi_col(df, country='US'):
    cols = list(df.columns)
    pats = []
    if country.upper()=='US':
        pats = [
            r'^phi[_ ]?US[_ ]?observed[_ ]?proxy$',
            r'^phi[_ ]?US',
            r'^phi.*observed.*proxy$',
            r'^phi$'
        ]
    else:
        pats = [
            r'^phi[_ ]?JP[_ ]?observed[_ ]?proxy$',
            r'^phi[_ ]?JP$',
            r'^phi[_ ]?JP',
            r'^phi_JP$',
            r'^phi.*observed.*proxy$',
            r'^phi$'
        ]
    for p in pats:
        rgx = re.compile(p, re.I)
        for c in cols:
            if rgx.match(c):
                return c
    # last resort: any column starting with 'phi'
    for c in cols:
        if str(c).lower().startswith('phi'):
            return c
    return None

def _detect_jp_files():
    # Preferred pre/post
    pre_candidates  = sorted(glob('data_processed/*JP*pre*.csv')) + sorted(glob('data_processed/proc_code04_JP_phi_series_observed_pre.csv'))
    post_candidates = sorted(glob('data_processed/*JP*post*.csv')) + sorted(glob('data_processed/proc_code04_JP_phi_series_observed_post.csv'))
    # Single-file fallbacks
    single_candidates = sorted(glob('data_processed/*JP*observed*.csv')) + sorted(glob('data_processed/*JP*.csv'))
    pre = Path(pre_candidates[-1]) if pre_candidates else None
    post = Path(post_candidates[-1]) if post_candidates else None
    single = None if (pre or post) else (Path(single_candidates[-1]) if single_candidates else None)
    # Fallback: merged US-JP file containing phi_JP
    if not (pre or post or single):
        merged = Path('data_processed/proc_merge_US_JP_phi_household.csv')
        if merged.exists():
            single = merged
    return pre, post, single


US_PRE  = Path('data_processed/proc_code04_US_phi_series_observed_pre.csv')
US_POST = Path('data_processed/proc_code04_US_phi_series_observed_post.csv')

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--jp-fill-forward', action='store_true', help='Forward-fill sparse JP phi to quarterly for smoother line.')
    ap.add_argument('--out', default='figures/phi_observed_pre_post_US_JP.png')
    ap.add_argument('--start-date', default='1980-01-01', help='Plot start date (default 1980-01-01).')
    args = ap.parse_args()
    if not US_PRE.exists() or not US_POST.exists():
        raise SystemExit('Missing US pre/post CSVs under data_processed/. Run the builder first.')
    start_ts = pd.Timestamp(args.start_date)
    x_us_pre,  df_us_pre  = _read_csv_flex(US_PRE)
    x_us_post, df_us_post = _read_csv_flex(US_POST)
    col_us_pre  = _find_phi_col(df_us_pre,  'US')
    col_us_post = _find_phi_col(df_us_post, 'US')

    fig, ax = plt.subplots(figsize=(12,5))
    if col_us_pre:
        m = x_us_pre >= start_ts
        ax.plot(x_us_pre[m],  df_us_pre.loc[m, col_us_pre],  lw=2.0, marker='o', ms=3.5, color='tab:orange', label='United States φ (pre)')
    if col_us_post:
        m = x_us_post >= start_ts
        ax.plot(x_us_post[m], df_us_post.loc[m, col_us_post], lw=1.8, linestyle='--', marker='o', ms=3.5, color='tab:orange', alpha=0.85, label='United States φ (post)')

    # --- JP (auto-detect) ---
    jp_pre, jp_post, jp_single = _detect_jp_files()
    if jp_single is not None:
        x_jp, df_jp = _read_csv_flex(jp_single)
        col_jp = _find_phi_col(df_jp, 'JP')
        if col_jp:
            keep = x_jp >= start_ts
            x_jp_k = x_jp[keep]
            y = df_jp.loc[keep, col_jp]
            if args.jp_fill_forward and not x_jp_k.empty:
                q_index = pd.date_range(start=x_jp_k.min(), end=x_jp_k.max(), freq='Q')
                y_full = pd.Series(y.values, index=x_jp_k).reindex(q_index).ffill()
                ax.plot(q_index, y_full, '-', lw=1.9, marker='o', ms=3.5, color='tab:blue', label='Japan φ (ffill)')
            else:
                ax.plot(x_jp_k, y, '-', lw=1.9, marker='o', ms=3.5, color='tab:blue', label='Japan φ')
    else:
        if jp_pre is not None:
            x_jp_pre, df_jp_pre = _read_csv_flex(jp_pre)
            col_jp_pre = _find_phi_col(df_jp_pre, 'JP')
            if col_jp_pre:
                y_pre = df_jp_pre[col_jp_pre]
                keep = x_jp_pre >= start_ts
                x_pre = x_jp_pre[keep]
                y_pre_k = y_pre.loc[keep]
                if args.jp_fill_forward and not x_pre.empty:
                    q_index = pd.date_range(start=x_pre.min(), end=x_pre.max(), freq='Q')
                    y_pre_full = pd.Series(y_pre_k.values, index=x_pre).reindex(q_index).ffill()
                    ax.plot(q_index, y_pre_full, '-', lw=1.9, marker='o', ms=3.5, color='tab:blue', label='Japan φ (pre, ffill)')
                else:
                    ax.plot(x_pre, y_pre_k, '-', lw=1.9, marker='o', ms=3.5, color='tab:blue', label='Japan φ (pre)')
        if jp_post is not None:
            x_jp_post, df_jp_post = _read_csv_flex(jp_post)
            col_jp_post = _find_phi_col(df_jp_post, 'JP')
            if col_jp_post:
                y_post = df_jp_post[col_jp_post]
                keep = x_jp_post >= start_ts
                x_post = x_jp_post[keep]
                y_post_k = y_post.loc[keep]
                if args.jp_fill_forward and not x_post.empty:
                    q_index = pd.date_range(start=x_post.min(), end=x_post.max(), freq='Q')
                    y_post_full = pd.Series(y_post_k.values, index=x_post).reindex(q_index).ffill()
                    ax.plot(q_index, y_post_full, '-', lw=1.9, marker='o', ms=3.5, color='tab:blue', label='Japan φ (post, ffill)')
                else:
                    ax.plot(x_post, y_post_k, '-', lw=1.9, marker='o', ms=3.5, color='tab:blue', label='Japan φ (post)')

    ax.axvline(pd.Timestamp('2010-03-31'), linestyle='--', alpha=0.5)
    ax.set_title('Off-balance share φ — US & JP (strict splice)')
    ax.set_xlabel('Date'); ax.set_ylabel('φ (share)'); ax.set_ylim(0,1)
    ax.grid(True, alpha=0.3); ax.legend(loc='best')

    Path('figures').mkdir(exist_ok=True, parents=True)
    out_png = Path(args.out)
    fig.tight_layout(); fig.savefig(out_png, dpi=220)
    print(f'[OK] saved {out_png}')

if __name__ == '__main__':
    main()
