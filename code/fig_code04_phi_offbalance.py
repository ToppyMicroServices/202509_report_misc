#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fig_code04_phi_offbalance_fixed_v2.py
#
# Fixed-style figure; also auto-detects JP files and columns.

import pandas as pd, matplotlib.pyplot as plt, re, argparse
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
            r'^phi[_ ]?US$',
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
    if not (pre or post or single):
        merged = Path('data_processed/proc_merge_US_JP_phi_household.csv')
        if merged.exists():
            single = merged
    return pre, post, single


US_PRE  = Path('data_processed/proc_code04_US_phi_series_observed_pre.csv')
US_POST = Path('data_processed/proc_code04_US_phi_series_observed_post.csv')

def _filter_after(x, series, start_ts):
    mask = x >= start_ts
    return x[mask], series.loc[mask]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-date', default='1980-01-01', help='Plot start date (default 1980-01-01).')
    ap.add_argument('--out', default='figures/US_JP_phi_offbalance_fixed.png')
    args = ap.parse_args()
    start_ts = pd.Timestamp(args.start_date)
    if not US_PRE.exists() or not US_POST.exists():
        raise SystemExit('Missing US pre/post CSVs under data_processed/. Run the builder first.')
    x_us_pre,  df_us_pre  = _read_csv_flex(US_PRE)
    x_us_post, df_us_post = _read_csv_flex(US_POST)
    col_us_pre  = _find_phi_col(df_us_pre,  'US')
    col_us_post = _find_phi_col(df_us_post, 'US')

    fig, ax = plt.subplots(figsize=(12,5))
    if col_us_pre:
        x_us_pre_f, y_us_pre = _filter_after(x_us_pre, df_us_pre[col_us_pre], start_ts)
        ax.plot(x_us_pre_f,  y_us_pre,  '-o', ms=3, lw=2.0, color='tab:orange', label='United States φ (pre)')
    if col_us_post:
        x_us_post_f, y_us_post = _filter_after(x_us_post, df_us_post[col_us_post], start_ts)
        ax.plot(x_us_post_f, y_us_post, '--o', ms=3, lw=1.8, color='tab:orange', alpha=0.85, label='United States φ (post)')

    # JP
    jp_pre, jp_post, jp_single = _detect_jp_files()
    if jp_single is not None:
        x_jp, df_jp = _read_csv_flex(jp_single)
        col_jp = _find_phi_col(df_jp, 'JP')
        if col_jp:
            x_jp_f, y_jp = _filter_after(x_jp, df_jp[col_jp], start_ts)
            if len(x_jp_f) > 0:
                # Build continuous quarterly line by forward filling; markers only at observed points
                s_obs = pd.Series(pd.to_numeric(y_jp, errors='coerce').values, index=x_jp_f).dropna()
                q_index = pd.date_range(start=x_jp_f.min(), end=x_jp_f.max(), freq='Q-DEC')
                s_line = s_obs.reindex(q_index).ffill()
                ax.plot(s_line.index, s_line.values, '-', lw=1.9, color='tab:blue', label='Japan φ')
                if not s_obs.empty:
                    ax.plot(s_obs.index, s_obs.values, 'o', ms=3, color='tab:blue', label='_nolegend_')
    else:
        if jp_pre is not None:
            x_jp_pre, df_jp_pre = _read_csv_flex(jp_pre)
            col_jp_pre = _find_phi_col(df_jp_pre, 'JP')
            if col_jp_pre:
                x_jp_pre_f, y_jp_pre = _filter_after(x_jp_pre, df_jp_pre[col_jp_pre], start_ts)
                if len(x_jp_pre_f) > 0:
                    q_idx = pd.date_range(start=x_jp_pre_f.min(), end=x_jp_pre_f.max(), freq='Q-DEC')
                    s = pd.Series(y_jp_pre.values, index=x_jp_pre_f)
                    s_full = s.reindex(q_idx).interpolate(limit_direction='both')
                    ax.plot(s_full.index, s_full.values, '-', lw=1.4, color='tab:blue', label='Japan φ (pre)')
                    ax.plot(x_jp_pre_f, y_jp_pre, 'o', ms=3, color='tab:blue', label=None)
        if jp_post is not None:
            x_jp_post, df_jp_post = _read_csv_flex(jp_post)
            col_jp_post = _find_phi_col(df_jp_post, 'JP')
            if col_jp_post:
                x_jp_post_f, y_jp_post = _filter_after(x_jp_post, df_jp_post[col_jp_post], start_ts)
                if len(x_jp_post_f) > 0:
                    s_obs = pd.Series(pd.to_numeric(y_jp_post, errors='coerce').values, index=x_jp_post_f).dropna()
                    q_index = pd.date_range(start=x_jp_post_f.min(), end=x_jp_post_f.max(), freq='Q-DEC')
                    s_line = s_obs.reindex(q_index).ffill()
                    ax.plot(s_line.index, s_line.values, '-', lw=1.9, color='tab:blue', label='Japan φ (post)')
                    if not s_obs.empty:
                        ax.plot(s_obs.index, s_obs.values, 'o', ms=3, color='tab:blue', label='_nolegend_')
                    q_idx = pd.date_range(start=x_jp_post_f.min(), end=x_jp_post_f.max(), freq='Q-DEC')
                    s = pd.Series(y_jp_post.values, index=x_jp_post_f)
                    s_full = s.reindex(q_idx).interpolate(limit_direction='both')
                    ax.plot(s_full.index, s_full.values, '-', lw=1.4, color='tab:blue', label='Japan φ (post)')
                    ax.plot(x_jp_post_f, y_jp_post, 'o', ms=3, color='tab:blue', label=None)

    ax.axvline(pd.Timestamp('2010-03-31'), linestyle='--', alpha=0.5)
    ax.set_title('Off-balance share φ — US & JP (strict splice)')
    ax.set_xlabel('Date'); ax.set_ylabel('φ (share)'); ax.set_ylim(0,1)
    ax.grid(True, alpha=0.25); ax.legend(loc='upper left')
    Path('figures').mkdir(parents=True, exist_ok=True)
    out_png = Path(args.out)
    fig.tight_layout(); fig.savefig(out_png, dpi=220)
    print(f'[OK] saved {out_png}')

if __name__ == '__main__':
    main()
