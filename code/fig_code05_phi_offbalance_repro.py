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
import matplotlib.dates as mdates
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
    preferred_single = Path('data_processed/proc_code02_JP_phi_quarterly.csv')
    if preferred_single.exists():
        return None, None, preferred_single
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
    ap.add_argument('--jp-csv', default=None, help='Explicit JP phi CSV to use (e.g., data_processed/proc_code02_JP_phi_quarterly.csv).')
    ap.add_argument('--jp-fill-forward', action='store_true', help='Forward-fill sparse JP phi to quarterly (Q-DEC) for smoother line.')
    ap.add_argument('--out', default='figures/fig_code05_phi_observed_pre_post_US_JP.png')
    ap.add_argument('--start-date', default='1980-01-01', help='Plot start date (default 1980-01-01).')
    ap.add_argument('--jp-right-axis', action='store_true', help='Plot Japan φ on right y-axis (e.g., focus on ~0.1 range).')
    ap.add_argument('--markevery', type=int, default=5, help='Marker thinning interval (default: 5).')
    args = ap.parse_args()
    if not US_PRE.exists() or not US_POST.exists():
        raise SystemExit('Missing US pre/post CSVs under data_processed/. Run the builder first.')
    start_ts = pd.Timestamp(args.start_date)
    x_us_pre,  df_us_pre  = _read_csv_flex(US_PRE)
    x_us_post, df_us_post = _read_csv_flex(US_POST)
    col_us_pre  = _find_phi_col(df_us_pre,  'US')
    col_us_post = _find_phi_col(df_us_post, 'US')

    fig, ax = plt.subplots(figsize=(8.4,5))
    # Colors/markers for better print/mono distinction
    color_us_pre = '#D55E00'   # vermillion
    color_us_post = '#E69F00'  # orange
    color_jp = 'tab:blue'
    m_us_pre = 'o'; m_us_post = 's'; m_jp = '^'
    me = max(1, int(args.markevery))
    if col_us_pre:
        m = x_us_pre >= start_ts
        y = pd.to_numeric(df_us_pre.loc[m, col_us_pre], errors='coerce')
    ax.plot(x_us_pre[m],  y,  lw=2.0, marker=m_us_pre, markevery=me, ms=5, color=color_us_pre, label='United States φ (pre)')
    if col_us_post:
        m = x_us_post >= start_ts
        y = pd.to_numeric(df_us_post.loc[m, col_us_post], errors='coerce')
    ax.plot(x_us_post[m], y, lw=2.0, linestyle='--', marker=m_us_post, markevery=me, ms=5, color=color_us_post, alpha=0.95, label='United States φ (post)')

    # --- JP (auto / explicit) ---
    if args.jp_csv:
        jp_pre = jp_post = None
        jp_single = Path(args.jp_csv)
        if not jp_single.exists():
            print(f"[WARN] --jp-csv not found: {jp_single}")
            jp_single = None
    else:
        jp_pre, jp_post, jp_single = _detect_jp_files()
    # Target axis for JP (left by default; right if requested)
    ax_jp = ax
    if args.jp_right_axis:
        ax_jp = ax.twinx()
        ax_jp.set_ylabel('Japan φ (share)')
        ax_jp.set_ylim(0, 0.3)

    if jp_single is not None:
        x_jp, df_jp = _read_csv_flex(jp_single)
        col_jp = _find_phi_col(df_jp, 'JP')
        if col_jp:
            keep = x_jp >= start_ts
            x_jp_k = x_jp[keep]
            y = pd.to_numeric(df_jp.loc[keep, col_jp], errors='coerce')
            y_nn = y.dropna()
            print(f"[INFO] JP(single) src={jp_single.name}, col={col_jp}, points={y_nn.shape[0]}")
            if args.jp_fill_forward and not x_jp_k.empty:
                s_obs = pd.Series(y.values, index=x_jp_k).dropna()
                q_index = pd.date_range(start=x_jp_k.min(), end=x_jp_k.max(), freq='QE-DEC')
                y_full = s_obs.reindex(q_index).ffill()
                ax_jp.plot(q_index, y_full, '-', lw=2.0, marker=m_jp, markevery=me, ms=5, color=color_jp, label='Japan φ')
            else:
                mask = y.notna().values
                if mask.any():
                    ax_jp.plot(x_jp_k[mask], y[mask], '-', lw=2.0, marker=m_jp, markevery=me, ms=5, color=color_jp, label='Japan φ')
    else:
        if jp_pre is not None:
            x_jp_pre, df_jp_pre = _read_csv_flex(jp_pre)
            col_jp_pre = _find_phi_col(df_jp_pre, 'JP')
            if col_jp_pre:
                y_pre = pd.to_numeric(df_jp_pre[col_jp_pre], errors='coerce')
                keep = x_jp_pre >= start_ts
                x_pre = x_jp_pre[keep]
                y_pre_k = y_pre.loc[keep]
                print(f"[INFO] JP(pre) src={jp_pre.name}, col={col_jp_pre}, points={y_pre_k.dropna().shape[0]}")
                if args.jp_fill_forward and not x_pre.empty:
                    s_obs = pd.Series(y_pre_k.values, index=x_pre).dropna()
                    q_index = pd.date_range(start=x_pre.min(), end=x_pre.max(), freq='QE-DEC')
                    y_pre_full = s_obs.reindex(q_index).ffill()
                    ax_jp.plot(q_index, y_pre_full, '-', lw=2.0, marker=m_jp, markevery=me, ms=5, color=color_jp, label='Japan φ (pre)')
                else:
                    mask = y_pre_k.notna().values
                    if mask.any():
                        ax_jp.plot(x_pre[mask], y_pre_k[mask], '-', lw=2.0, marker=m_jp, markevery=me, ms=5, color=color_jp, label='Japan φ (pre)')
        if jp_post is not None:
            x_jp_post, df_jp_post = _read_csv_flex(jp_post)
            col_jp_post = _find_phi_col(df_jp_post, 'JP')
            if col_jp_post:
                y_post = pd.to_numeric(df_jp_post[col_jp_post], errors='coerce')
                keep = x_jp_post >= start_ts
                x_post = x_jp_post[keep]
                y_post_k = y_post.loc[keep]
                print(f"[INFO] JP(post) src={jp_post.name}, col={col_jp_post}, points={y_post_k.dropna().shape[0]}")
                if args.jp_fill_forward and not x_post.empty:
                    s_obs = pd.Series(y_post_k.values, index=x_post).dropna()
                    q_index = pd.date_range(start=x_post.min(), end=x_post.max(), freq='QE-DEC')
                    y_post_full = s_obs.reindex(q_index).ffill()
                    ax_jp.plot(q_index, y_post_full, '-', lw=2.0, marker=m_jp, markevery=me, ms=5, color=color_jp, label='Japan φ (post)')
                else:
                    mask = y_post_k.notna().values
                    if mask.any():
                        ax_jp.plot(x_post[mask], y_post_k[mask], '-', lw=2.0, marker=m_jp, markevery=me, ms=5, color=color_jp, label='Japan φ (post)')

    # Vertical reference with label
    ax.axvline(pd.Timestamp('2010-03-31'), linestyle='--', color='gray', alpha=0.7, label='Accounting rule change (2010Q1)')
    # Axes formatting
    ax.set_title('Off-balance share (φ): US vs Japan')
    ax.set_xlabel('Date'); ax.set_ylabel('φ (share)'); ax.set_ylim(0,1)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, alpha=0.3)
    # Build combined legend from both axes if right-axis is used
    handles, labels = ax.get_legend_handles_labels()
    if args.jp_right_axis:
        h2, l2 = ax_jp.get_legend_handles_labels()
        handles += h2; labels += l2
    if labels:
        # Deduplicate preserving first occurrence
        seen = {}
        ordered = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
                ordered.append((h, l))
        # Desired order: US pre → Rule change → US post → Japan*
        def pick(label_query):
            for i,(h,l) in enumerate(ordered):
                if l == label_query:
                    return ordered.pop(i)
            return None
        final = []
        for key in ['United States φ (pre)', 'Accounting rule change (2010Q1)', 'United States φ (post)']:
            item = pick(key)
            if item: final.append(item)
        # Any Japan-related labels next
        rest = [(h,l) for (h,l) in ordered if 'Japan' in l]
        others = [(h,l) for (h,l) in ordered if 'Japan' not in l]
        final.extend(rest)
        final.extend(others)
    ax.legend([h for h,_ in final], [l for _,l in final], loc='upper left', ncol=1, frameon=True)

    Path('figures').mkdir(exist_ok=True, parents=True)
    out_png = Path(args.out)
    if out_png.name and not out_png.name.startswith('fig_code05_'):
        out_png = out_png.with_name('fig_code05_' + out_png.name)
    fig.tight_layout(); fig.savefig(out_png, dpi=220)
    print(f'[OK] saved {out_png}')

if __name__ == '__main__':
    main()
