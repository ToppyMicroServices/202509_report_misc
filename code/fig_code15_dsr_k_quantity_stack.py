#!/usr/bin/env python3
"""fig_code15_dsr_k_quantity_stack

Stacked bar (or area) plot of DSR change decomposition contributions per quarter:
  quantity_contrib_pp, k_rate_contrib_pp, k_term_contrib_pp (and residual optional)
using outputs from proc_code08_build_dsr_k_quantity_decomposition.

By default produces two panels (US, JP) side by side.

Inputs (defaults):
  data_processed/proc_code08_US_DSR_k_quantity_decomposition.csv
  data_processed/proc_code08_JP_DSR_k_quantity_decomposition.csv (optional)

Outputs:
  figures/fig_code15_DSR_k_quantity_stack.png
  figures/fig_code15_DSR_k_quantity_stack_data.csv (long tidy for LaTeX)
"""
from __future__ import annotations
import argparse, pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from util_code01_lib_io import figure_name_with_code, safe_savefig, safe_to_csv

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data_processed'

DEFAULT_US = PROC / 'proc_code08_US_DSR_k_quantity_decomposition.csv'
DEFAULT_JP = PROC / 'proc_code08_JP_DSR_k_quantity_decomposition.csv'

COLORS = {
    'quantity_contrib_pp':'#0072b2',
    'k_rate_contrib_pp':'#e69f00',
    'k_term_contrib_pp':'#56b4e9',
    'residual_pp':'#999999'
}

def load_file(p: Path, country: str):
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=['Date'])
    needed = {'Date','quantity_contrib_pp','k_rate_contrib_pp','k_term_contrib_pp','delta_DSR_pp'}
    if not needed.issubset(df.columns):
        return None
    df['country'] = country
    return df.sort_values('Date')

def build_long(us_df, jp_df):
    frames = [d for d in [us_df,jp_df] if d is not None]
    if not frames:
        return None
    wide = pd.concat(frames, ignore_index=True)
    value_cols = ['quantity_contrib_pp','k_rate_contrib_pp','k_term_contrib_pp']
    # residual optional if not tiny
    if 'residual_pp' in wide.columns and wide['residual_pp'].abs().max() > 1e-3:
        value_cols.append('residual_pp')
    long = wide.melt(id_vars=['Date','country'], value_vars=value_cols, var_name='component', value_name='pp')
    return wide, long

def main():
    ap = argparse.ArgumentParser(description='DSR decomposition stacked contributions – fig_code15')
    ap.add_argument('--us', default=str(DEFAULT_US))
    ap.add_argument('--jp', default=str(DEFAULT_JP))
    ap.add_argument('--out', default='DSR_k_quantity_stack.png')
    ap.add_argument('--start-year', type=int)
    ap.add_argument('--end-year', type=int)
    ap.add_argument('--stack-alpha', type=float, default=0.9)
    ap.add_argument('--show-residual', action='store_true', help='Force display of residual even if small')
    args = ap.parse_args()
    us_df = load_file(Path(args.us), 'US')
    jp_df = load_file(Path(args.jp), 'JP')
    wide, long = build_long(us_df, jp_df)
    if wide is None:
        raise SystemExit('No decomposition inputs found.')
    if args.start_year:
        wide = wide[wide['Date'].dt.year >= args.start_year]
        long = long[long['Date'].dt.year >= args.start_year]
    if args.end_year:
        wide = wide[wide['Date'].dt.year <= args.end_year]
        long = long[long['Date'].dt.year <= args.end_year]
    countries = list(wide['country'].unique())
    fig, axes = plt.subplots(1, len(countries), figsize=(6.0*len(countries), 4.2), sharey=True)
    if len(countries) == 1:
        axes = [axes]
    for ax, c in zip(axes, countries):
        sub = wide[wide['country']==c]
        # Build stacked bars
        date_ord = sub['Date']
        comp_cols = ['quantity_contrib_pp','k_rate_contrib_pp','k_term_contrib_pp']
        if args.show_residual and 'residual_pp' in sub.columns:
            comp_cols.append('residual_pp')
        bottom = np.zeros(len(sub))
        for comp in comp_cols:
            vals = sub[comp].values
            ax.bar(date_ord, vals, bottom=bottom, width=70, color=COLORS.get(comp,'gray'), label=comp.replace('_pp',''), alpha=args.stack_alpha, edgecolor='black', linewidth=0.25)
            bottom += vals
        ax.axhline(0, color='k', linewidth=0.8, alpha=0.7)
        ax.set_title(f'{c}')
        ax.grid(True, axis='y', alpha=0.25)
        ax.set_xlabel('Date')
    axes[0].set_ylabel('Contribution to ΔDSR (pp)')
    # Legend consolidated
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), frameon=True)
    plt.tight_layout(rect=(0,0,1,0.93))
    out_png = figure_name_with_code(__file__, ROOT/'figures'/Path(args.out).name)
    (ROOT/'figures').mkdir(parents=True, exist_ok=True)
    safe_savefig(fig, out_png)
    print('saved:', out_png)
    # Export tidy
    tidy_csv = figure_name_with_code(__file__, ROOT/'figures'/(Path(args.out).stem + '_data.csv'))
    safe_to_csv(long, tidy_csv, index=False)
    print('saved:', tidy_csv)

if __name__ == '__main__':  # pragma: no cover
    main()
