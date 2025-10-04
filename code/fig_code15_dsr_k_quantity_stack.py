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
import matplotlib.dates as mdates
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

PRETTY = {
    'quantity_contrib_pp': 'Quantity',
    'k_rate_contrib_pp': 'Rate (Δk)',
    'k_term_contrib_pp': 'Term (Δk)',
    'residual_pp': 'Residual',
}

def load_file(p: Path, country: str):
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=['Date'])
    # di_source が無い古い出力も想定
    if 'di_source' not in df.columns:
        df['di_source'] = 'UNKNOWN'
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
    if 'residual_pp' in wide.columns and wide['residual_pp'].abs().max() > 1e-3:
        value_cols.append('residual_pp')
    id_vars = ['Date','country'] + (['di_source'] if 'di_source' in wide.columns else [])
    long = wide.melt(id_vars=id_vars, value_vars=value_cols, var_name='component', value_name='pp')
    return wide, long

def main():
    ap = argparse.ArgumentParser(description='DSR decomposition stacked contributions – fig_code15')
    ap.add_argument('--us', default=str(DEFAULT_US))
    ap.add_argument('--jp', default=str(DEFAULT_JP))
    ap.add_argument('--out', default='DSR_k_quantity_stack.png')
    ap.add_argument('--start-year', type=int)
    ap.add_argument('--end-year', type=int)
    ap.add_argument('--stack-alpha', type=float, default=0.9)
    ap.add_argument('--plot-type', type=str, choices=['line','bar','area'], default='line',
                    help='Plot style: line (default), bar (stacked bars), or area (stacked area)')
    ap.add_argument('--show-residual', action='store_true', help='Force display of residual even if small')
    ap.add_argument('--country-order', type=str, default=None, help='Comma-separated order, e.g., US,JP')
    ap.add_argument('--allow-us-dsr-over-k', action='store_true',
                    help='US の di_source が DSR_over_k でも続行（既定は停止）')
    args = ap.parse_args()

    us_df = load_file(Path(args.us), 'US')
    jp_df = load_file(Path(args.jp), 'JP')
    built = build_long(us_df, jp_df)
    if not built:
        raise SystemExit('No decomposition inputs found.')
    wide, long = built

    # 既定では US は独立DIが使われていることを要求
    if 'country' in wide.columns and 'di_source' in wide.columns:
        us_mask = (wide['country'] == 'US')
        if us_mask.any():
            sources = sorted(wide.loc[us_mask, 'di_source'].dropna().unique())
            if any(s != 'Z1_debt_over_DPI' for s in sources) and not args.allow_us_dsr_over_k:
                raise SystemExit(f"US di_source is not Z1_debt_over_DPI (found {sources}). "
                                 f"Rebuild proc_code08 with independent DI or pass --allow-us-dsr-over-k.")

    if args.start_year:
        wide = wide[wide['Date'].dt.year >= args.start_year]
        long = long[long['Date'].dt.year >= args.start_year]
    if args.end_year:
        wide = wide[wide['Date'].dt.year <= args.end_year]
        long = long[long['Date'].dt.year <= args.end_year]

    countries = list(wide['country'].unique())
    if args.country_order:
        try:
            order_list = [x.strip() for x in args.country_order.split(',')]
            filtered = [c for c in order_list if c in countries]
            if filtered:
                countries = filtered
        except Exception:
            pass

    fig, axes = plt.subplots(1, len(countries), figsize=(6.0*len(countries), 4.2), sharey=True)
    if len(countries) == 1:
        axes = [axes]

    # Global xlim across panels (after filtering)
    global_min = wide['Date'].min()
    global_max = wide['Date'].max()

    for ax, c in zip(axes, countries):
        sub = wide[wide['country']==c]
        date_ord = sub['Date']
        comp_cols = ['quantity_contrib_pp','k_rate_contrib_pp','k_term_contrib_pp']
        if args.show_residual and 'residual_pp' in sub.columns:
            comp_cols.append('residual_pp')
        # Grid behind
        ax.set_axisbelow(True)
        if args.plot_type == 'bar':
            bottom = np.zeros(len(sub))
            alpha_this = 1.0 if c == 'US' else args.stack_alpha
            for comp in comp_cols:
                vals = sub[comp].values
                face = COLORS.get(comp,'gray')
                # Quantityだけは縞に見える原因となる枠線を消す
                edge = 'none' if comp == 'quantity_contrib_pp' else 'black'
                lw = 0.0 if comp == 'quantity_contrib_pp' else 0.4
                ax.bar(
                    date_ord, vals,
                    bottom=bottom, width=70,
                    color=face, label=PRETTY.get(comp, comp),
                    alpha=alpha_this,
                    edgecolor=edge, linewidth=lw,
                    hatch=None,
                    antialiased=True,
                )
                bottom += vals
        elif args.plot_type == 'area':
            # Stacked area using fill_between between cumulative bottoms
            bottom = np.zeros(len(sub))
            for comp in comp_cols:
                vals = sub[comp].values
                top = bottom + vals
                ax.fill_between(date_ord, bottom, top,
                                color=COLORS.get(comp,'gray'), alpha=args.stack_alpha,
                                label=PRETTY.get(comp, comp), linewidth=0.0, antialiased=True)
                bottom = top
        else:  # line
            for comp in comp_cols:
                ax.plot(
                    date_ord, sub[comp].values,
                    color=COLORS.get(comp,'gray'), label=PRETTY.get(comp, comp),
                    linewidth=1.8, marker='o', markersize=3.0, alpha=0.95, zorder=3,
                )
        # Per-panel styling and common x-axis formatting
        ax.axhline(0, color='k', linewidth=0.8, alpha=0.7, zorder=2)
        ax.set_title(f'{c}', fontsize=12)
        ax.grid(True, axis='y', alpha=0.25, linestyle='-')
        # Apply common x-axis limits and 2-year ticks
        ax.set_xlim(global_min, global_max)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.set_xlabel('Year')
    axes[0].set_ylabel('Contribution to ΔDSR (pp)', fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='upper center', ncol=len(handles), frameon=True)
    leg.get_frame().set_alpha(0.95)
    leg.get_frame().set_edgecolor('black')
    plt.tight_layout(rect=(0,0,1,0.93))

    out_png = figure_name_with_code(__file__, ROOT/'figures'/Path(args.out).name)
    (ROOT/'figures').mkdir(parents=True, exist_ok=True)
    saved_png = safe_savefig(fig, out_png, overwrite=True)
    print('saved:', saved_png)

    tidy_csv = figure_name_with_code(__file__, ROOT/'data_processed'/(Path(args.out).stem + '_data.csv'))
    saved_csv = safe_to_csv(long, tidy_csv, index=False, overwrite=True)
    print('saved:', saved_csv)

if __name__ == '__main__':  # pragma: no cover
    main()