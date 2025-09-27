"""fig_code13_dsr_creditgrowth_us_jp_timeseries

Purpose:
    Two-panel time series comparison of Household Debt Service Ratio (DSR, %) and
    Credit growth (pp YoY) for United States (top) and Japan (bottom). Both
    series are plotted on a unified y-scale (Percent / pp) to maximize visual
    comparability across countries. No regression annotation; this is a pure
    time-series "line view" counterpart to the scatter style figures.

Default inputs (quarter-end index, required columns: DSR_pct, CreditGrowth_ppYoY):
    data_processed/proc_code03_US_DSR_CreditGrowth_panel.csv
    data_processed/proc_code03_JP_DSR_CreditGrowth_panel.csv
Legacy fallback auto-detected if prefixed missing:
    data_processed/US_DSR_CreditGrowth_panel.csv
    data_processed/JP_DSR_CreditGrowth_panel.csv

Features:
    - Unified y-axis range across both panels (5% padding beyond global min/max)
    - Optional year filters (--start, --end)
    - Optional moving average smoothing (--ma WINDOW)
    - Latest point numeric annotation (can disable with --no-latest)
    - Uses figure_name_with_code + safe_savefig for provenance & no overwrite

Example:
    python code/fig_code13_dsr_creditgrowth_us_jp_timeseries.py \\
        --start 2000 --end 2024 \\
        --out figures/DSR_CreditGrowth_TimeSeries_US_JP.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from util_code01_lib_io import safe_savefig, ensure_unique, figure_name_with_code
from util_code02_colors import COLOR_US, COLOR_JP, COLOR_NEUTRAL

PREF_US = Path('data_processed/proc_code03_US_DSR_CreditGrowth_panel.csv')
PREF_JP = Path('data_processed/proc_code03_JP_DSR_CreditGrowth_panel.csv')
LEG_US = Path('data_processed/US_DSR_CreditGrowth_panel.csv')
LEG_JP = Path('data_processed/JP_DSR_CreditGrowth_panel.csv')
DEFAULT_US = PREF_US if PREF_US.exists() else (LEG_US if LEG_US.exists() else PREF_US)
DEFAULT_JP = PREF_JP if PREF_JP.exists() else (LEG_JP if LEG_JP.exists() else PREF_JP)
DEFAULT_OUT = Path('figures/DSR_CreditGrowth_TimeSeries_US_JP.png')

COL_DSR = 'DSR_pct'
COL_CG  = 'CreditGrowth_ppYoY'

# Consistent country colors
US_COLOR = COLOR_US
JP_COLOR = COLOR_JP
LINE_WIDTH = 1.4
MARKER_SIZE = 3.5

def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0)
    for c in (COL_DSR, COL_CG):
        if c not in df.columns:
            raise ValueError(f"column '{c}' missing in {path}")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df[COL_DSR] = pd.to_numeric(df[COL_DSR], errors='coerce')
    df[COL_CG]  = pd.to_numeric(df[COL_CG], errors='coerce')
    return df.dropna(subset=[COL_DSR, COL_CG])

def apply_filters(df: pd.DataFrame, start: int|None, end: int|None) -> pd.DataFrame:
    if start:
        df = df[df.index.year >= start]
    if end:
        df = df[df.index.year <= end]
    return df

def maybe_ma(df: pd.DataFrame, window: int|None) -> pd.DataFrame:
    if not window or window <= 1:
        return df
    return df.assign(
        **{COL_DSR: df[COL_DSR].rolling(window, min_periods=1).mean(),
           COL_CG:  df[COL_CG].rolling(window, min_periods=1).mean()}
    )

def plot_panels(us: pd.DataFrame, jp: pd.DataFrame, out: Path, title: str|None, annotate_latest: bool, show: bool=False):
    # Compute unified y-axis range across both countries
    all_vals = pd.concat([us[[COL_DSR, COL_CG]], jp[[COL_DSR, COL_CG]]]).values.ravel()
    all_vals = all_vals[np.isfinite(all_vals)]
    ymin, ymax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    span = ymax - ymin if ymax > ymin else 1.0
    pad = span * 0.05
    ymin -= pad
    ymax += pad

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    specs = [
        (us, 'United States', US_COLOR),
        (jp, 'Japan', JP_COLOR)
    ]
    for ax, (df, label, c_country) in zip(axes, specs):
        # DSR: dot + line (circle markers)
        ax.plot(df.index, df[COL_DSR], '-o', color=c_country, label=COL_DSR,
                lw=LINE_WIDTH, ms=MARKER_SIZE, alpha=0.95)
        # Credit growth: dot + line (triangle markers)
        ax.plot(df.index, df[COL_CG], '-^', color=COLOR_NEUTRAL, label=COL_CG,
                lw=LINE_WIDTH*0.9, ms=MARKER_SIZE, alpha=0.85)
        ax.set_title(f"{label}: DSR and Credit growth (ΔYoY, pp)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(ymin, ymax)
        if annotate_latest:
            last = df.iloc[-1]
            ax.text(df.index[-1], last[COL_DSR], f" {last[COL_DSR]:.1f}", color=c_country, va='center')
            ax.text(df.index[-1], last[COL_CG],  f" {last[COL_CG]:.1f}",  color=COLOR_NEUTRAL,  va='center')
        ax.legend(loc='upper right', frameon=True)
    axes[0].set_ylabel('Percent / pp')
    axes[1].set_ylabel('Percent / pp')
    axes[1].set_xlabel('Quarter')
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    out.parent.mkdir(parents=True, exist_ok=True)
    out = figure_name_with_code(__file__, out)
    final = ensure_unique(out)
    safe_savefig(fig, final)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return final

def main(argv=None):
    ap = argparse.ArgumentParser(description='US vs JP DSR & Credit growth time-series dual panels')
    ap.add_argument('--us-csv', default=str(DEFAULT_US))
    ap.add_argument('--jp-csv', default=str(DEFAULT_JP))
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--start', type=int, default=None)
    ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--ma', type=int, default=None, help='optional moving-average window (quarters)')
    ap.add_argument('--no-latest', action='store_true', help='disable latest value annotation')
    ap.add_argument('--title', default='United States & Japan: DSR and Credit growth (ΔYoY, pp)')
    args = ap.parse_args(argv)

    us = load_panel(Path(args.us_csv))
    jp = load_panel(Path(args.jp_csv))
    us = apply_filters(us, args.start, args.end)
    jp = apply_filters(jp, args.start, args.end)
    if us.empty or jp.empty:
        print('[ERROR] filtered data empty; adjust year range or check inputs')
        return 1
    us = maybe_ma(us, args.ma)
    jp = maybe_ma(jp, args.ma)
    show_flag = (len(sys.argv) == 1)
    final = plot_panels(us, jp, Path(args.out), args.title, not args.no_latest, show=show_flag)
    print(f'[OK] figure saved: {final}')
    return 0

if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
