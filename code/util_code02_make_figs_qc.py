"""util_code02_make_figs_qc

Lightweight quality‑check (QC) plotting helpers extracted from the former
``make_figs_qc`` scratch module. These are intentionally minimal so that
figure scripts can import or replicate patterns without a heavy dependency.

Primary entry point: ``quick_line`` for a fast time‑series sanity plot.
Intended usage during data preparation (before polished paper figures).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from util_code01_lib_io import safe_savefig

def quick_line(csv_path: str, value_col: str|None=None, out_png: str|None=None):
    """Render a simple line plot from a CSV.

    Parameters
    ----------
    csv_path : str
        Path to a CSV. Optional DATE column (any case) parsed to index.
    value_col : str | None
        Column to plot. If None, the first non-DATE column is auto-selected.
    out_png : str | None
        If provided, save the figure (overwriting existing file) instead of
        leaving it open. Returns the Matplotlib Figure object for optional
        further tweaks.
    """
    df = pd.read_csv(csv_path)
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.set_index('DATE')
    if value_col is None:
        candidates = [c for c in df.columns if c.upper() != 'DATE']
        if not candidates:
            raise ValueError('No value column')
        value_col = candidates[0]
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(df.index, df[value_col], label=value_col)
    ax.set_title(Path(csv_path).stem)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if out_png:
        safe_savefig(fig, out_png, overwrite=True)
    return fig

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--col', default=None)
    ap.add_argument('--out-png', default=None)
    args = ap.parse_args(argv)
    quick_line(args.csv, args.col, args.out_png)
    return 0

if __name__ == '__main__':  # pragma: no cover
    import sys
    sys.exit(main())
