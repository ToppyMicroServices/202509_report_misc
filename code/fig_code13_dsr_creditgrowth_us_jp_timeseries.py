#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fig_code13_dsr_creditgrowth_us_jp_timeseries.py

Two-panel time-series comparison of DSR and Credit Growth (ΔYoY, pp)
for the United States and Japan.

Publication-quality defaults:
  • Country color consistency: both DSR and Credit Growth share the same hue per country
      - United States: darkgoldenrod
      - Japan: dodgerblue
  • Line style encodes variable:
      - DSR: solid, thicker
      - Credit Growth: dashed + triangle markers, thinner
  • Shared legend on top
  • markevery=4 (downsampled markers)
  • PDF/EPS/PNG output (+ optional TIFF)
  • --mono flag for grayscale preview
"""

import argparse
import sys
import warnings
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------
FRL_SINGLE_COL_INCH = 3.30
PANEL_HEIGHT_INCH = 1.90
TOP_BOTTOM_PAD = 0.25
DPI_DEFAULT = 300
FONT_FAMILY_PREF = ["Arial", "Helvetica", "DejaVu Sans"]

DATA_DIR = Path("data_processed")
FIG_DIR = Path("figures")
FIG_BASENAME = "fig_code13_DSR_CreditGrowth_TimeSeries_US_JP"

# ---------------------------------------------------------------------
# Country color palette (Okabe–Ito safe)
# ---------------------------------------------------------------------
COLOR_US = "#b8860b"   # darkgoldenrod
COLOR_JP = "#1e90ff"   # dodgerblue

# ---------------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------------
def _configure_matplotlib(fontscale: float = 1.0):
    plt.rcParams.update({
        "path.simplify": True,
        "path.simplify_threshold": 0.8,
        "font.family": FONT_FAMILY_PREF,
        "axes.titlesize": 9 * fontscale,
        "axes.labelsize": 8.5 * fontscale,
        "xtick.labelsize": 8 * fontscale,
        "ytick.labelsize": 8 * fontscale,
        "legend.fontsize": 8 * fontscale,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": DPI_DEFAULT,
        "figure.dpi": DPI_DEFAULT,
    })

# ---------------------------------------------------------------------
# Data loader compatible with proc_code03
# ---------------------------------------------------------------------
def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)

    # detect date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif df.columns[0].lower().startswith("unnamed"):
        df["date"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    else:
        raise ValueError(f"No recognizable date column in {path.name}")

    # detect DSR column
    if "dsr" not in df.columns:
        if "DSR_pct" in df.columns:
            df["dsr"] = pd.to_numeric(df["DSR_pct"], errors="coerce")
        else:
            raise ValueError(f"'dsr' or 'DSR_pct' not found in {path.name}")

    # detect credit growth column
    cg_candidates = [
        "credit_growth_yoy_pp",
        "CreditGrowth_ppYoY",
        "CreditGrowthYoY_pct",
        "credit_growth_yoy_proxy_pct",
        "credit_growth_pct",
        "credit_yoy_pct",
        "credit_yoy_pp",
        "credit_growth_yoy_percentage_points",
    ]
    cg_col = next((c for c in cg_candidates if c in df.columns), None)
    if cg_col is None:
        raise ValueError(f"credit-growth column not found in {path.name}")
    df = df.rename(columns={cg_col: "credit_growth_yoy_pp"})

    out = df[["date", "dsr", "credit_growth_yoy_pp"]].copy()
    out = out.sort_values("date").reset_index(drop=True)
    return out

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _clip_period(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]
    return df

def _apply_ma(df: pd.DataFrame, window: int | None) -> pd.DataFrame:
    if not window or window <= 1:
        return df
    out = df.copy()
    out["credit_growth_yoy_pp"] = (
        out["credit_growth_yoy_pp"].rolling(window, min_periods=1).mean()
    )
    return out

def _build_figure_size(n_panels=2):
    width = FRL_SINGLE_COL_INCH
    height = n_panels * PANEL_HEIGHT_INCH + TOP_BOTTOM_PAD
    return (width, height)

def _format_axes(ax, title: str):
    ax.set_title(title, loc="left", weight="bold")
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.tick_params(axis="both", which="both", length=3)

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def _plot_panel(ax, df: pd.DataFrame, country: str, markevery: int = 4, mono: bool = False):
    # color assignment
    if mono:
        color = "black"
    else:
        color = COLOR_US if country == "US" else COLOR_JP

    # DSR (solid, thicker)
    ax.plot(
        df["date"], df["dsr"],
        color=color, linestyle="-", linewidth=1.8,
        label="DSR", solid_capstyle="round"
    )

    # Credit Growth (dashed, thinner, same color)
    ax.plot(
        df["date"], df["credit_growth_yoy_pp"],
        color=color, linestyle="--", linewidth=1.1,
        marker="^", markersize=3.0,
        markevery=markevery, dash_capstyle="round",
        label="Credit growth (ΔYoY, pp)"
    )

    ax.set_ylabel("Percent / percentage points (pp)")

# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
def _export(fig: plt.Figure, outbase: Path, export_tiff: bool = False, dpi: int = DPI_DEFAULT):
    out_png = outbase.with_suffix(".png")
    out_pdf = outbase.with_suffix(".pdf")
    out_eps = outbase.with_suffix(".eps")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_eps, dpi=dpi, bbox_inches="tight")
    if export_tiff:
        fig.savefig(outbase.with_suffix(".tiff"), dpi=max(300, dpi), bbox_inches="tight")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="FRL-ready color figure: DSR vs Credit Growth (US vs JP, same hue per country)"
    )
    parser.add_argument("--us-csv", default=str(DATA_DIR / "proc_code03_US_DSR_CreditGrowth_panel.csv"))
    parser.add_argument("--jp-csv", default=str(DATA_DIR / "proc_code03_JP_DSR_CreditGrowth_panel.csv"))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--cg-ma", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=DPI_DEFAULT)
    parser.add_argument("--export-tiff", action="store_true")
    parser.add_argument("--outfile", default=str(FIG_DIR / FIG_BASENAME))
    parser.add_argument("--mono", action="store_true", help="Preview in monochrome")
    args = parser.parse_args()

    _configure_matplotlib()
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)

    us = _load_csv(Path(args.us_csv))
    jp = _load_csv(Path(args.jp_csv))

    us = _clip_period(us, args.start, args.end)
    jp = _clip_period(jp, args.start, args.end)
    if len(us) == 0 or len(jp) == 0:
        raise ValueError("Empty dataset after applying start/end filters.")

    us = _apply_ma(us, args.cg_ma)
    jp = _apply_ma(jp, args.cg_ma)

    fig_size = _build_figure_size(n_panels=2)
    fig, axes = plt.subplots(2, 1, figsize=fig_size, sharex=True)

    _format_axes(axes[0], "(a) United States")
    _plot_panel(axes[0], us, country="US", markevery=4, mono=args.mono)

    _format_axes(axes[1], "(b) Japan")
    _plot_panel(axes[1], jp, country="JP", markevery=4, mono=args.mono)
    axes[1].set_xlabel("Year")

    # Shared legend on top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2,
        frameon=False, bbox_to_anchor=(0.5, 1.02)
    )

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    _export(fig, Path(args.outfile), export_tiff=args.export_tiff, dpi=args.dpi)
    plt.close(fig)

    print(f"Saved to: {Path(args.outfile).with_suffix('.png')}")
    print(f"Saved to: {Path(args.outfile).with_suffix('.pdf')}")
    print(f"Saved to: {Path(args.outfile).with_suffix('.eps')}")
    if args.export_tiff:
        print(f"Saved to: {Path(args.outfile).with_suffix('.tiff')}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        warnings.warn(str(e))
        sys.exit(1)
