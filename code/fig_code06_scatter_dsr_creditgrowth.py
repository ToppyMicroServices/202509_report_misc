#!/usr/bin/env python3
"""
fig_code06_scatter_dsr_creditgrowth.py

FRL-friendly scatter (credit growth vs DSR) + binned line with IQR band.
- Robust column detection (DSR & credit growth), with "--dsr-col" / "--credit-col" overrides.
- Auto percent handling: if values look like fractions (|median| <= 0.3), convert to percent.
- Single-column figure size, light y-grid, compact margins.
- Figures -> figures/, tidy CSV (binned) -> data_processed/.

Inputs (default):
  data_processed/proc_code03_US_DSR_CreditGrowth_panel.csv
Expected columns (any of the aliases below are accepted automatically):
  DSR: ['DSR_pct','DSR','dsr_pct','DebtServiceRatio_pct','DSR_percent','DSR_frac','dsr']
  Credit growth YoY: many variants (see CREDIT_CANDIDATES in code)

Outputs:
  figures/fig_code06_SCATTER_US_DSR_vs_CreditGrowth_FINAL.png
  figures/fig_code06_SCATTER_US_DSR_vs_CreditGrowth_FINAL_binned.png
  data_processed/fig_code06_SCATTER_US_DSR_vs_CreditGrowth_FINAL_binned_data.csv
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# ---------- IO utils (fallback if your project helpers are unavailable) ----------
try:
    from util_code01_lib_io import figure_name_with_code, safe_savefig, safe_to_csv
except Exception:
    def figure_name_with_code(_this_file: str, out_path: Path) -> Path:
        return Path(out_path)
    def safe_savefig(fig, path: Path, dpi: int = 600, overwrite: bool = True, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
    def safe_to_csv(df: pd.DataFrame, path: Path, overwrite: bool = True, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs.pop('overwrite', None)
        df.to_csv(path, **kwargs)

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data_processed"
FIGS = ROOT / "figures"

DEFAULT_PANEL = PROC / "proc_code03_US_DSR_CreditGrowth_panel.csv"

# ---------- Column aliases ----------
DSR_CANDIDATES = [
    "DSR_pct", "DSR", "dsr_pct", "DebtServiceRatio_pct",
    "DSR_percent", "DSR_Percent", "DSR_frac", "dsr"
]

CREDIT_CANDIDATES = [
    "credit_growth_yoy_pct", "CreditGrowthYoY_pct", "CreditYoY_pct",
    "YoY_credit_growth_pct", "credit_growth_yoy_proxy_pct",
    "credit_proxy_yoy_pct", "credit_growth_pct", "credit_yoy_pct",
    "Credit_YoY_pct", "CreditGrowthYoY", "CreditGrowth_pct",
    "Credit_YoY", "YoY_credit_growth", "credit_growth_yoy",
    "credit_growth", "CreditGrowthYoY_proxy", "Credit_Growth_YoY_pct",
    # Panel variants in this repo
    "CreditGrowth_ppYoY", "CreditGrowth_pp", "CreditGrowth_pp_yoy"
]

# ---------- Styling ----------
def _set_matplotlib_rc():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "-",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        # Embed fonts in vector outputs
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

# ---------- Helpers ----------
def _pick_col(df: pd.DataFrame, prefer: str | None, candidates: list[str], label: str) -> str:
    """Pick a column by explicit name (if provided) or from a list of aliases."""
    if prefer:
        if prefer in df.columns:
            return prefer
        raise SystemExit(f"{label} column '{prefer}' not found; available: {list(df.columns)}")
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"{label} column not found; tried {candidates}")

def _to_percent_if_fraction(s: pd.Series) -> pd.Series:
    """If median(|x|) <= 0.3, treat as fraction and convert to percent."""
    med = float(np.nanmedian(np.abs(s.values)))
    if med <= 0.3:
        return s * 100.0
    return s

def _load_panel(path: Path, dsr_col_arg: str | None, credit_col_arg: str | None,
                start_year: int | None, end_year: int | None) -> pd.DataFrame:
    # Accept CSV indexed by date or with a Date column
    df = pd.read_csv(path)
    if df.columns[0].lower() in ("date", "time", "period"):
        df["__Date"] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index("__Date")
    elif "Date" in df.columns:
        df["__Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("__Date")
    else:
        # assume first column is index
        df = pd.read_csv(path, parse_dates=True, index_col=0)

    # Normalize to quarter-end
    df.index = pd.to_datetime(df.index).to_period("Q").to_timestamp("Q")
    df = df.sort_index()
    # Pick columns
    dsr_col = _pick_col(df, dsr_col_arg, DSR_CANDIDATES, "DSR")
    credit_col = _pick_col(df, credit_col_arg, CREDIT_CANDIDATES, "credit-growth")
    out = df[[dsr_col, credit_col]].copy()
    out.columns = ["DSR_raw", "CreditYoY_raw"]

    # Date filtering
    if start_year:
        out = out[out.index.year >= start_year]
    if end_year:
        out = out[out.index.year <= end_year]

    # Drop missing rows and convert to percent if necessary
    out = out.dropna()
    out["DSR_pct"] = _to_percent_if_fraction(out["DSR_raw"])
    out["CreditYoY_pct"] = _to_percent_if_fraction(out["CreditYoY_raw"])
    out = out[["DSR_pct", "CreditYoY_pct"]]
    print(f"[INFO] DSR source='{dsr_col}', Credit source='{credit_col}', rows={len(out)}")
    return out

# ---------- Plotters ----------
def _nice_percent_ticks(vmin: float, vmax: float, target_ticks: int = 5):
    # choose step from preferred list to get 3–6 ticks
    span = max(vmax - vmin, 1e-6)
    candidates = [10.0, 5.0, 2.0, 1.0, 0.5]
    best = candidates[-1]
    for step in candidates:
        cnt = span / step
        if 3 <= cnt <= 6:
            best = step
            break
    # compute start at a multiple of step
    start = np.floor(vmin / best) * best
    ticks = np.arange(start, vmax + 0.5 * best, best)
    # trim to range with small padding
    ticks = ticks[(ticks >= vmin - 1e-9) & (ticks <= vmax + 1e-9)]
    return ticks, best

def _format_axes_percent(ax):
    # determine ticks based on current limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xticks, xstep = _nice_percent_ticks(xmin, xmax)
    yticks, ystep = _nice_percent_ticks(ymin, ymax)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # decimals: if step >= 5 -> 0 decimals, else 1
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f%%' if xstep >= 5 else '%.1f%%'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%' if ystep >= 5 else '%.1f%%'))

def make_scatter(
    df: pd.DataFrame,
    out_png: Path,
    dpi: int = 600,
    show_hexbin: bool = False,
    title: str | None = None,
    overwrite: bool = True,
    save_pdf: bool = False,
    save_eps: bool = False,
    save_tiff: bool = False,
):
    """Single-column scatter with minimal chrome; legend distinguishes pre/post-2010."""
    # ~90 mm width => 3.54 inches
    fig, ax = plt.subplots(figsize=(3.54, 2.8), constrained_layout=True)
    x = df["DSR_pct"].to_numpy()
    y = df["CreditYoY_pct"].to_numpy()

    if show_hexbin:
        # Density background (kept subtle; FRL often prefers simpler figures)
        ax.hexbin(x, y, gridsize=20, cmap="cividis", mincnt=1, linewidths=0, alpha=0.6)

    mask_pre = df.index < pd.Timestamp("2010-01-01")
    # marker size ~ 4–6 px equivalent: use small pt^2; alpha 0.6–0.8
    ax.scatter(x[mask_pre], y[mask_pre], s=20, facecolor="none",
               edgecolor="#D97706", linewidth=0.9, alpha=0.75, label="pre-2010")
    ax.scatter(x[~mask_pre], y[~mask_pre], s=18, facecolor="#123A5B",
               edgecolor="none", alpha=0.70, label="post-2010")

    # Minimal in-figure footnote: n and clipping note (main notes should go in caption)
    n = len(df)

    ax.set_xlabel("Debt Service Ratio (%, PNFS)")
    ax.set_ylabel("Credit growth YoY (%)")

    # Trim to 1–99th pct to avoid huge axes due to a few outliers
    x_lo, x_hi = np.nanquantile(x, [0.01, 0.99])
    y_lo, y_hi = np.nanquantile(y, [0.01, 0.99])
    ax.set_xlim(x_lo - 0.02*(x_hi-x_lo), x_hi + 0.02*(x_hi-x_lo))
    ax.set_ylim(y_lo - 0.05*(y_hi-y_lo), y_hi + 0.05*(y_hi-y_lo))

    # set limits from 1–99th pct then format percent ticks & labels
    _format_axes_percent(ax)
    ax.grid(True, axis="y")
    leg = ax.legend(loc="lower right", frameon=True, fontsize=8)
    leg.get_frame().set_edgecolor("0.7")
    leg.get_frame().set_alpha(0.95)

    # small footnote
    ax.text(0.01, 0.02, f"n={n}; clipped 1–99p", transform=ax.transAxes,
            fontsize=7, color="#444", ha="left", va="bottom",
            bbox=dict(fc="white", ec="0.85", alpha=0.9, lw=0.5, pad=2.5))

    if title:
        ax.set_title(title, fontsize=9)
    save_path = figure_name_with_code(__file__, out_png)
    safe_savefig(fig, save_path, dpi=dpi, overwrite=overwrite)
    # Default: also save embedded-font PDF
    fig.savefig(str(save_path.with_suffix('.pdf')), bbox_inches="tight")
    if save_eps:
        fig.savefig(str(save_path.with_suffix('.eps')), bbox_inches="tight", format='eps')
    if save_tiff:
        fig.savefig(str(save_path.with_suffix('.tiff')), bbox_inches="tight", dpi=max(dpi, 600), format='tiff')
    print("saved:", save_path)

def make_binned(
    df: pd.DataFrame,
    out_png: Path,
    tidy_out: Path,
    bin_width: float = 0.25,
    dpi: int = 600,
    bin_mode: str = "quantile",
    n_bins: int = 12,
    n_boot: int = 400,
    show_mean: bool = False,
    title: str | None = None,
    overwrite: bool = True,
):
    """Modern binned relation: quantile bins + 95% CI (bootstrap) + median line.

    - bin_mode: 'quantile' (default) uses equal-count bins; 'width' uses fixed bin_width.
    - n_boot: number of bootstrap resamples per bin for 95% CI of the mean.
    - show_mean: if True, overlay mean line (thin, dashed) in addition to the median.
    """
    s_x = df["DSR_pct"].astype(float)
    s_y = df["CreditYoY_pct"].astype(float)

    if bin_mode == "quantile":
        cats = pd.qcut(s_x, q=n_bins, duplicates='drop')
        mids = np.array([iv.mid for iv in cats.cat.categories]) if hasattr(cats, 'cat') else None
        g = df.groupby(cats, observed=True)
    else:
        x = s_x.to_numpy()
        x_min = float(np.floor(np.nanmin(x) / bin_width) * bin_width)
        x_max = float(np.ceil(np.nanmax(x)  / bin_width) * bin_width)
        edges = np.arange(x_min, x_max + bin_width*0.5, bin_width)
        bins = pd.cut(s_x, edges, include_lowest=True)
        mids = np.array([iv.mid for iv in bins.cat.categories]) if hasattr(bins, 'cat') else None
        g = df.groupby(bins, observed=True)

    # Aggregate
    agg = g.agg(DSR_mid=("DSR_pct", "mean"),
                mean_yoy=("CreditYoY_pct", "mean"),
                median_yoy=("CreditYoY_pct", "median"),
                q25=("CreditYoY_pct", lambda s: s.quantile(0.25)),
                q75=("CreditYoY_pct", lambda s: s.quantile(0.75)),
                n=("CreditYoY_pct", "count"))
    # No dropna before CI assignment to keep lengths aligned

    # Bootstrap 95% CI for mean (within each bin)
    ci_lo = []
    ci_hi = []
    rng = np.random.default_rng(42)
    for _, grp in g:
        vals = grp["CreditYoY_pct"].dropna().to_numpy()
        if len(vals) >= 3:
            draws = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
            ci_lo.append(np.percentile(draws, 2.5))
            ci_hi.append(np.percentile(draws, 97.5))
        else:
            ci_lo.append(np.nan)
            ci_hi.append(np.nan)
    # Assign CI arrays; lengths align with observed=True and no dropna above
    agg["ci_lo"], agg["ci_hi"] = ci_lo, ci_hi

    # Now sort and reset index
    agg = agg.sort_values("DSR_mid").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(3.54, 2.8), constrained_layout=True)
    # Prefer 95% CI shading; fallback to IQR if CI is NaN
    lo = np.where(np.isnan(agg["ci_lo"]), agg["q25"].to_numpy(), agg["ci_lo"].to_numpy())
    hi = np.where(np.isnan(agg["ci_hi"]), agg["q75"].to_numpy(), agg["ci_hi"].to_numpy())
    ax.fill_between(agg["DSR_mid"], lo, hi, color="#9DBAD6", alpha=0.28, lw=0)
    # Median line as primary signal
    ax.plot(agg["DSR_mid"], agg["median_yoy"], lw=1.9, color="#1F4E79", marker="o", ms=3.5)
    if show_mean:
        ax.plot(agg["DSR_mid"], agg["mean_yoy"], lw=1.0, ls="--", color="#3E5C76", alpha=0.7)

    ax.set_xlabel("DSR (%, bin mid)")
    ax.set_ylabel("Credit growth YoY (%)")
    _format_axes_percent(ax)
    ax.grid(True, axis="y")
    if title:
        ax.set_title(title, fontsize=9)

    save_path = figure_name_with_code(__file__, out_png)
    safe_savefig(fig, save_path, dpi=dpi, overwrite=overwrite)
    print("saved:", save_path)

    tidy = agg.rename(columns={"median_yoy":"median", "mean_yoy":"mean"})
    tidy_path = figure_name_with_code(__file__, tidy_out)
    safe_to_csv(tidy, tidy_path, index=False, overwrite=True)
    print("saved:", tidy_path)

# ---------- Main ----------
def main():
    _set_matplotlib_rc()
    ap = argparse.ArgumentParser(description="FRL-friendly scatter of Credit growth vs DSR (plus binned line)")
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--dsr-col", default=None, help="explicit DSR column name (optional)")
    ap.add_argument("--credit-col", default=None, help="explicit credit growth column name (optional)")
    ap.add_argument("--start-year", type=int, default=None)
    ap.add_argument("--end-year", type=int, default=None)
    ap.add_argument("--bin-width", type=float, default=0.25)
    ap.add_argument("--bin-mode", choices=["quantile","width"], default="quantile")
    ap.add_argument("--n-bins", type=int, default=12)
    ap.add_argument("--boots", type=int, default=400, help="bootstrap reps per bin for 95% CI")
    ap.add_argument("--show-mean", action="store_true", help="overlay mean line (dashed)")
    ap.add_argument("--hexbin", action="store_true", help="overlay density background (off by default)")
    ap.add_argument("--dpi", type=int, default=600)
    ap.set_defaults(overwrite=True)
    ap.add_argument("--overwrite", action="store_true", dest="overwrite",
                    help="overwrite outputs (default: on)")
    ap.add_argument("--keep-versions", action="store_false", dest="overwrite",
                    help="append numeric suffix instead of overwriting")
    ap.add_argument("--save-eps", action="store_true")
    ap.add_argument("--save-tiff", action="store_true")
    ap.add_argument("--title", default=None, help="small title (optional; default: no title)")
    # Outputs (figures -> figures/, tidy CSV -> data_processed/)
    ap.add_argument("--scatter-out", default="fig_code06_SCATTER_US_DSR_vs_CreditGrowth_FINAL.png")
    ap.add_argument("--binned-out",  default="fig_code06_SCATTER_US_DSR_vs_CreditGrowth_FINAL_binned.png")
    ap.add_argument("--binned-csv",  default="fig_code06_SCATTER_US_DSR_vs_CreditGrowth_FINAL_binned_data.csv")

    args = ap.parse_args()
    panel_path = Path(args.panel)

    df = _load_panel(panel_path, args.dsr_col, args.credit_col, args.start_year, args.end_year)

    make_scatter(
        df=df,
        out_png=FIGS / args.scatter_out,
        dpi=args.dpi,
        show_hexbin=args.hexbin,
        title=args.title,
        overwrite=args.overwrite,
        save_eps=args.save_eps,
        save_tiff=args.save_tiff,
    )

    make_binned(
        df=df,
        out_png=FIGS / args.binned_out,
        tidy_out=PROC / args.binned_csv,
        bin_width=args.bin_width,
        dpi=args.dpi,
        bin_mode=args.bin_mode,
        n_bins=args.n_bins,
        n_boot=args.boots,
        show_mean=args.show_mean,
        title=args.title,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    main()
