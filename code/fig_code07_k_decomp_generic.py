"""fig_code07_k_decomp_generic

Visualise the rate and term contributions to Delta DSR over time.

Highlights:
- Default run plots US and JP side-by-side (paper layout).
- Reads the proc_code08 outputs that already contain k_rate_contrib_pp and k_term_contrib_pp.
- Legacy single-CSV mode (`--csv`) is still available.
- Trimming, year filters, and future-date handling retain the previous defaults.

Required columns (per CSV):
    Date and at least one of the following:
        - k_rate_contrib_pp
        - k_term_contrib_pp
    (Legacy dk_rate/dk_term only files are no longer supported; rebuild via proc_code08 first.)

Default axes: end-year=2025, end-quarter=2 (i.e. through 2025Q2).
"""
import argparse, pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional
from util_code01_lib_io import figure_name_with_code, safe_savefig, safe_to_csv


COMPONENT_ORDER = ["k_rate_contrib_pp", "k_term_contrib_pp"]
COLORS = {
    "k_rate_contrib_pp": "#e69f00",
    "k_term_contrib_pp": "#56b4e9",
}
PRETTY = {
    "k_rate_contrib_pp": "Rate (Delta k)",
    "k_term_contrib_pp": "Term (Delta k)",
}
LINE_STYLE = {
    "k_rate_contrib_pp": {"marker": "o", "markersize": 4.5, "linewidth": 2.0},
    "k_term_contrib_pp": {"marker": "s", "markersize": 4.0, "linewidth": 2.0},
}


def _trim_trailing_zero_blocks(df: pd.DataFrame, contrib_cols, eps: float):
    """Return a trimmed copy of df where trailing rows with all-zero
    (within eps) contributions are removed. If all rows are zero, keep the first.
    """
    if not contrib_cols:
        return df
    mask_nonzero = (df[contrib_cols].abs() > eps).any(axis=1)
    if not mask_nonzero.any():  # all zero -> keep first row only
        return df.iloc[[0]].copy()
    last_idx = mask_nonzero[mask_nonzero].index[-1]
    return df.loc[:last_idx].copy()


def _load_and_prepare(csv_path: Path, args, country: Optional[str] = None) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    if "Date" not in df.columns:
        raise SystemExit(f"CSV missing required 'Date' column: {csv_path}")
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

    # Normalise known legacy column names so downstream plotting is uniform.
    if "US_dk_rate_per_quarter" in df.columns:
        if "dk_rate" not in df.columns:
            df = df.rename(columns={"US_dk_rate_per_quarter": "dk_rate"})
        else:
            df = df.drop(columns=["US_dk_rate_per_quarter"])

    # year-range parsing
    if args.year_range:
        try:
            y0, y1 = args.year_range.split(":")
            args.start_year = int(y0)
            args.end_year = int(y1)
        except Exception:
            raise SystemExit(f"Invalid --year-range (expected YYYY:YYYY): {args.year_range}")

    # end-year alias
    if args.max_year is not None:
        args.end_year = args.max_year

    # filter
    if args.start_year is not None:
        df = df[df["Date"].dt.year >= args.start_year]
    if args.end_year is not None:
        df = df[df["Date"].dt.year <= args.end_year]
        if args.end_quarter is not None:
            mask = (df["Date"].dt.year < args.end_year) | (
                (df["Date"].dt.year == args.end_year) & (df["Date"].dt.quarter <= args.end_quarter)
            )
            df = df[mask]

    if args.clip_future:
        today = pd.Timestamp.today().normalize()
        df = df[df["Date"] <= today]

    contrib_cols = [c for c in COMPONENT_ORDER if c in df.columns]
    if not contrib_cols:
        raise SystemExit(
            "No contribution columns found (expected k_rate_contrib_pp / k_term_contrib_pp). "
            "Run proc_code08 to build the decomposition with pp units or pass a CSV that includes them."
        )

    if args.trim:
        df = _trim_trailing_zero_blocks(df, contrib_cols, args.zero_eps)

    if args.last_n_years is not None and len(df):
        max_year = df["Date"].dt.year.max()
        cutoff_year = max_year - args.last_n_years + 1
        df = df[df["Date"].dt.year >= cutoff_year]

    if country is not None:
        df = df.copy()
        df["country"] = country

    return df, contrib_cols


def _build_long(entries: list[tuple[str, pd.DataFrame, list[str]]]) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    frames = []
    for country, df, contrib_cols in entries:
        if df.empty:
            continue
        if "country" not in df.columns:
            df = df.copy()
            df["country"] = country
        value_cols = [c for c in COMPONENT_ORDER if c in contrib_cols and c in df.columns]
        if not value_cols:
            continue
        keep_cols = ["Date", "country"] + value_cols
        frames.append(df[keep_cols].copy())
    if not frames:
        return None
    wide = pd.concat(frames, ignore_index=True)
    wide = wide.sort_values(["country", "Date"]).reset_index(drop=True)
    long = wide.melt(
        id_vars=["Date", "country"],
        value_vars=[c for c in COMPONENT_ORDER if c in wide.columns],
        var_name="component",
        value_name="pp",
    )
    long = long.sort_values(["country", "Date", "component"]).reset_index(drop=True)
    return wide, long


def main():
    ap = argparse.ArgumentParser()
    # Single-CSV compatibility
    ap.add_argument("--csv", required=False, help="Single processed CSV to plot (compat mode)")
    ap.add_argument("--out", required=False, help="Output PNG (default depends on mode)")
    ap.add_argument("--title", default="k decomposition")
    # Paper defaults
    ap.add_argument("--start-year", type=int, default=2008, help="Earliest calendar year to include (inclusive). Default 2008 for paper-ready view")
    ap.add_argument("--end-year", type=int, default=2025, help="Latest calendar year to include (inclusive) (default: 2025)")
    ap.add_argument("--end-quarter", type=int, default=2, choices=[1,2,3,4], help="Final quarter within end-year to include (default: 2 => up to end of Q2)")
    ap.add_argument("--max-year", type=int, default=None, help="Alias for --end-year; if provided overrides --end-year")
    ap.add_argument("--trim-trailing-zeros", dest="trim", action="store_true", default=True, help="Trim trailing all-zero blocks (default)")
    ap.add_argument("--no-trim-trailing-zeros", dest="trim", action="store_false", help="Disable trimming of trailing zero block")
    ap.add_argument("--clip-future", dest="clip_future", action="store_true", default=True, help="Drop rows with Date > today (default)")
    ap.add_argument("--no-clip-future", dest="clip_future", action="store_false", help="Allow future-dated rows to remain")
    ap.add_argument("--last-n-years", type=int, default=None, help="If set, keep only the last N calendar years (relative to final retained Date)")
    ap.add_argument("--zero-eps", type=float, default=0.0, help="Tolerance to treat small magnitudes as zero when trimming")
    ap.add_argument("--year-range", default=None, help="Alternative to start/end: format YYYY:YYYY (inclusive)")
    ap.add_argument("--debug-save-trimmed-csv", action="store_true", help="Also save the final trimmed dataframe(s) as CSV in data_processed/")
    # New: dual-panel defaults
    ap.add_argument("--us-csv", required=False, help="Processed US CSV (default: data_processed/proc_code01_k_decomposition_US_quarterly.csv)")
    ap.add_argument("--jp-csv", required=False, help="Processed JP CSV (default: data_processed/proc_code01_k_decomposition_JP_quarterly.csv)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    # Single CSV mode when --csv is provided
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise SystemExit(f"CSV not found: {csv_path}")
        df, contrib_cols = _load_and_prepare(csv_path, args)
        plt.figure(figsize=(11, 4.6))
        ax = plt.gca()
        ax.set_axisbelow(True)
        for comp in contrib_cols:
            style = LINE_STYLE.get(comp, {})
            ax.plot(
                df["Date"],
                df[comp],
                color=COLORS.get(comp, "#333333"),
                label=PRETTY.get(comp, comp),
                linewidth=style.get("linewidth", 1.8),
                marker=style.get("marker", "o"),
                markersize=style.get("markersize", 3.8),
                alpha=0.95,
            )
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.5)
        # Paper-style axes
        ax.set_ylabel("Contribution to Delta DSR (pp)")
        ax.set_title(args.title, color="black")
        ax.grid(True, axis="y", alpha=0.25)
        # 2-year ticks and shared range from this df
        if len(df):
            xmin, xmax = df["Date"].min(), df["Date"].max()
            ax.set_xlim(xmin, xmax)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.legend(loc="upper left", frameon=True)
        plt.tight_layout()
        out_default = root / "figures" / "K_DECOMP_generic.png"
        out_png = Path(args.out) if args.out else out_default
        out_path = figure_name_with_code(__file__, out_png)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        safe_savefig(plt.gcf(), out_path, overwrite=True)
        print("saved:", out_path)

        csv_label = "Single"
        upper_name = csv_path.stem.upper()
        if "US" in upper_name:
            csv_label = "US"
        elif "JP" in upper_name:
            csv_label = "JP"

        maybe_long = _build_long([(csv_label, df, contrib_cols)])
        if maybe_long:
            _, long = maybe_long
            data_dir = root / "data_processed"
            data_dir.mkdir(parents=True, exist_ok=True)
            tidy_csv = figure_name_with_code(
                __file__, data_dir / f"{out_path.stem}_data.csv"
            )
            saved_csv = safe_to_csv(long, tidy_csv, index=False, overwrite=True)
            print("saved:", saved_csv)

        if args.debug_save_trimmed_csv and len(df):
            base_dir = root / "data_processed"
            base_dir.mkdir(parents=True, exist_ok=True)
            csv_out = base_dir / f"{out_path.stem}_TRIMMED.csv"
            df.to_csv(csv_out, index=False)
            print("trimmed csv:", csv_out)
        return

    # Dual-panel default (paper-ready): US (left), JP (right)
    def _default_csv(country: str) -> Path:
        if country == "US":
            return root / "data_processed" / "proc_code08_US_DSR_k_quantity_decomposition.csv"
        else:
            return root / "data_processed" / "proc_code08_JP_DSR_k_quantity_decomposition.csv"

    us_csv = Path(args.us_csv) if args.us_csv else _default_csv("US")
    jp_csv = Path(args.jp_csv) if args.jp_csv else _default_csv("JP")
    available = []
    if us_csv.exists():
        available.append(("US", us_csv))
    else:
        print(f"[WARN] US CSV not found: {us_csv}")
    if jp_csv.exists():
        available.append(("JP", jp_csv))
    else:
        print(f"[WARN] JP CSV not found: {jp_csv}")
    if not available:
        raise SystemExit("No CSVs found for US/JP.")

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(6.2*n, 4.4), sharey=True)
    if n == 1:
        axes = [axes]

    # Pre-load to compute global x-range
    prepared = []
    for country, p in available:
        df, contrib_cols = _load_and_prepare(p, args, country=country)
        prepared.append((country, df, contrib_cols))

    # Global x limits across countries
    if prepared:
        global_min = min(df["Date"].min() for _, df, _ in prepared if len(df))
        global_max = max(df["Date"].max() for _, df, _ in prepared if len(df))

    handles_labels = None
    for ax, (country, df, contrib_cols) in zip(axes, prepared):
        ax.set_axisbelow(True)
        for comp in contrib_cols:
            style = LINE_STYLE.get(comp, {})
            ax.plot(
                df["Date"],
                df[comp],
                color=COLORS.get(comp, "#333333"),
                label=PRETTY.get(comp, comp),
                linewidth=style.get("linewidth", 1.8),
                marker=style.get("marker", "o"),
                markersize=style.get("markersize", 3.8),
                alpha=0.95,
            )
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.5)
        ax.set_title(country)
        ax.grid(True, axis="y", alpha=0.25)
        # Common x-limits and ticks (2-year)
        if prepared:
            ax.set_xlim(global_min, global_max)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.set_xlabel("Year")
        if handles_labels is None:
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                handles_labels = (handles, labels)

    axes[0].set_ylabel("Contribution to Delta DSR (pp)")
    if handles_labels is not None:
        legend = fig.legend(*handles_labels, loc="upper center", ncol=len(handles_labels[1]), frameon=True)
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor("black")
        plt.tight_layout(rect=(0,0,1,0.93))
    else:
        plt.tight_layout()

    out_default = root / "figures" / "K_DECOMP_US_JP_generic.png"
    out_png = Path(args.out) if args.out else out_default
    out_path = figure_name_with_code(__file__, out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe_savefig(fig, out_path, overwrite=True)
    print("saved:", out_path)

    maybe_long = _build_long(prepared)
    if maybe_long:
        _, long = maybe_long
        data_dir = root / "data_processed"
        data_dir.mkdir(parents=True, exist_ok=True)
        tidy_csv = figure_name_with_code(
            __file__, data_dir / f"{out_path.stem}_data.csv"
        )
        saved_csv = safe_to_csv(long, tidy_csv, index=False, overwrite=True)
        print("saved:", saved_csv)

    if args.debug_save_trimmed_csv:
        base_dir = root / "data_processed"
        base_dir.mkdir(parents=True, exist_ok=True)
        stem = out_path.stem
        for country, df, _ in prepared:
            csv_out = base_dir / f"{stem}_{country}_TRIMMED.csv"
            df.to_csv(csv_out, index=False)
            print("trimmed csv:", csv_out)


if __name__ == "__main__":
    main()
