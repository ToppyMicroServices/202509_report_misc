"""fig_code07_k_decomp_generic

Generic plot of k decomposition contributions over time with optional
date filtering and trimming of trailing all-zero periods.

Expected columns in --csv:
  Date (parseable as datetime) and one or more of:
    - dk_rate                 (rate contribution)
    - dk_term                 (term contribution)
    - US_dk_rate_per_quarter  (alternate rate contribution label)

Default behavior improvements vs. earlier version:
  1. Sorts by Date and drops duplicate Date rows (keeps first) to avoid jumps.
  2. By default trims any trailing block of rows where every plotted
     contribution column is exactly zero (or within tolerance).
  3. Optional start/end year clipping (inclusive) for reproducible figure ranges.
  4. Skips plotting missing series gracefully (only legend entries for series present).

CLI flags:
  --start-year YEAR              (optional) earliest year to keep
  --end-year YEAR                (optional) latest year to keep
  --trim-trailing-zeros / --no-trim-trailing-zeros (default: trim)
  --zero-eps FLOAT               tolerance for defining "zero" (default 0)

Output file name is passed via --out but automatically prefixed with
fig_code07_ using figure_name_with_code() for consistency.

Default axis cap: end-year defaults to 2025 and end-quarter defaults to 2
(i.e. show data up to 2025-Q2). Adjust with --end-year / --end-quarter.

When --debug-save-trimmed-csv is specified the trimmed dataset is now
saved into data_processed/ (not figures/) with name
fig_code07_<basename>_TRIMMED.csv for easier tracking under processed data.
Zero-arg run now: reads default CSV and writes default PNG.
"""
import argparse, pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from util_code01_lib_io import safe_savefig, figure_name_with_code


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=False, help="Processed CSV (default: data_processed/k_decomposition_JP_quarterly.csv; prefixed variant: data_processed/proc_code01_k_decomposition_JP_quarterly.csv)")
    ap.add_argument("--out", required=False, help="Output PNG (default: figures/K_DECOMP_JP_generic.png)")
    ap.add_argument("--title", default="k decomposition")
    ap.add_argument("--start-year", type=int, default=None, help="Earliest calendar year to include (inclusive)")
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
    ap.add_argument("--debug-save-trimmed-csv", action="store_true", help="Also save the final trimmed dataframe as CSV next to figure")
    args = ap.parse_args()

    # Resolve defaults if not provided
    root = Path(__file__).resolve().parents[1]  # .../release
    # Prefer new prefixed processed filename; fallback to legacy if absent
    preferred_csv = root / "data_processed" / "proc_code01_k_decomposition_JP_quarterly.csv"
    legacy_csv = root / "data_processed" / "k_decomposition_JP_quarterly.csv"
    default_csv = preferred_csv if preferred_csv.exists() else legacy_csv
    default_png = root / "figures" / "K_DECOMP_JP_generic.png"
    csv_path = Path(args.csv) if args.csv else default_csv
    out_png = Path(args.out) if args.out else default_png
    if not csv_path.exists():
        # If user pointed to preferred but only legacy exists (or vice versa)
        if csv_path == preferred_csv and legacy_csv.exists():
            print(f"[INFO] fallback to legacy k decomposition CSV: {legacy_csv.name}")
            csv_path = legacy_csv
        elif csv_path == legacy_csv and preferred_csv.exists():
            print(f"[INFO] using newly prefixed k decomposition CSV: {preferred_csv.name}")
            csv_path = preferred_csv
        if not csv_path.exists():
            raise SystemExit(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["Date"])  # Expect Date col
    if "Date" not in df.columns:
        raise SystemExit("CSV missing required 'Date' column")
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="first").reset_index(drop=True)

    # Parse combined year-range if supplied (overrides individual args)
    if args.year_range:
        try:
            y0, y1 = args.year_range.split(":")
            args.start_year = int(y0)
            args.end_year = int(y1)
        except Exception as e:  # pragma: no cover
            raise SystemExit(f"Invalid --year-range (expected YYYY:YYYY): {args.year_range}")

    # Alias handling: --max-year overrides --end-year if supplied
    if args.max_year is not None:
        args.end_year = args.max_year

    # Filter by explicit year range if requested
    if args.start_year is not None:
        df = df[df["Date"].dt.year >= args.start_year]
    if args.end_year is not None:
        before = len(df)
        df = df[df["Date"].dt.year <= args.end_year]
        if len(df) < before:
            print(f"Applied end-year <= {args.end_year}: removed {before - len(df)} rows.")
        # Apply quarter cut only for rows in the end-year
        if args.end_quarter is not None:
            before_q = len(df)
            # Keep rows where (year < end_year) OR (year == end_year AND quarter <= end_quarter)
            mask = (df["Date"].dt.year < args.end_year) | (
                (df["Date"].dt.year == args.end_year) & (df["Date"].dt.quarter <= args.end_quarter)
            )
            df = df[mask]
            if len(df) < before_q:
                print(f"Applied end-quarter Q{args.end_quarter} in {args.end_year}: removed {before_q - len(df)} rows beyond that quarter.")

    # Clip future dates (greater than 'today') unless disabled
    if args.clip_future:
        today = pd.Timestamp.today().normalize()
        before = len(df)
        df = df[df["Date"] <= today]
        if len(df) < before:
            print(f"Removed {before - len(df)} future rows (Date > {today.date()}).")

    contrib_cols = [c for c in ["dk_rate", "dk_term", "US_dk_rate_per_quarter"] if c in df.columns]
    if not contrib_cols:
        raise SystemExit("No contribution columns found (expected one of dk_rate, dk_term, US_dk_rate_per_quarter)")

    original_len = len(df)
    if args.trim:
        df = _trim_trailing_zero_blocks(df, contrib_cols, args.zero_eps)
    trimmed_len = len(df)

    if trimmed_len < original_len:
        print(f"Trimmed trailing zero block: removed {original_len - trimmed_len} rows (from {original_len} to {trimmed_len}).")

    # Apply last N years window after trimming (and future clip) for stable axis
    if args.last_n_years is not None and len(df):
        max_year = df["Date"].dt.year.max()
        cutoff_year = max_year - args.last_n_years + 1
        before = len(df)
        df = df[df["Date"].dt.year >= cutoff_year]
        if len(df) < before:
            print(f"Applied last {args.last_n_years} years window: {before - len(df)} rows dropped (cutoff_year={cutoff_year}).")

    plt.figure(figsize=(11, 4.6))
    ax = plt.gca()

    # Plot available contributions with consistent styling
    if "dk_rate" in contrib_cols:
        ax.plot(df["Date"], df["dk_rate"], marker="o", ms=4.5, lw=2.0, color="#e69f00", label="Rate contribution (dk_rate)")
    if "dk_term" in contrib_cols:
        ax.plot(df["Date"], df["dk_term"], marker="s", ms=4.0, lw=2.0, color="#56b4e9", label="Term contribution (dk_term)")
    if "US_dk_rate_per_quarter" in contrib_cols:
        ax.plot(df["Date"], df["US_dk_rate_per_quarter"], marker="o", ms=4.5, lw=2.0, color="#e69f00", label="Rate contribution (dk_rate)")

    ax.axhline(0.0, color="k", lw=1.0, alpha=0.5)
    ax.set_ylabel("Δk per quarter")
    ax.set_title(args.title, color="black")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    # Diagnostics: final range
    if len(df):
        print(f"Final date range: {df['Date'].min().date()} -> {df['Date'].max().date()} (rows={len(df)})")
    else:
        print("No data rows after filtering – nothing plotted.")

    out_path = figure_name_with_code(__file__, out_png)
    safe_savefig(plt.gcf(), out_path)
    print("saved:", out_path)

    if args.debug_save_trimmed_csv and len(df):
        # Always place trimmed CSV into data_processed directory
        base_dir = root / "data_processed"
        base_dir.mkdir(parents=True, exist_ok=True)
        csv_out = base_dir / f"{out_path.stem}_TRIMMED.csv"
        df.to_csv(csv_out, index=False)
        print("trimmed csv:", csv_out)


if __name__ == "__main__":
    main()
