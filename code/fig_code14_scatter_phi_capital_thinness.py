#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Credit growth (t+1) vs off-balance share φ, split by CET1/RWA terciles.

Design goals
------------
- Fixed tercile order and fixed colors (Okabe–Ito friendly):
    Low  -> orange, Mid -> green, High -> blue
- Group-specific smoothing (LOWESS) only when there are enough points.
  Otherwise, show scatter only (no forced fit).
- Scatter and line use the same color within each tercile.
- Clean layout for journal figures; export PNG/PDF/EPS (+ optional TIFF).

Notes / Findings (2025-10-05)
-----------------------------
- Empirically, only the Mid CET1 tercile shows a smooth positive φ–credit growth relation.
- Low/High groups are essentially flat due to sparse x-variation.
- JP_only subset: φ range is extremely narrow and the fit is unstable; therefore we
  should NOT draw any fitted line (neither LOWESS nor OLS). The figure will show
  scatter points only (no horizontal ticks).
"""

from pathlib import Path
import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import LOWESS; fall back gracefully if unavailable
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_LOWESS = True
except Exception:
    HAS_LOWESS = False

# -------------------------
# Figure defaults (FRL-ish)
# -------------------------
WIDTH_IN  = 3.30          # single-column width
HEIGHT_IN = 2.40
DPI       = 300
FONT_FAM  = ["Arial", "Helvetica", "DejaVu Sans"]

# -------------------------
# Tercile order and colors
# -------------------------
TERCILE_CANON = [
    "Low CET1 tercile (thin)",
    "Mid CET1 tercile",
    "High CET1 tercile (thick)",
]
PALETTE = {
    "Low CET1 tercile (thin)":   "#E69F00",  # orange
    "Mid CET1 tercile":          "#009E73",  # green
    "High CET1 tercile (thick)": "#7570B3",  # purple (changed from blue)
}

# Aliases to normalize user-provided tercile labels
TERCILE_ALIASES = {
    "low":  "Low CET1 tercile (thin)",
    "mid":  "Mid CET1 tercile",
    "high": "High CET1 tercile (thick)",
    "1":    "Low CET1 tercile (thin)",
    "2":    "Mid CET1 tercile",
    "3":    "High CET1 tercile (thick)",
}


def _configure_matplotlib():
    plt.rcParams.update({
        "path.simplify": True,
        "path.simplify_threshold": 0.8,
        "font.family": FONT_FAM,
        # FRL-compact text sizes
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        # Axes style
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": DPI,
        "figure.dpi": DPI,
    })


def _detect_column(df: pd.DataFrame, candidates: list[str], required_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Required column not found for '{required_name}'. "
        f"Tried candidates: {candidates}"
    )


def _normalize_tercile_labels(series: pd.Series) -> pd.Series:
    out = []
    for v in series.astype(str):
        key = v.strip().lower()
        if key in TERCILE_ALIASES:
            out.append(TERCILE_ALIASES[key])
            continue
        # try partial matches
        if "low" in key:
            out.append(TERCILE_ALIASES["low"])
        elif "mid" in key or "med" in key:
            out.append(TERCILE_ALIASES["mid"])
        elif "high" in key:
            out.append(TERCILE_ALIASES["high"])
        elif key in ("low",):
            out.append(TERCILE_ALIASES["low"])
        elif key in ("mid", "medium"):
            out.append(TERCILE_ALIASES["mid"])
        elif key in ("high",):
            out.append(TERCILE_ALIASES["high"])
        elif key in ("1.0", "1"):
            out.append(TERCILE_ALIASES["1"])
        elif key in ("2.0", "2"):
            out.append(TERCILE_ALIASES["2"])
        elif key in ("3.0", "3"):
            out.append(TERCILE_ALIASES["3"])
        else:
            # if unknown, keep original (will be dropped if not canonicalized)
            out.append(v)
    s = pd.Series(out)
    # keep only canonical labels; others will be marked NaN
    s = s.where(s.isin(TERCILE_CANON), other=np.nan)
    return s


def _load_and_prepare(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    phi_col = _detect_column(
        df,
        ["phi", "phi_share", "off_balance_share", "phi_current"],
        "phi"
    )
    if "credit_growth_ppYoY_lead" not in df.columns:
        raise ValueError(
            "Required column 'credit_growth_ppYoY_lead' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )
    y_col = "credit_growth_ppYoY_lead"
    tercile_col = _detect_column(
        df,
        ["cet1_tercile_label", "cet1_tercile", "tercile", "cet1_bucket", "Thin_tercile", "CET1_tercile"],
        "cet1_tercile_label"
    )

    # Parse/clean columns
    df = df.copy()
    df["phi"] = pd.to_numeric(df[phi_col], errors="coerce")
    df["cg_next_pp"] = pd.to_numeric(df[y_col], errors="coerce")

    # Create canonical tercile labels early (needed for group-wise normalization)
    df["tercile"] = _normalize_tercile_labels(df[tercile_col])

    # --- Group-wise unit normalization (handles mixed pp/bps across groups) ---
    # If a `country` column exists, normalize per (country, tercile); else per tercile only.
    grp_keys = []
    if "country" in df.columns:
        grp_keys = ["country", "tercile"]
    elif "tercile" in df.columns:
        grp_keys = ["tercile"]

    if grp_keys:
        for _, idx in df.groupby(grp_keys).groups.items():
            vals = pd.to_numeric(df.loc[idx, "cg_next_pp"], errors="coerce")
            p99_g = vals.abs().quantile(0.99)
            # If this group's scale looks like bps (e.g., p99≈300), convert just this group to pp
            if pd.notna(p99_g) and (p99_g > 50) and (p99_g < 10000):
                df.loc[idx, "cg_next_pp"] = vals / 100.0

    # Tercile-wise summary after normalization (helps verify the fix)
    try:
        summ = df.groupby("tercile")["cg_next_pp"].agg(["count", "min", "median", "max"]).round(2)
        print("[fig_code14] post-normalization by tercile:\n", summ)
    except Exception:
        pass

    # --- Final global safety check: if p99 still looks like bps, divide by 100 ---
    try:
        p99_all = df["cg_next_pp"].abs().quantile(0.99)
        if pd.notna(p99_all) and (p99_all > 50) and (p99_all < 10000):
            df["cg_next_pp"] = df["cg_next_pp"]/100.0
            print("[fig_code14] global safety: divided y by 100 (bps->pp)")
        # print concise global summary
        p01 = df["cg_next_pp"].quantile(0.01)
        med = df["cg_next_pp"].quantile(0.50)
        p99 = df["cg_next_pp"].quantile(0.99)
        print(f"[fig_code14] global y(pp) p01={p01:.2f}  median={med:.2f}  p99={p99:.2f}")
    except Exception:
        pass

    # Drop rows missing essentials
    df = df.dropna(subset=["phi", "cg_next_pp", "tercile"])
    # Lock order
    df["tercile"] = pd.Categorical(df["tercile"], categories=TERCILE_CANON, ordered=True)
    return df


def _plot(ax: plt.Axes,
          df: pd.DataFrame,
          include_ols: bool = False,
          allow_lowess: bool = True,
          min_n_fit: int = 6,
          min_unique_x: int = 5,
          frac: float = 0.6,
          gap_threshold: float = 0.10,
          y_limits: tuple[float, float] | None = None):
    """
    Plot scatter + LOWESS (if eligible per group) or mean bar otherwise.
    """
    for key in TERCILE_CANON:
        sub = df[df["tercile"] == key]
        if sub.empty:
            continue

        c = PALETTE[key]
        # Scatter
        ax.scatter(sub["phi"], sub["cg_next_pp"], s=20, color=c, edgecolor="white", label=key)

        # If an OLS line will be drawn for Mid, suppress LOWESS for that group
        suppress_lowess_for_mid = include_ols and (key == "Mid CET1 tercile")

        # Conditional LOWESS fit
        do_fit = (allow_lowess and not suppress_lowess_for_mid and HAS_LOWESS and
                  (len(sub) >= min_n_fit) and (sub["phi"].nunique() >= min_unique_x))
        if do_fit:
            sub_sorted = sub.sort_values("phi")
            drawn = _plot_lowess_gapaware(
                ax,
                sub_sorted["phi"].to_numpy(),
                sub_sorted["cg_next_pp"].to_numpy(),
                color=c,
                lw=2.2,
                frac=frac,
                gap_threshold=gap_threshold,
                min_seg_span=0.025,
                min_seg_n=3,
            )
            # If no segment was drawn (e.g., ultra-tight x cluster), do nothing (no mean tick).

        # OLS overlay for Mid CET1 tercile
        if include_ols and key == "Mid CET1 tercile":
            # Restrict OLS fit to the focus region: 0.3 <= phi <= 0.5
            sub_sorted = sub.sort_values("phi")
            fit = sub_sorted[(sub_sorted["phi"] >= 0.3) & (sub_sorted["phi"] <= 0.5)]
            if len(fit) >= 3 and fit["phi"].nunique() >= 2:
                x_fit = fit["phi"].to_numpy()
                y_fit = fit["cg_next_pp"].to_numpy()

                # Try to use precomputed CSV beta for US Mid (if this subset is US-only)
                alpha, beta = None, None
                try:
                    if "country" in fit.columns and fit["country"].nunique() == 1 and fit["country"].iloc[0] == "US":
                        beta_path = Path("data_processed") / "fig_code14_regression_summary.csv"
                        if beta_path.exists():
                            bdf = pd.read_csv(beta_path)
                            row = bdf[(bdf["country"] == "US") & (bdf["tercile"].str.contains("Mid"))]
                            if not row.empty:
                                beta = float(row.iloc[0]["beta_phi"])
                                # derive intercept to match sample means
                                alpha = float(y_fit.mean() - beta * x_fit.mean())
                except Exception:
                    alpha, beta = None, None

                if beta is None or alpha is None:
                    # fallback: estimate from the (restricted) data
                    import statsmodels.api as sm
                    X = sm.add_constant(x_fit)
                    model = sm.OLS(y_fit, X).fit()
                    alpha = float(model.params[0])
                    beta = float(model.params[1])

                # Draw OLS as dashed dark-gray line, extended to x=0.6
                xs_line = np.linspace(0.3, 0.6, 200)
                ols_color = "#333333"
                ax.plot(xs_line, alpha + beta * xs_line, color=ols_color,
                        linewidth=1.5, linestyle="--")
        else:
            # No forced fit: show scatter only (no mean bar)
            pass

    # Cosmetics
    ax.set_xlabel("Off-balance share φ (current)")
    ax.set_ylabel("Next-quarter credit growth (pp YoY)")
    ax.grid(True, linestyle=":")
    # Legend: lock order and colors explicitly
    handles = [plt.Line2D([], [], color=PALETTE[k], marker='o', linestyle='', markersize=5, label=k)
               for k in TERCILE_CANON if (df["tercile"] == k).any()]
    ax.legend(handles=handles, frameon=False, loc="upper left")

    if y_limits is not None:
        ax.set_ylim(*y_limits)


def _plot_lowess_gapaware(ax: plt.Axes,
                          x: np.ndarray,
                          y: np.ndarray,
                          color: str,
                          lw: float = 2.2,
                          frac: float = 0.6,
                          gap_threshold: float = 0.10,
                          min_seg_span: float = 0.025,
                          min_seg_n: int = 3) -> bool:
    """
    Draw LOWESS without bridging large x gaps. Additionally, skip any segment
    whose x-span is too small (min_seg_span) or has too few points (min_seg_n)
    to avoid near-vertical artifacts when a cluster is tightly concentrated in x.
    Returns True if at least one segment was drawn; otherwise False.
    """
    if len(x) < 4:
        return False
    order = np.argsort(x)
    xs = np.asarray(x)[order]
    ys = np.asarray(y)[order]
    fitted = lowess(ys, xs, frac=frac, return_sorted=True)
    fx, fy = fitted[:, 0], fitted[:, 1]
    seg_start = 0
    drawn = False
    for i in range(1, len(fx)):
        if (fx[i] - fx[i-1]) > gap_threshold:
            if i - seg_start >= min_seg_n and (fx[i-1] - fx[seg_start]) >= min_seg_span:
                ax.plot(fx[seg_start:i], fy[seg_start:i], color=color, lw=lw)
                drawn = True
            seg_start = i
    # last segment
    if len(fx) - seg_start >= min_seg_n and (fx[-1] - fx[seg_start]) >= min_seg_span:
        ax.plot(fx[seg_start:], fy[seg_start:], color=color, lw=lw)
        drawn = True
    return drawn


def _export(fig: plt.Figure, outbase: Path, dpi: int = DPI, export_tiff: bool = False):
    out_png = outbase.with_suffix(".png")
    out_pdf = outbase.with_suffix(".pdf")
    out_eps = outbase.with_suffix(".eps")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_eps, dpi=dpi, bbox_inches="tight")
    if export_tiff:
        fig.savefig(outbase.with_suffix(".tiff"), dpi=max(300, dpi), bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(
        description="Credit growth (t+1) vs phi by CET1/RWA terciles (conditional LOWESS)."
    )
    parser.add_argument(
        "--csv",
        default=str(Path("data_processed") / "proc_code09_phi_capital_interaction_panel.csv"),
        help="Input CSV path (default: data_processed/proc_code09_phi_capital_interaction_panel.csv)",
    )
    parser.add_argument("--outfile", default=str(Path("figures") / "fig_code14_scatter_phi_capital_thinness"))
    parser.add_argument("--title", default="Credit growth (t+1) vs φ by CET1/RWA terciles")
    parser.add_argument("--no-title", action="store_true", help="Hide figure title (FRL-compact)")
    parser.add_argument("--title-size", type=float, default=8.0, help="Title font size (default: 8)")
    parser.add_argument("--min-n-fit", type=int, default=6, help="Min observations per tercile to run LOWESS")
    parser.add_argument("--min-ux", type=int, default=5, help="Min unique x-values per tercile to run LOWESS")
    parser.add_argument("--frac", type=float, default=0.6, help="LOWESS smoothing fraction")
    parser.add_argument("--gap", type=float, default=0.10, help="Max x-gap to connect in LOWESS plotting")
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--export-tiff", action="store_true")
    args = parser.parse_args()

    _configure_matplotlib()
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)

    df = _load_and_prepare(Path(args.csv))

    # Compute robust y-limits from the central 1–99 percentiles with a small pad
    y_lo = float(df["cg_next_pp"].quantile(0.01))
    y_hi = float(df["cg_next_pp"].quantile(0.99))
    span = y_hi - y_lo
    pad = max(1.5, 0.1 * span)
    y_limits = (y_lo - pad, y_hi + pad)

    # Write a small debug view of the input actually used by this figure
    debug_csv = Path("data_processed") / "fig_code14_input_check.csv"
    try:
        cols_dbg = [c for c in ["country", "phi", "cg_next_pp", "tercile"] if c in df.columns]
        df[cols_dbg].to_csv(debug_csv, index=False)
        print(f"[fig_code14] Wrote debug CSV: {debug_csv}")
    except Exception as _e:
        warnings.warn(f"Could not write debug CSV: {_e}")

    base = Path(args.outfile)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))
    _plot(ax, df, include_ols=False, min_n_fit=args.min_n_fit, min_unique_x=args.min_ux, frac=args.frac, gap_threshold=args.gap, y_limits=y_limits)
    if not args.no_title:
        ax.set_title(args.title, fontsize=args.title_size, pad=2)
    _export(fig, base.with_name(base.name + "_wo_OLS"), dpi=args.dpi, export_tiff=args.export_tiff)
    plt.close(fig)
    print(f"Saved: {base.with_name(base.name + '_wo_OLS').with_suffix('.png')}")

    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))
    _plot(ax, df, include_ols=True, min_n_fit=args.min_n_fit, min_unique_x=args.min_ux, frac=args.frac, gap_threshold=args.gap, y_limits=y_limits)
    if not args.no_title:
        ax.set_title(args.title + " (with OLS)", fontsize=args.title_size, pad=2)
    _export(fig, base.with_name(base.name + "_w_OLS"), dpi=args.dpi, export_tiff=args.export_tiff)
    plt.close(fig)
    print(f"Saved: {base.with_name(base.name + '_w_OLS').with_suffix('.png')}")

    # --- Optional: country-specific outputs (JP-only / US-only) ---
    if "country" in df.columns:
        for cc in ("JP", "US"):
            sub = df[df["country"] == cc].copy()
            if sub.empty:
                continue

            # Robust y-limits for this subset
            y_lo_c = float(sub["cg_next_pp"].quantile(0.01))
            y_hi_c = float(sub["cg_next_pp"].quantile(0.99))
            span_c = y_hi_c - y_lo_c
            pad_c = max(1.5, 0.1 * span_c)
            y_limits_c = (y_lo_c - pad_c, y_hi_c + pad_c)

            # JP/US specific debug CSV
            try:
                cols_dbg = [c for c in ["country", "phi", "cg_next_pp", "tercile"] if c in sub.columns]
                dbg_path = Path("data_processed") / f"fig_code14_input_check_{cc}.csv"
                sub[cols_dbg].to_csv(dbg_path, index=False)
                print(f"[fig_code14] Wrote debug CSV ({cc}): {dbg_path}")
            except Exception as _e:
                warnings.warn(f"Could not write {cc} debug CSV: {_e}")

            # File base: add country tag
            base_c = base.with_name(base.name + f"_{cc}_only")

            # wo_OLS figure for this country
            fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))
            _plot(ax, sub,
                  include_ols=False,
                  allow_lowess=(cc != "JP"),  # JP_only: no fitted line
                  min_n_fit=args.min_n_fit, min_unique_x=args.min_ux,
                  frac=args.frac, gap_threshold=args.gap, y_limits=y_limits_c)
            if not args.no_title:
                ax.set_title(args.title + f" — {cc} only", fontsize=args.title_size, pad=2)
            _export(fig, base_c.with_name(base_c.name + "_wo_OLS"), dpi=args.dpi, export_tiff=args.export_tiff)
            plt.close(fig)
            print(f"Saved: {base_c.with_name(base_c.name + '_wo_OLS').with_suffix('.png')}")

            # w_OLS figure for this country
            fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))
            _plot(ax, sub,
                  include_ols=(cc != "JP"),
                  allow_lowess=(cc != "JP"),
                  min_n_fit=args.min_n_fit, min_unique_x=args.min_ux,
                  frac=args.frac, gap_threshold=args.gap, y_limits=y_limits_c)
            if not args.no_title:
                ax.set_title(args.title + f" — {cc} only (with OLS)", fontsize=args.title_size, pad=2)
            _export(fig, base_c.with_name(base_c.name + "_w_OLS"), dpi=args.dpi, export_tiff=args.export_tiff)
            plt.close(fig)
            print(f"Saved: {base_c.with_name(base_c.name + '_w_OLS').with_suffix('.png')}")

    # --- Automatic OLS regression summary (US Mid CET1, phi <= 0.6) ---
    try:
        # Only proceed if necessary columns exist
        if {"country", "tercile", "phi", "cg_next_pp"}.issubset(df.columns):
            # Filter to US + Mid CET1 tercile + phi <= 0.6
            sub_reg = df[(df["country"] == "US") &
                         (df["tercile"] == "Mid CET1 tercile") &
                         (df["phi"] <= 0.6)].copy()
            sub_reg = sub_reg.dropna(subset=["phi", "cg_next_pp"])  # ensure no NaNs
            if len(sub_reg) >= 5:
                import statsmodels.api as sm
                X = sm.add_constant(sub_reg["phi"])  # const + phi
                y = sub_reg["cg_next_pp"]
                model = sm.OLS(y, X).fit(cov_type="HC3")  # robust SE by default
                res = {
                    "country": "US",
                    "tercile": "Mid CET1 tercile",
                    "n": int(len(sub_reg)),
                    "phi_min": float(sub_reg["phi"].min()),
                    "phi_max": float(sub_reg["phi"].max()),
                    "beta_phi": float(model.params.get("phi", float("nan"))),
                    "t_value": float(model.tvalues.get("phi", float("nan"))),
                    "p_value": float(model.pvalues.get("phi", float("nan"))),
                    "r2_adj": float(model.rsquared_adj),
                }
                reg_out = Path("data_processed") / "fig_code14_regression_summary.csv"
                pd.DataFrame([res]).to_csv(reg_out, index=False)
                print(f"[fig_code14] Wrote OLS summary → {reg_out}")
            else:
                print("[fig_code14] OLS summary skipped: not enough US Mid observations (need >=5).")
    except Exception as e:
        warnings.warn(f"OLS regression failed: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        warnings.warn(str(e))
        sys.exit(1)
