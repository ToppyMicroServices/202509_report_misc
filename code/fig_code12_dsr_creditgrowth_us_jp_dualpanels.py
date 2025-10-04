"""fig_code12_dsr_creditgrowth_us_jp_dualpanels

Definitive comparative scatter: Credit growth (pp YoY) vs Debt Service Ratio (DSR) for US and Japan,
stacked vertically in two panels with a shared (or optional independent) scale.

Default inputs (panel format, prefixed preferred):
    data_processed/proc_code03_US_DSR_CreditGrowth_panel.csv  (index=QuarterEnd, cols: DSR_pct, CreditGrowth_ppYoY)
    data_processed/proc_code03_JP_DSR_CreditGrowth_panel.csv  (same structure)
Legacy fallback names (auto-detected if prefixed missing):
    data_processed/US_DSR_CreditGrowth_panel.csv
    data_processed/JP_DSR_CreditGrowth_panel.csv

Features:
    - Unified axis ranges (unless --independent-axes) for visual comparability
    - OLS regression (CreditGrowth_ppYoY ~ 1 + DSR_pct) with dotted fit line and rich annotation
    - Year filtering (--start / --end) and last-N trimming (--last-n, default all)
    - Batch mode producing 4 variants (shared, independent, US-only, JP-only) when --batch or when no args
    - Safe figure saving (no overwrite) and automatic filename prefixing
    - Optional simplified annotation (--simple-annot)

Example:
    python code/fig_code12_dsr_creditgrowth_us_jp_dualpanels.py --batch --prefix figures/SCATTER_DSR_CreditGrowth

Note: Panel CSVs typically originate from fig_code06 internal computations.
"""
from __future__ import annotations
import argparse
import sys
from typing import Literal
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from util_code01_lib_io import safe_savefig, figure_name_with_code
from util_code02_colors import COLOR_US, COLOR_JP

PREF_US = Path('data_processed/proc_code03_US_DSR_CreditGrowth_panel.csv')
PREF_JP = Path('data_processed/proc_code03_JP_DSR_CreditGrowth_panel.csv')
LEG_US = Path('data_processed/US_DSR_CreditGrowth_panel.csv')
LEG_JP = Path('data_processed/JP_DSR_CreditGrowth_panel.csv')
DEFAULT_US = PREF_US if PREF_US.exists() else (LEG_US if LEG_US.exists() else PREF_US)
DEFAULT_JP = PREF_JP if PREF_JP.exists() else (LEG_JP if LEG_JP.exists() else PREF_JP)
DEFAULT_OUT_DUAL = Path('figures/SCATTER_DSR_CreditGrowth_US_JP_STACKED.png')
DEFAULT_OUT_DUAL_INDEP = Path('figures/SCATTER_DSR_CreditGrowth_US_JP_STACKED_INDEPAX.png')
DEFAULT_OUT_US_SINGLE = Path('figures/SCATTER_US_DSR_vs_CreditGrowth_FINAL.png')  # legacy fig_code06 name
DEFAULT_OUT_JP_SINGLE = Path('figures/SCATTER_JP_DSR_vs_CreditGrowth_FINAL.png')

# Country color specification (consistent across figures)
US_COLOR = COLOR_US
JP_COLOR = COLOR_JP
REG_LINE_ALPHA = 0.9

def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    # index_col=0 converts the first (empty) index column back to datetime
    df = pd.read_csv(path, index_col=0)
    # Expected columns: DSR_pct, CreditGrowth_ppYoY
    for c in ['DSR_pct','CreditGrowth_ppYoY']:
        if c not in df.columns:
            raise ValueError(f"column '{c}' missing in {path}")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Coerce to numeric & drop NaNs
    df['DSR_pct'] = pd.to_numeric(df['DSR_pct'], errors='coerce')
    df['CreditGrowth_ppYoY'] = pd.to_numeric(df['CreditGrowth_ppYoY'], errors='coerce')
    return df.dropna(subset=['DSR_pct','CreditGrowth_ppYoY'])

def ols_slope(df: pd.DataFrame):
    if df.empty:
        return np.nan, np.nan
    X = np.vstack([np.ones(len(df)), df['DSR_pct'].values]).T
    y = df['CreditGrowth_ppYoY'].values
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    n = len(y)
    k = X.shape[1]
    # Residual sum of squares
    if residuals.size:
        rss = residuals[0]
    else:
        # compute manually if lstsq did not return residuals (rare when n==k)
        rss = float(np.sum((y - X @ beta)**2))
    tss = float(np.sum((y - y.mean())**2)) if n > 0 else np.nan
    r2 = 1 - rss / tss if tss > 0 else np.nan
    sigma2 = rss / (n - k) if n > k else np.nan
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(xtx_inv) * sigma2)
    except np.linalg.LinAlgError:
        se = [np.nan, np.nan]
    intercept, slope = beta
    se_intercept, se_slope = se
    return intercept, slope, se_intercept, se_slope, r2, n

def _p_value_from_t(t: float, df: int) -> float:
    """Return two-sided p-value for a t statistic.
    Uses scipy if available; otherwise normal approximation (sufficient for df>30)."""
    if not np.isfinite(t) or df <= 0:
        return float('nan')
    try:  # pragma: no cover
        from scipy import stats  # type: ignore
        return float(2 * stats.t.sf(abs(t), df))
    except Exception:  # fallback normal approx
        # survival function for |Z| > |t|
        return float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))

def make(
    us_df: pd.DataFrame,
    jp_df: pd.DataFrame,
    out: Path,
    title: str|None=None,
    show: bool=False,
    mode: Literal['dual','us','jp']='dual',
    simple_annot: bool=False,
    marker: str='o',
    point_alpha: float=0.85,
    country_axis_labels: bool=False,
    independent_axes: bool=False,
):
    # 軸範囲 (共有前提の場合はここで統一レンジ計算)
    x_min = min(us_df['DSR_pct'].min(), jp_df['DSR_pct'].min())
    x_max = max(us_df['DSR_pct'].max(), jp_df['DSR_pct'].max())
    y_min = min(us_df['CreditGrowth_ppYoY'].min(), jp_df['CreditGrowth_ppYoY'].min())
    y_max = max(us_df['CreditGrowth_ppYoY'].max(), jp_df['CreditGrowth_ppYoY'].max())

    if mode == 'dual':
        if independent_axes:
            fig, axes = plt.subplots(2, 1, figsize=(6.4, 8.2), sharex=False, sharey=False)
        else:
            fig, axes = plt.subplots(2, 1, figsize=(6.4, 8.2), sharex=True, sharey=True)
        panels = [
            (us_df, 'United States', US_COLOR),
            (jp_df, 'Japan', JP_COLOR),
        ]
        for ax, (df, label, c) in zip(axes, panels):
            ax.scatter(df['DSR_pct'], df['CreditGrowth_ppYoY'], s=26, alpha=point_alpha, color=c, edgecolor='none', marker=marker)
            intercept, slope, se_int, se_slope, r2, n = ols_slope(df)
            if np.isfinite(slope):
                if independent_axes:
                    lx_min = df['DSR_pct'].min(); lx_max = df['DSR_pct'].max()
                else:
                    lx_min, lx_max = x_min, x_max
                x_line = np.array([lx_min, lx_max])
                y_line = intercept + slope * x_line
                ax.plot(x_line, y_line, '--', color=c, lw=1.4, alpha=REG_LINE_ALPHA,
                        label=f"OLS slope={slope:.3f}")
                if np.isfinite(r2):
                    dfree = n - 2
                    t_beta = slope / se_slope if se_slope and np.isfinite(se_slope) else float('nan')
                    p_beta = _p_value_from_t(t_beta, dfree)
                    t_alpha = intercept / se_int if se_int and np.isfinite(se_int) else float('nan')
                    p_alpha = _p_value_from_t(t_alpha, dfree)
                    if simple_annot:
                        # compact: slope form + t,p, R2, N
                        txt = (f"y = {slope:.3f}·x + {intercept:.3f}\n"
                               f"β t={t_beta:.2f}, p={p_beta:.3g}\nR² = {r2:.3f}, N={n}")
                    else:
                        txt = (f"y = α + β·x\nα = {intercept:.2f} (SE {se_int:.2f}, t={t_alpha:.2f}, p={p_alpha:.3g})\n"
                               f"β = {slope:.2f} (SE {se_slope:.2f}, t={t_beta:.2f}, p={p_beta:.3g})\n"
                               f"R² = {r2:.3f}, N = {n}")
                    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
                            fontsize=9, bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=c, alpha=0.85))
                leg = ax.legend(loc='best', fontsize=9, frameon=True)
                if leg:
                    leg.get_frame().set_edgecolor('black')
                    leg.get_frame().set_facecolor('white')
                    handles = getattr(leg, 'legend_handles', None) or getattr(leg, 'legendHandles', None) or []
                    for h in handles:
                        try:
                            if hasattr(h, 'set_color'):
                                h.set_color('black')
                            if hasattr(h, 'set_markerfacecolor'):
                                h.set_markerfacecolor('black')
                            if hasattr(h, 'set_markeredgecolor'):
                                h.set_markeredgecolor('black')
                        except Exception:
                            pass
            if country_axis_labels:
                ax.set_title(f"{'US' if label.startswith('United States') else 'Japan'}: Credit-to-GDP YoY change vs DSR (PNFS)", fontsize=13, color='black')
                ylab = 'CreditGrowth_US_ppYoY (pp, YoY)' if label.startswith('United States') else 'CreditGrowth_JP_ppYoY (pp, YoY)'
                ax.set_ylabel(ylab)
            else:
                ax.set_title(label, fontsize=12, color='black')
                ax.set_ylabel('Credit growth (pp YoY)')
            ax.grid(True, alpha=0.3)
            if independent_axes:
                # 個別レンジ
                ax.set_xlim(df['DSR_pct'].min(), df['DSR_pct'].max())
                ax.set_ylim(df['CreditGrowth_ppYoY'].min(), df['CreditGrowth_ppYoY'].max())
            else:
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
        axes[-1].set_xlabel('Debt Service Ratio (%)' if not country_axis_labels else 'DSR_PNFS_US (level, %)')
    else:
        is_us = (mode == 'us')
        df = us_df if is_us else jp_df
        c = US_COLOR if is_us else JP_COLOR
        label = 'United States' if is_us else 'Japan'
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.0))
        ax.scatter(df['DSR_pct'], df['CreditGrowth_ppYoY'], s=36, alpha=point_alpha, color=c, edgecolor='none', marker=marker)
        intercept, slope, se_int, se_slope, r2, n = ols_slope(df)
        if np.isfinite(slope):
            x_line = np.array([x_min, x_max])
            y_line = intercept + slope * x_line
            ax.plot(x_line, y_line, '--', color=c, lw=1.6, alpha=REG_LINE_ALPHA,
                    label=f"OLS slope={slope:.3f}")
            if np.isfinite(r2):
                dfree = n - 2
                t_beta = slope / se_slope if se_slope and np.isfinite(se_slope) else float('nan')
                p_beta = _p_value_from_t(t_beta, dfree)
                t_alpha = intercept / se_int if se_int and np.isfinite(se_int) else float('nan')
                p_alpha = _p_value_from_t(t_alpha, dfree)
                if simple_annot:
                    txt = (f"y = {slope:.3f}·x + {intercept:.3f}\n"
                           f"β t={t_beta:.2f}, p={p_beta:.3g}\nR² = {r2:.3f}, N={n}")
                else:
                    txt = (f"y = α + β·x\nα = {intercept:.2f} (SE {se_int:.2f}, t={t_alpha:.2f}, p={p_alpha:.3g})\n"
                           f"β = {slope:.2f} (SE {se_slope:.2f}, t={t_beta:.2f}, p={p_beta:.3g})\n"
                           f"R² = {r2:.3f}, N = {n}")
                ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
                        fontsize=9, bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=c, alpha=0.85))
            leg = ax.legend(loc='best', fontsize=9, frameon=True)
            if leg:
                leg.get_frame().set_edgecolor('black')
                leg.get_frame().set_facecolor('white')
                handles = getattr(leg, 'legend_handles', None) or getattr(leg, 'legendHandles', None) or []
                for h in handles:
                    try:
                        if hasattr(h, 'set_color'):
                            h.set_color('black')
                        if hasattr(h, 'set_markerfacecolor'):
                            h.set_markerfacecolor('black')
                        if hasattr(h, 'set_markeredgecolor'):
                            h.set_markeredgecolor('black')
                    except Exception:
                        pass
        if country_axis_labels:
            ax.set_title(f"{'US' if is_us else 'Japan'}: Credit-to-GDP YoY change vs DSR (PNFS)", fontsize=14, color='black')
            ax.set_xlabel('DSR_PNFS_US (level, %)' if is_us else 'DSR_PNFS_JP (level, %)')
            ax.set_ylabel('CreditGrowth_US_ppYoY (pp, YoY)' if is_us else 'CreditGrowth_JP_ppYoY (pp, YoY)')
        else:
            ax.set_title(f"{label}: Credit growth vs DSR (quarterly)", fontsize=12, color='black')
            ax.set_xlabel('Debt Service Ratio (%)')
            ax.set_ylabel('Credit growth (pp YoY)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    if title and mode=='dual' and not country_axis_labels:
        fig.suptitle(title, fontsize=13, y=0.97)
    out.parent.mkdir(parents=True, exist_ok=True)
    out = figure_name_with_code(__file__, out)
    # Default: overwrite existing files to keep filenames stable
    safe_savefig(fig, out, overwrite=True)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out

def build_us_on_the_fly() -> pd.DataFrame | None:
    """If panel CSV is missing, attempt raw reconstruction similar to legacy fig_code06."""
    try:
        from util_code01_lib_io import find
    except Exception:
        return None
    # BIS DSR
    files = find(["bis_dp_search_export_*.csv"])
    if not files:
        return None
    bis = pd.read_csv(sorted(files, key=lambda p: p.stat().st_size, reverse=True)[0], skiprows=2)
    bis.columns = [c.strip() for c in bis.columns]
    keycol = [c for c in bis.columns if 'KEY' in c][0]
    tcol = [c for c in bis.columns if 'TIME' in c][0]
    vcol = [c for c in bis.columns if 'OBS_VALUE' in c][0]
    pick = bis[bis[keycol].str.contains('Q.US.P', na=False)].copy()
    if pick.empty: return None
    pick[tcol] = pd.to_datetime(pick[tcol], errors='coerce')
    pick[vcol] = pd.to_numeric(pick[vcol], errors='coerce')
    pick = pick.dropna(subset=[tcol, vcol]).set_index(tcol).sort_index()
    dsr = pick[[vcol]].rename(columns={vcol: 'DSR_pct'})
    # Credit proxy
    hh = find(['HHMSDODNS.csv'])
    if not hh: return None
    s = pd.read_csv(hh[0])
    dt = [c for c in s.columns if 'DATE' in c.upper()][0]
    val = [c for c in s.columns if c != dt][0]
    s[dt] = pd.to_datetime(s[dt], errors='coerce')
    s[val] = pd.to_numeric(s[val], errors='coerce')
    s = s.dropna(subset=[dt, val]).set_index(dt).sort_index()
    cg = (s[val].pct_change(4) * 100).rename('CreditGrowth_ppYoY')
    df = dsr.join(cg, how='inner').dropna()
    return df

def main(argv=None):
    ap = argparse.ArgumentParser(description='US vs JP DSR vs Credit Growth dual-panel (same scale)')
    ap.add_argument('--us-csv', default=str(DEFAULT_US))
    ap.add_argument('--jp-csv', default=str(DEFAULT_JP))
    ap.add_argument('--out', default=None, help='output file (auto by mode if omitted)')
    ap.add_argument('--start', type=int, default=None, help='filter start year (inclusive)')
    ap.add_argument('--end', type=int, default=None, help='filter end year (inclusive)')
    ap.add_argument('--title', default='DSR vs Credit Growth: US vs Japan (Quarterly)')
    ap.add_argument('--mode', choices=['dual','us','jp'], default='dual', help='dual (both panels) or single country')
    ap.add_argument('--last-n', default='all', help="use only the last N observations (after year filtering); 'all' for full sample (default)")
    ap.add_argument('--show', action='store_true', help='force interactive display even with arguments')
    ap.add_argument('--simple-annot', action='store_true', help='simpler annotation (formula + R2)')
    ap.add_argument('--marker', default='o', help='matplotlib marker style (e.g. o, x, s)')
    ap.add_argument('--alpha', type=float, default=0.65, help='point alpha (default 0.65)')
    ap.add_argument('--country-axis-labels', action='store_true', help='use country variable style axis labels')
    ap.add_argument('--independent-axes', action='store_true', help='dualモード: 各国で別スケール (sharex/sharey 無効)')
    ap.add_argument('--batch', action='store_true', help='代表4パターンの図を一括出力 (dual shared / dual indep / US / JP)')
    ap.add_argument('--prefix', default='figures/SCATTER_DSR_CreditGrowth', help='--batch 使用時のファイル名前接頭辞')
    args = ap.parse_args(argv)

    # If user invoked with no CLI arguments at all, auto-enable batch mode
    # so that all representative figures are produced without needing --batch.
    import sys as _sys
    if len(_sys.argv) == 1 and not args.batch:
        args.batch = True
        print('[INFO] no arguments detected -> auto batch mode (producing all variants)')

    # Load US (with fallback build if file missing and mode needs US)
    us_csv_path = Path(args.us_csv)
    if us_csv_path.exists():
        us = load_panel(us_csv_path)
    else:
        us = build_us_on_the_fly() if args.mode in ('dual','us') else None
        if us is None and args.mode in ('dual','us'):
            print('[ERROR] US panel not found and raw rebuild failed')
            return 1
    jp = load_panel(Path(args.jp_csv))
    def _year_filter(df):
        if args.start:
            df = df[df.index.year >= args.start]
        if args.end:
            df = df[df.index.year <= args.end]
        return df
    us_f = _year_filter(us) if us is not None else None
    jp_f = _year_filter(jp) if jp is not None else None
    # Interpret last-n
    last_n = None
    if args.last_n not in (None, ''):
        if isinstance(args.last_n, str):
            if args.last_n.lower() in ('all','full','*','max'):
                last_n = None  # full sample
            else:
                try:
                    last_n = int(args.last_n)
                except ValueError:
                    print(f"[WARN] invalid --last-n '{args.last_n}' ignored (using full sample)")
                    last_n = None
        else:
            last_n = int(args.last_n)
    if last_n:
        if us_f is not None:
            us_f = us_f.iloc[-last_n:]
        if jp_f is not None:
            jp_f = jp_f.iloc[-last_n:]
    if args.mode == 'dual':
        if us_f.empty or jp_f.empty:
            print('[ERROR] filtered data empty (check year range or inputs)')
            return 1
    elif args.mode == 'us':
        if us_f is None or us_f.empty:
            print('[ERROR] US data empty')
            return 1
    elif args.mode == 'jp':
        if jp_f is None or jp_f.empty:
            print('[ERROR] JP data empty')
            return 1
    # Determine output path if not provided
    # Batch generation path
    if args.batch:
        prefix_path = Path(args.prefix)
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        generated = []
        show_flag = False  # batch時は自動表示しない
        # 1. dual shared axes
        dual_shared_out = prefix_path.with_name(prefix_path.name + '_DUAL_SHARED.png')
        generated.append(make(
            us_f, jp_f, figure_name_with_code(__file__, dual_shared_out), args.title, show=show_flag, mode='dual',
            simple_annot=args.simple_annot, marker=args.marker, point_alpha=args.alpha,
            country_axis_labels=args.country_axis_labels, independent_axes=False))
        # 2. dual independent axes
        dual_indep_out = prefix_path.with_name(prefix_path.name + '_DUAL_INDEPAX.png')
        generated.append(make(
            us_f, jp_f, figure_name_with_code(__file__, dual_indep_out), args.title, show=show_flag, mode='dual',
            simple_annot=args.simple_annot, marker=args.marker, point_alpha=args.alpha,
            country_axis_labels=args.country_axis_labels, independent_axes=True))
        # 3. single US (軸レンジをUSに合わせるため us_f を両引数に)
        us_single_out = prefix_path.with_name(prefix_path.name + '_US.png')
        generated.append(make(
            us_f, us_f, figure_name_with_code(__file__, us_single_out), args.title, show=show_flag, mode='us',
            simple_annot=args.simple_annot, marker=args.marker, point_alpha=args.alpha,
            country_axis_labels=args.country_axis_labels, independent_axes=False))
        # 4. single JP (同様に jp_f を両引数に)
        jp_single_out = prefix_path.with_name(prefix_path.name + '_JP.png')
        generated.append(make(
            jp_f, jp_f, figure_name_with_code(__file__, jp_single_out), args.title, show=show_flag, mode='jp',
            simple_annot=args.simple_annot, marker=args.marker, point_alpha=args.alpha,
            country_axis_labels=args.country_axis_labels, independent_axes=False))
        print('[OK] batch export complete:')
        for f in generated:
            print('  -', f)
        return 0
    # Single generation path
    if args.out:
        out_path = Path(args.out)
    else:
        if args.mode == 'dual':
            out_path = DEFAULT_OUT_DUAL_INDEP if args.independent_axes else DEFAULT_OUT_DUAL
        elif args.mode == 'us':
            out_path = DEFAULT_OUT_US_SINGLE
        else:
            out_path = DEFAULT_OUT_JP_SINGLE
    show_flag = args.show or (len(sys.argv) == 1)
    final = make(
        us_f, jp_f, out_path, args.title, show=show_flag, mode=args.mode,
        simple_annot=args.simple_annot, marker=args.marker, point_alpha=args.alpha,
        country_axis_labels=args.country_axis_labels, independent_axes=args.independent_axes,
    )
    print(f'[OK] figure written: {final}')
    return 0

if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
