"""fig_code06_scatter_dsr_creditgrowth

US single-country scatter: Credit growth YoY (pp) vs Debt Service Ratio (DSR, %).

Two data sourcing modes:
    (A) Raw source mode (default):
            - BIS DSR (largest bis_dp_search_export_*.csv)
            - HHMSDODNS (mortgage outstanding) -> 4Q pct change proxy
        (B) Processed panel fallback (--panel or auto):
                        - Prefers data_processed/proc_code03_US_DSR_CreditGrowth_panel.csv (if present)
                            else falls back to legacy data_processed/US_DSR_CreditGrowth_panel.csv
                            Expected columns: DSR_pct, CreditGrowth_ppYoY (already aligned quarterly)

If raw sources are missing, the script auto-falls back to the panel CSV when present.
Outputs always get fig_code06_ prefix via figure_name_with_code.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from pathlib import Path
from util_code01_lib_io import find, safe_savefig, ensure_unique, safe_to_csv, figure_name_with_code

OUTDIR = Path("figures")
PREF_PANEL = Path("data_processed/proc_code03_US_DSR_CreditGrowth_panel.csv")
LEG_PANEL  = Path("data_processed/US_DSR_CreditGrowth_panel.csv")
PANEL_DEFAULT = PREF_PANEL if PREF_PANEL.exists() else (LEG_PANEL if LEG_PANEL.exists() else LEG_PANEL)

def load_bis_dsr(country_code="US", sector="P"):
    """Return (df, err) where df has a single column 'DSR'.

    Picks the largest BIS export file (assumed most complete). Filters by KEY pattern Q.<country>.<sector>.
    """
    files = find(["bis_dp_search_export_*.csv"])
    if not files: return None, "BIS DSR export not found"
    files = sorted(files, key=lambda p: p.stat().st_size, reverse=True)
    df = pd.read_csv(files[0], skiprows=2)
    df.columns = [c.strip() for c in df.columns]
    keycol = [c for c in df.columns if "KEY" in c][0]
    tcol   = [c for c in df.columns if "TIME" in c][0]
    vcol   = [c for c in df.columns if "OBS_VALUE" in c][0]
    pick = df[df[keycol].str.contains(f"Q.{country_code}.{sector}", na=False)].copy()
    if pick.empty: return None, f"no DSR for {country_code}.{sector}"
    pick[tcol] = pd.to_datetime(pick[tcol], errors="coerce")
    pick[vcol] = pd.to_numeric(pick[vcol], errors="coerce")
    pick = pick.dropna(subset=[tcol, vcol]).set_index(tcol).sort_index()
    return pick[[vcol]].rename(columns={vcol:"DSR"}), None

def load_credit_growth_proxy_us():
    """Return (df, err) containing quarterly CreditYoY_% from HHMSDODNS.

    CreditYoY_% = 4-quarter percent change * 100.
    """
    files = find(["HHMSDODNS.csv"])
    if not files: return None, "US mortgage CSV not found"
    s = pd.read_csv(files[0])
    dt = [c for c in s.columns if "DATE" in c.upper()][0]
    val= [c for c in s.columns if c != dt][0]
    s[dt] = pd.to_datetime(s[dt], errors="coerce")
    s[val]= pd.to_numeric(s[val], errors="coerce")
    s = s.dropna(subset=[dt, val]).set_index(dt).sort_index()
    g = (s[val].pct_change(4)*100).rename("CreditYoY_%")
    return g.to_frame(), None

def _from_panel(panel_path: Path):
    if not panel_path.exists():
        return None, f"panel CSV not found: {panel_path}"
    df = pd.read_csv(panel_path, index_col=0)
    # try to parse index
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    needed = {'DSR_pct','CreditGrowth_ppYoY'}
    if not needed.issubset(df.columns):
        return None, f"panel missing columns {needed - set(df.columns)}"
    tidy = df[['DSR_pct','CreditGrowth_ppYoY']].dropna()
    tidy = tidy.rename(columns={'DSR_pct':'DSR','CreditGrowth_ppYoY':'CreditYoY_%'})
    if tidy.empty:
        return None, 'panel after dropna empty'
    return tidy, None

def build_dataset(use_panel: bool=False, panel_path: Path=PANEL_DEFAULT):
    if use_panel:
        return _from_panel(panel_path)
    dsr, e1 = load_bis_dsr("US","P")
    cg , e2 = load_credit_growth_proxy_us()
    if dsr is None or cg is None:
        # fallback if panel exists
        panel, pe = _from_panel(panel_path)
        if panel is not None:
            print(f"[INFO] fallback to panel CSV ({panel_path}) due to raw missing: {e1 or ''} {e2 or ''}")
            return panel, None
        return None, f"raw missing ({e1},{e2}) and panel fallback failed ({pe})"
    df = dsr.join(cg, how='inner').dropna()
    if df.empty:
        panel, pe = _from_panel(panel_path)
        if panel is not None:
            print(f"[INFO] fallback to panel CSV ({panel_path}) due to join empty")
            return panel, None
        return None, f"no overlap (and panel fallback failed: {pe})"
    return df, None

def make(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='US DSR vs Credit growth scatter (raw or panel fallback)')
    ap.add_argument('--panel', action='store_true', help='Force use of processed panel CSV instead of raw sources')
    ap.add_argument('--panel-csv', default=str(PANEL_DEFAULT))
    ap.add_argument('--out-prefix', default='SCATTER_US_DSR_vs_CreditGrowth_FINAL')
    args = ap.parse_args(argv)
    df, err = build_dataset(use_panel=args.panel, panel_path=Path(args.panel_csv))
    if df is None:
        print('[DSR scatter] skip:', err)
        return 0
    fig, ax = plt.subplots(figsize=(5.5,4.2))
    ax.scatter(df["DSR"], df["CreditYoY_%"], alpha=0.85, edgecolor='k', linewidths=0.3)

    # OLS with intercept
    n = len(df)
    X = np.vstack([np.ones(n), df["DSR"].values]).T
    y = df["CreditYoY_%"].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - y.mean())**2))
    r2 = 1 - sse/sst if sst > 0 else float('nan')
    df_resid = max(n - 2, 1)
    # variance-covariance matrix
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
        sigma2 = sse / df_resid
        se_slope = sqrt(sigma2 * xtx_inv[1,1]) if xtx_inv[1,1] > 0 else float('nan')
    except Exception:
        se_slope = float('nan')
    if se_slope and se_slope != 0 and not np.isnan(se_slope):
        t_slope = beta[1] / se_slope
    else:
        t_slope = float('nan')
    # p-value (two-sided) using scipy if available else normal approximation
    def _p_two_sided(t, df_):
        try:
            from scipy.stats import t as student_t  # type: ignore
            return 2*student_t.sf(abs(t), df_)
        except Exception:
            # normal approx
            from math import erf, sqrt
            if np.isnan(t):
                return float('nan')
            return 2*(1-0.5*(1+erf(abs(t)/sqrt(2))))
    p_slope = _p_two_sided(t_slope, df_resid)

    xg = np.linspace(df["DSR"].min(), df["DSR"].max(), 200)
    ax.plot(xg, beta[0] + beta[1]*xg, linewidth=1.0, label=f"OLS slope={beta[1]:.3f}")
    ax.set_xlabel("Debt Service Ratio (US, PNFS)")
    ax.set_ylabel("Credit growth YoY (proxy, %)")
    ax.set_title("US: Credit growth vs DSR (quarterly)", color="black")
    ax.grid(True, alpha=0.3)
    # Annotation box
    stats_text = (
        f"n={n}\nR²={r2:.3f}\nintercept={beta[0]:.3f}\n"
        f"slope={beta[1]:.3f}\nt={t_slope:.2f}\np={p_slope:.3g}"
    )
    ax.legend(loc="best")
    ax.text(0.02, 0.98, stats_text, ha='left', va='top', transform=ax.transAxes,
            fontsize=9, family='monospace', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', linewidth=0.5))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig_path = figure_name_with_code(__file__, OUTDIR/f"{args.out_prefix}.png")
    fig_written = ensure_unique(fig_path)
    safe_savefig(fig, fig_written)
    csv_path = figure_name_with_code(__file__, OUTDIR/f"{args.out_prefix}_beta.csv")
    final_csv = safe_to_csv(pd.DataFrame({
        "intercept":[beta[0]],
        "slope":[beta[1]],
        "R2":[r2],
        "t_slope":[t_slope],
        "p_slope":[p_slope],
        "n":[n]
    }), csv_path, index=False)
    print("WROTE:", fig_written)
    print("WROTE:", final_csv)

if __name__=="__main__":
    make()
