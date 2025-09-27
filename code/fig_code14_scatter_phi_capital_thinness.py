#!/usr/bin/env python3
"""fig_code14_scatter_phi_capital_thinness

Stratified scatter + OLS interaction regression:
  y = next‑quarter credit growth (pp YoY) (credit_growth_ppYoY_lead)
  x = current phi (off‑balance share)
  strata = Thin_tercile (by CET1/RWA percent) from proc_code09 panel.

Panel CSV expected (default): data_processed/proc_code09_phi_capital_interaction_panel.csv

Outputs:
  figures/fig_code14_SCATTER_phi_capital_interaction.png
  figures/fig_code14_SCATTER_phi_capital_interaction_regression.csv (coeff summary)

Model estimated (pooled; country FE optional via demeaning):
  y = a + b1*phi + b2*Thin + b3*(phi*Thin) + e

We report b3 (interaction). Simple OLS with (X'X)^(-1). Option --demean-country applies within‑country demeaning
to (y,phi,Thin) before regression (absorbs country FE approximately).

Note: For robustness (Driscoll–Kraay / wild cluster) a specialized stats package is needed; placeholder simple SE implemented.
"""
from __future__ import annotations
import argparse, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from util_code01_lib_io import figure_name_with_code, safe_savefig, safe_to_csv

ROOT = Path(__file__).resolve().parents[1]
PANEL_DEFAULT = ROOT / 'data_processed' / 'proc_code09_phi_capital_interaction_panel.csv'

COLORS = {'Low':'#d55e00','Mid':'#0072b2','High':'#009e73'}  # colorblind friendly


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Date'])
    needed = {'phi','Thin','phi_x_Thin','credit_growth_ppYoY_lead','Thin_tercile','country'}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f'Panel missing columns: {missing}')
    df = df.dropna(subset=['phi','Thin','credit_growth_ppYoY_lead']).copy()
    return df


def regress(df: pd.DataFrame, demean_country: bool=False):
    work = df.copy()
    if demean_country:
        for col in ['credit_growth_ppYoY_lead','phi','Thin']:
            work[col] = work[col] - work.groupby('country')[col].transform('mean')
        work['phi_x_Thin'] = work['phi'] * work['Thin']
    y = work['credit_growth_ppYoY_lead'].values
    X = np.column_stack([
        np.ones(len(work)),
        work['phi'].values,
        work['Thin'].values,
        work['phi_x_Thin'].values
    ])
    # OLS
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n,k = X.shape
    sigma2 = (resid @ resid) / max(n-k,1)
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(xtx_inv) * sigma2)
    except Exception:
        se = np.full(k, np.nan)
    # t stats & p (normal approx)
    from math import erf, sqrt
    def pval(t):
        if np.isnan(t): return np.nan
        return 2*(1-0.5*(1+erf(abs(t)/sqrt(2))))
    t_stats = beta / se
    p_vals = [pval(t) for t in t_stats]
    cols = ['intercept','phi','Thin','phi_x_Thin']
    res_df = pd.DataFrame({'coef':beta,'se':se,'t':t_stats,'p':p_vals}, index=cols)
    return res_df, resid


def plot_strata(df: pd.DataFrame, out_png: Path, title: str, show_country_labels: bool=False):
    plt.figure(figsize=(6.6,5.0))
    ax = plt.gca()
    for terc in ['Low','Mid','High']:
        sub = df[df['Thin_tercile']==terc]
        if sub.empty: continue
        ax.scatter(sub['phi'], sub['credit_growth_ppYoY_lead'], label=f'{terc} CET1 tercile', alpha=0.75,
                   edgecolor='k', linewidths=0.3, s=38, color=COLORS.get(terc,'gray'))
        # Fit line within tercile
        if len(sub) >= 3:
            x = sub['phi'].values; y = sub['credit_growth_ppYoY_lead'].values
            X = np.column_stack([np.ones(len(sub)), x])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, b[0] + b[1]*xs, color=COLORS.get(terc,'gray'), linewidth=1.4)
    ax.set_xlabel('Off-balance share φ (current)')
    ax.set_ylabel('Next-quarter credit growth (pp YoY)')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    safe_savefig(plt.gcf(), out_png)
    print('saved:', out_png)


def main():
    ap = argparse.ArgumentParser(description='Interaction scatter phi x capital thinness – fig_code14')
    ap.add_argument('--panel', default=str(PANEL_DEFAULT))
    ap.add_argument('--out', default='SCATTER_phi_capital_interaction.png')
    ap.add_argument('--demean-country', action='store_true')
    ap.add_argument('--title', default='Credit growth (t+1) vs φ stratified by capital (CET1/RWA terciles)')
    args = ap.parse_args()
    panel = load_panel(Path(args.panel))
    # Regression
    reg_res, resid = regress(panel, demean_country=args.demean_country)
    # Save regression summary
    figs_dir = ROOT / 'figures'
    figs_dir.mkdir(parents=True, exist_ok=True)
    reg_csv = figure_name_with_code(__file__, figs_dir / (Path(args.out).stem + '_regression.csv'))
    safe_to_csv(reg_res.reset_index().rename(columns={'index':'term'}), reg_csv, index=False)
    print('saved:', reg_csv)
    # Plot stratified scatter
    out_png = figure_name_with_code(__file__, figs_dir / Path(args.out).name)
    plot_strata(panel, out_png, args.title)
    # Print interaction coef
    if 'phi_x_Thin' in reg_res.index:
        b3 = reg_res.loc['phi_x_Thin','coef']; se3 = reg_res.loc['phi_x_Thin','se']; p3 = reg_res.loc['phi_x_Thin','p']
        print(f"[INFO] interaction phi_x_Thin coef={b3:.4f} se={se3:.4f} p≈{p3:.3g}")

if __name__ == '__main__':  # pragma: no cover
    main()
