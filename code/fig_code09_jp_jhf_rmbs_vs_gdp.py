"""fig_code09_jp_jhf_rmbs_vs_gdp

Build and plot the ratio (percent) of JHF RMBS outstanding to Japanese nominal GDP.

Process:
    1. Load JHF RMBS time series (DATE + value column) and Japanese GDP (DATE + value).
    2. Inner join on DATE and compute ratio_pct = (RMBS / GDP) * 100 using the first two value columns.
    3. Save derived ratio CSV (safe, no overwrite) and generate a simple time-series line plot.

Outputs:
        data_processed/fig_code09_JP_JHF_ratio_pct*.csv (fig_code09_ prefix enforced)
        figures/fig_code09_JP_housing_vs_RMBS_components_quarterly*.png

Input auto-detection (jhf series):
        Priority order unless --jhf explicitly provided:
            1. data_processed/proc_code06_JP_JHF_RMBS_vs_GDP_quarterly.csv (canonical)
            2. data_processed/JP_JHF_RMBS_vs_GDP.csv (legacy)
            3. data_processed/JP_JHF_RMBS_vs_GDP.xlsx (legacy; first two columns interpreted as DATE + value)
    GDP fallback search if --gdp missing path:
        - data_raw/JPNNGDP.csv (default)
    Script will SKIP with warning if either series cannot be loaded.
"""
from __future__ import annotations
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from util_code01_lib_io import safe_savefig, safe_to_csv, figure_name_with_code, find
import matplotlib.ticker as mtick

def _load_jhf(path: Path) -> pd.DataFrame:
    """Load JHF RMBS vs GDP input.

    Accepts CSV or XLSX. Normalizes to index DATE with one value column.
    If multiple numeric columns exist, uses the first.
    """
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == '.xlsx':
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    # DATE detection: prefer 'DATE'
    dcol = None
    for c in df.columns:
        if str(c).upper() == 'DATE':
            dcol = c; break
    if dcol is None:
        # assume first column is date
        dcol = df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol], errors='coerce')
    df = df.dropna(subset=[dcol]).set_index(dcol).sort_index()
    # pick first numeric column (exclude index)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return df[[c]].rename(columns={c: 'JHF_RMBS'})
    return pd.DataFrame()

def _load_gdp(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    dcol = None
    for c in df.columns:
        if str(c).upper() in ('DATE','OBSERVATION_DATE'):
            dcol = c; break
    if dcol is None:
        dcol = df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol], errors='coerce')
    df = df.dropna(subset=[dcol]).set_index(dcol).sort_index()
    # first numeric column
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return df[[c]].rename(columns={c: 'NGDP'})
    return pd.DataFrame()

def autodetect_inputs(jhf_arg: str, gdp_arg: str):
    """Return (jhf_path, gdp_path) with auto-fallbacks if provided defaults missing."""
    jhf_path = Path(jhf_arg)
    if not jhf_path.exists():
        # Prefer canonical proc_code06 output if present
        canon = Path('data_processed/proc_code06_JP_JHF_RMBS_vs_GDP_quarterly.csv')
        if canon.exists():
            jhf_path = canon
        else:
            alt = Path('data_processed/JP_JHF_RMBS_vs_GDP.xlsx')
            if alt.exists():
                # treat as candidate only if non-empty
                try:
                    tmp = pd.read_excel(alt, nrows=5)
                    if len(tmp) > 0:
                        jhf_path = alt
                except Exception:
                    pass
            if not jhf_path.exists():
                cand = find(['proc_code06_JP_JHF_RMBS_vs_GDP*.csv', 'JP_JHF_RMBS_vs_GDP*.csv', 'JP_JHF_RMBS_vs_GDP*.xlsx'])
                if cand:
                    # pick first with non-zero rows if possible
                    chosen=None
                    for c in cand:
                        try:
                            if c.suffix.lower()=='.xlsx':
                                t=pd.read_excel(c)
                            else:
                                t=pd.read_csv(c)
                            if len(t)>0:
                                chosen=c; break
                        except Exception:
                            continue
                    if chosen is None:
                        chosen=cand[0]
                    jhf_path=chosen
    gdp_path = Path(gdp_arg)
    if not gdp_path.exists():
        alt = Path('data_raw/JPNNGDP.csv')
        if alt.exists():
            gdp_path = alt
    return jhf_path, gdp_path

def load_inputs(jhf_path: Path, gdp_path: Path):
    jhf = _load_jhf(jhf_path)
    gdp = _load_gdp(gdp_path)
    if jhf.empty or gdp.empty:
        return pd.DataFrame(), pd.DataFrame()
    return jhf, gdp

def build_ratio(jhf: pd.DataFrame, gdp: pd.DataFrame):
    """Return DataFrame with ratio_pct using explicit column name conventions.

    Expects jhf to have 'JHF_RMBS' and gdp to have 'NGDP'.
    """
    # Align indices: convert both to quarter period (Q-DEC) and back to a common timestamp (use quarter start)
    jhf_q = jhf.copy()
    jhf_q.index = jhf_q.index.to_period('Q').to_timestamp()
    gdp_q = gdp.copy()
    gdp_q.index = gdp_q.index.to_period('Q').to_timestamp()
    df = jhf_q.join(gdp_q, how='inner')
    if 'JHF_RMBS' not in df.columns or 'NGDP' not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    out['ratio_pct'] = (df['JHF_RMBS'] / df['NGDP']) * 100
    return out

def fallback_build_from_components():
    """Attempt to construct a JHF RMBS vs GDP dataframe (DATE + value) from components file.

    Uses 'JP_RMBS_components_semiannual_JSDA_2012_2025.csv' plus GDP to synthesize a JHF-only stock
    if a 'jhf' column is identifiable. Returns DataFrame with index DATE and one column 'JHF_RMBS'.
    """
    comp_path = Path('data_processed/JP_RMBS_components_semiannual_JSDA_2012_2025.csv')
    if not comp_path.exists():
        return pd.DataFrame()
    try:
        comp = pd.read_csv(comp_path)
    except Exception:
        return pd.DataFrame()
    # date col
    dcol = None
    for c in comp.columns:
        if 'date' in c.lower():
            dcol = c; break
    if dcol is None:
        dcol = comp.columns[0]
    comp[dcol] = pd.to_datetime(comp[dcol], errors='coerce')
    comp = comp.dropna(subset=[dcol]).sort_values(dcol)
    jhf_cols = [c for c in comp.columns if 'jhf' in c.lower() and pd.api.types.is_numeric_dtype(comp[c])]
    if not jhf_cols:
        return pd.DataFrame()
    jhf = comp[[dcol, jhf_cols[0]]].rename(columns={jhf_cols[0]:'JHF_RMBS_raw'})
    jhf = jhf.dropna(subset=['JHF_RMBS_raw'])
    jhf = jhf.set_index(dcol).sort_index()
    # Heuristic unit normalisation: if large (>=1e6) assume hundred million yen
    if jhf['JHF_RMBS_raw'].max() > 1e6:
        jhf['JHF_RMBS'] = jhf['JHF_RMBS_raw'] * 0.1  # 100m yen -> bny
    else:
        jhf['JHF_RMBS'] = jhf['JHF_RMBS_raw']
    return jhf[['JHF_RMBS']]

def plot_ratio(df: pd.DataFrame, out_png: str, show: bool=False, overwrite: bool = True) -> Path:
    """Plot ratio_pct, save, optionally show; return saved Path.

    Styling tweaks:
      - Percent y-axis (1 decimal)
      - Title includes sample range
    """
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(
        df.index,
        df['ratio_pct'],
        label='JHF RMBS / GDP %',
        marker='o',
        markersize=4,
        linestyle='-',
        linewidth=1.3,
        markerfacecolor='white',
        markeredgecolor='#1f77b4'
    )
    if len(df.index):
        ax.set_title(f'JP JHF RMBS vs GDP Ratio ({df.index.min().date()} – {df.index.max().date()})')
    else:
        ax.set_title('JP JHF RMBS vs GDP Ratio')
    ax.set_ylabel('% of NGDP')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    saved = safe_savefig(fig, out_png, overwrite=overwrite)
    if show:
        try:
            plt.show()
        except Exception:
            pass
    return saved

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--jhf', default='data_processed/proc_code06_JP_JHF_RMBS_vs_GDP_quarterly.csv', help='JHF RMBS vs GDP CSV (proc_code06; legacy CSV/XLSX auto-fallback)')
    ap.add_argument('--gdp', default='data_raw/JPNNGDP.csv', help='JP nominal GDP CSV (DATE + value)')
    ap.add_argument('--out-csv', default='data_processed/JP_JHF_RMBS_to_GDP_ratio_quarterly.csv', help='Base output CSV (prefix added automatically)')
    ap.add_argument('--out-png', default='figures/JP_JHF_RMBS_to_GDP_ratio.png', help='Base output PNG (prefix added)')
    ap.add_argument('--annual-summary', action='store_true', help='Also write annual summary CSV (year-end stock & annual avg ratio)')
    ap.add_argument('--show', action='store_true', help='Display figure window after saving')
    ap.set_defaults(overwrite=True)
    ap.add_argument('--overwrite', action='store_true', dest='overwrite', help='Overwrite outputs (default)')
    ap.add_argument('--keep-versions', action='store_false', dest='overwrite', help='Append numeric suffix instead of overwriting')
    ap.add_argument('--prefer-proc-code06', action='store_true', help='Force using proc_code06 quarterly file if present (default already prefers it)')
    args = ap.parse_args(argv)
    # Auto fast-path: prefer proc_code06 quarterly when available
    proc06_path = Path('data_processed/proc_code06_JP_JHF_RMBS_vs_GDP_quarterly.csv')
    used_proc06 = False
    if (args.prefer_proc_code06 or args.jhf == str(proc06_path)) and proc06_path.exists():
        try:
            df06 = pd.read_csv(proc06_path)
            # Expect columns DATE, JHF_RMBS_bny, NGDP_bny, ratio_pct
            if 'DATE' in df06.columns and 'ratio_pct' in df06.columns:
                df06['DATE'] = pd.to_datetime(df06['DATE'], errors='coerce')
                df06 = df06.dropna(subset=['DATE']).set_index('DATE').sort_index()
                ratio = df06[['ratio_pct']].copy()
                used_proc06 = True
        except Exception:
            used_proc06 = False
    if not used_proc06:
        jhf_path, gdp_path = autodetect_inputs(args.jhf, args.gdp)
        constructed = False
        if not jhf_path.exists():
            # try component-based fallback
            jhf_fb = fallback_build_from_components()
            if jhf_fb.empty:
                print(f"[SKIP] JHF input missing (tried {args.jhf} / .xlsx and component fallback)")
                return 0
            else:
                jhf = jhf_fb; constructed = True
        if not gdp_path.exists():
            print(f"[SKIP] GDP input missing (expected {gdp_path})")
            return 0
    if not used_proc06:
        if not constructed:
            jhf, gdp = load_inputs(jhf_path, gdp_path)
            if jhf.empty or gdp.empty:
                print('[SKIP] failed to load one or both inputs (empty after parsing)')
                return 0
        else:
            gdp = _load_gdp(gdp_path)
            if gdp.empty:
                print('[SKIP] GDP load failed for constructed JHF series')
                return 0
        ratio = build_ratio(jhf, gdp)
    if ratio.empty:
        print('[WARN] empty ratio')
        return 1
    out_df = ratio.reset_index().rename(columns={'index':'DATE'})
    # Prefix CSV filename with this script code for provenance consistency
    csv_target = figure_name_with_code(__file__, Path(args.out_csv))
    saved_csv = safe_to_csv(out_df, csv_target, index=False, overwrite=args.overwrite)
    if args.overwrite:
        print(f"[INFO] CSV written (overwrite): {saved_csv}")
    else:
        print(f"[INFO] CSV saved (versioned): {saved_csv}")
    fig_path = plot_ratio(
        ratio,
        str(figure_name_with_code(__file__, Path(args.out_png))),
        show=args.show,
        overwrite=args.overwrite,
    )
    print(f"[INFO] Figure saved: {fig_path}")
    if args.annual_summary:
        # Year-end & annual average ratio
        ann_end = ratio.resample('YE-DEC').last()
        ann_avg = ratio.resample('YE-DEC').mean().rename(columns={'ratio_pct':'ratio_pct_avg'})
        annual = ann_end.join(ann_avg)
        annual['ratio_pct_year_end'] = ann_end['ratio_pct']
        annual_path = figure_name_with_code(__file__, Path('data_processed/JP_JHF_RMBS_to_GDP_ratio_annual_summary.csv'))
        safe_to_csv(
            annual.reset_index().rename(columns={'index':'DATE'}),
            annual_path,
            index=False,
            overwrite=args.overwrite,
        )
        print(f"[INFO] annual summary saved: {annual_path}")
    print('[OK] wrote outputs')
    return 0

if __name__ == '__main__':  # pragma: no cover
    import sys
    sys.exit(main())
