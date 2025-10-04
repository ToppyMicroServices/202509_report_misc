"""fig_code11_jp_jhf_rmbs_vs_gdp_experimental

Experimental end-to-end builder using modularized fetch & util code.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
from fetch_code05_fred_us_mbs_gdp import fetch_us_mbs, fetch_us_gdp
from fetch_code06_esri_jp_gdp import fetch_esri_jp_gdp  # legacy optional (manual URL recommended)
from util_code04_jhf_excel_loader import load_jhf_mbs_excel
from util_code05_ratio_builders import load_local_jp_gdp, build_jp_ratio, plot_ratio
import warnings
warnings.filterwarnings('ignore', message="'Q' is deprecated", category=FutureWarning)
from util_code01_lib_io import safe_to_csv, figure_name_with_code

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gdp-csv', default='data_raw/JPNNGDP.csv')
    ap.add_argument('--jhf-excel', default=None)
    ap.add_argument('--esri', action='store_true', help='Attempt ESRI GDP fetch (adds comparison)')
    ap.add_argument('--unit', default='okuen', choices=['okuen','billion_yen'])
    # Ensure outputs go to standard folders with fig_code11_ prefix
    ap.add_argument('--out-csv', default='data_processed/fig_code11_JP_JHF_RMBS_vs_GDP_experimental.csv')
    ap.add_argument('--out-png', default='figures/fig_code11_JP_JHF_RMBS_vs_GDP_experimental.png')
    args = ap.parse_args()
    status = {}
    # GDP
    try:
        jp_gdp_q = load_local_jp_gdp(args.gdp_csv)
        status['jp_gdp'] = ('OK', f'rows={len(jp_gdp_q)}')
    except Exception as e:
        print('[ERROR] JP GDP load failed', e)
        jp_gdp_q = None
        status['jp_gdp'] = ('ERROR', str(e))
    # JHF
    jhf_q = None
    if args.jhf_excel:
        try:
            jhf_q = load_jhf_mbs_excel(args.jhf_excel)
            status['jhf'] = ('OK', f'rows={len(jhf_q)}')
        except Exception as e:
            status['jhf'] = ('ERROR', str(e))
    # Build
    if jp_gdp_q is not None and jhf_q is not None and not jp_gdp_q.empty and not jhf_q.empty:
        result = build_jp_ratio(jp_gdp_q, jhf_q, jhf_unit=args.unit)
        if not result.empty:
            result.index.name = 'Date'
            from util_code01_lib_io import figure_name_with_code as _fn_code
            out_csv_path = _fn_code(__file__, Path(args.out_csv))
            out_csv = safe_to_csv(result.reset_index(), out_csv_path, index=False, overwrite=True)
            print('[OK] wrote', out_csv)
            if args.out_png:
                from pathlib import Path
                out_png_path = figure_name_with_code(__file__, Path(args.out_png))
                plot_ratio(result, str(out_png_path), overwrite=True)
        else:
            print('[WARN] empty join result')
    else:
        print('[WARN] missing inputs for ratio (need both jp_gdp and jhf)')
        if jp_gdp_q is None or (jp_gdp_q is not None and jp_gdp_q.empty):
            print('  - JP GDP missing/empty (check --gdp-csv path)')
        if jhf_q is None:
            print('  - JHF input missing (provide --jhf-excel)')
        elif jhf_q.empty:
            print('  - JHF parsed dataframe empty (Excel parse issue)')
    # Optional ESRI fetch
    if args.esri:
        _ = fetch_esri_jp_gdp(status, skip=False)
    # US reference quick fetch (no saving)
    _ = fetch_us_mbs(status); _ = fetch_us_gdp(status)
    print('STATUS:')
    for k,v in status.items():
        print(' ', k, '->', v)
    return 0

if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
