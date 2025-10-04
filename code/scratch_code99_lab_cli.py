"""scratch_code99_lab_cli

Minimal lab / exploratory CLI wrapper (former util_code99_tmp).

Purpose:
  - Provide a very small command-line utility to: (a) load local JP GDP, (b) load JHF Excel, (c) build ratio, (d) optional plot.
  - All heavy fetch / parsing logic now lives in dedicated fetch_code / util_code modules.

DEPRECATED: This file is for ad-hoc experimentation only and may be removed after 2025Q4.

Example:
  python code/scratch_code99_lab_cli.py \
      --gdp-csv data_processed/JPNNGDP.csv \
      --jhf-excel data_raw/JHF_RemainingBalance.xlsx \
      --out-csv data_processed/JP_JHF_RMBS_vs_GDP_lab.csv \
      --out-png figures/JP_JHF_RMBS_vs_GDP_lab.png
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np
from util_code05_ratio_builders import load_local_jp_gdp, build_jp_ratio, plot_ratio
from util_code04_jhf_excel_loader import load_jhf_mbs_excel
from util_code01_lib_io import safe_to_csv


def _load_jhf_or_dummy(jhf_excel: str|None, jp_gdp_q: pd.DataFrame, allow_dummy: bool) -> pd.DataFrame:
    if jhf_excel and Path(jhf_excel).exists():
        return load_jhf_mbs_excel(jhf_excel)
    if allow_dummy and not jp_gdp_q.empty:
        rng = jp_gdp_q.index
        dummy_ratio = np.linspace(0.05, 0.12, len(rng))
        vals = jp_gdp_q['JP_GDP'].values * dummy_ratio
        return pd.DataFrame({'JP_JHF_RMBS': vals}, index=rng)
    return pd.DataFrame()


def main(argv=None):
    ap = argparse.ArgumentParser(description='Minimal lab ratio builder (deprecated)')
    ap.add_argument('--gdp-csv', required=True, help='JP NGDP CSV (IMF processed etc.)')
    ap.add_argument('--jhf-excel', default=None, help='JHF monthly Excel (optional)')
    ap.add_argument('--jhf-unit', default='okuen', choices=['okuen','billion_yen'])
    ap.add_argument('--allow-dummy', action='store_true', help='Allow synthetic JHF series')
    ap.add_argument('--out-csv', default='data_processed/JP_JHF_RMBS_vs_GDP_lab.csv')
    ap.add_argument('--out-png', default=None)
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args(argv)

    try:
        jp_gdp_q = load_local_jp_gdp(args.gdp_csv)
    except Exception as e:
        print('[FATAL] JP GDP load failed:', e)
        return 1

    jhf_q = _load_jhf_or_dummy(args.jhf_excel, jp_gdp_q, args.allow_dummy)
    if jhf_q.empty:
        print('[WARN] JHF series empty; aborting.')
        return 1

    final_df = build_jp_ratio(jp_gdp_q, jhf_q, jhf_unit=args.jhf_unit)
    if final_df.empty:
        print('[WARN] Ratio result empty')
        return 1

    out = final_df.copy()
    out.index.name = 'Date'
    out = out.rename(columns={'JP_JHF_RMBS_Bn':'JP_RMBS_Bn'})
    saved = safe_to_csv(out, args.out_csv, index=False, overwrite=True)
    print(f'[INFO] CSV saved: {saved} rows={len(out)}')

    if not args.no_plot and args.out_png:
        plot_ratio(final_df, out_png=args.out_png)
        print(f'[INFO] PNG saved: {args.out_png}')

    print('[DONE] scratch CLI complete (deprecated)')
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
