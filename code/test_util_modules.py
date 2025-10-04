"""Basic tests for util_ modules.

These are lightweight and avoid network. Run with:
  python -m pytest -q code/test_util_modules.py

They validate:
  - ensure_unique generates new path
  - safe_to_csv writes and does not overwrite
  - read_fred_two_col handles simple synthetic CSV
  - ratio builder produces expected columns and math
"""
from pathlib import Path
import pandas as pd
import numpy as np
from util_code01_lib_io import ensure_unique, safe_to_csv, read_fred_two_col
from util_code05_ratio_builders import build_jp_ratio


def test_ensure_unique(tmp_path: Path):
    p = tmp_path / 'file.csv'
    p.write_text('a')
    p1 = ensure_unique(p)
    assert p1 != p
    p1.write_text('b')
    p2 = ensure_unique(p)
    assert p2 not in (p, p1)


def test_safe_to_csv_no_overwrite(tmp_path: Path):
    df = pd.DataFrame({'A':[1,2]})
    p1 = safe_to_csv(df, tmp_path / 'data.csv', index=False)
    p2 = safe_to_csv(df, tmp_path / 'data.csv', index=False)
    assert p1.name == 'data.csv'
    assert p2.name.startswith('data_')
    assert p1 != p2


def test_read_fred_two_col(tmp_path: Path):
    csv = tmp_path / 'SERIESX.csv'
    csv.write_text('DATE,VAL\n2024-01-01,1\n2024-02-01,2\n')
    df = read_fred_two_col(csv)
    assert df.index.name is not None
    assert 'SERIESX' in df.columns
    assert df.iloc[0,0] == 1


def test_build_jp_ratio_math():
    # GDP 100, 200 ; JHF 10, 30 (億円) conv 0.1 -> 1,3 Bn
    dates = pd.to_datetime(['2024-03-31','2024-06-30'])
    gdp = pd.DataFrame({'JP_GDP':[1000.0, 2000.0]}, index=dates)  # already in billion? treat as value
    jhf = pd.DataFrame({'JP_JHF_RMBS':[10.0, 30.0]}, index=dates)  # 億円
    out = build_jp_ratio(gdp, jhf, jhf_unit='okuen')
    assert list(out.columns) == ['JP_JHF_RMBS_Bn','NGDP_Bn','MBS_to_GDP_pct']
    # 10億円=1Bn vs GDP 1000 -> 0.1% ; 30億円=3Bn vs 2000 -> 0.15%
    assert np.isclose(out.iloc[0].MBS_to_GDP_pct, 0.1, atol=1e-6)
    assert np.isclose(out.iloc[1].MBS_to_GDP_pct, 0.15, atol=1e-6)
