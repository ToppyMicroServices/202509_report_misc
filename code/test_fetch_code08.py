"""Tests for fetch_code08_fred_series.fetch_fred_series

Networkはモックし、以下を検証:
 1) 正常系: CSV -> QuarterEnd 変換, 保存ファイル生成
 2) スキップ系: do_download=False で空/None
 3) 失敗系: urlopen 例外で空/None
"""
import sys, types
from pathlib import Path
import pandas as pd
from unittest.mock import patch

import pytest

sys.path.append('code')
from fetch_code08_fred_series import fetch_fred_series  # noqa


class DummyResponse:
    def __init__(self, data: bytes):
        self._data = data
    def read(self):
        return self._data
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def sample_csv_bytes():
    # 1月/4月 → Q1/Q2 の QuarterEnd へ変換される想定
    return b"DATE,TESTSER\n2024-01-01,1\n2024-04-01,2\n"


def test_fetch_normal(tmp_path: Path, sample_csv_bytes):
    def _fake_urlopen(url, timeout=0):
        return DummyResponse(sample_csv_bytes)
    with patch('urllib.request.urlopen', _fake_urlopen):
        df, saved = fetch_fred_series('TESTSER', out_dir=tmp_path, do_download=True)
    assert not df.empty
    assert list(df.columns) == ['TESTSER']
    # Quarter end に正規化され 3月末 / 6月末
    assert df.index.tolist() == [pd.Timestamp('2024-03-31'), pd.Timestamp('2024-06-30')]
    assert saved is not None and saved.exists()
    assert saved.name.startswith('TESTSER') and saved.suffix == '.csv'


def test_fetch_skip(tmp_path: Path):
    df, saved = fetch_fred_series('TESTSER', out_dir=tmp_path, do_download=False)
    assert df.empty
    assert saved is None


def test_fetch_failure(tmp_path: Path):
    def _raise(*a, **k):
        raise OSError('network down')
    with patch('urllib.request.urlopen', _raise):
        df, saved = fetch_fred_series('FAILSER', out_dir=tmp_path, do_download=True)
    assert df.empty
    assert saved is None
