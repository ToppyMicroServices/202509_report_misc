"""fetch_code09_legacy_tmp_extracts

Legacy/deprecated fetch helpers (extracted from ``util_code99_tmp``).

Use cases:
    Only for reproducing historical exploratory notebooks. Not part of the
    maintained pipeline. May be removed without notice.

Functions:
    fetch_us_mbs(status):  US agency & GSE MBS outstanding (adds US_MBS_$Bn)
    fetch_us_gdp(status):  US nominal GDP SAAR (adds US_GDP_$Bn quarterly level)
    fetch_esri_jp_gdp(status, skip=False): JP nominal GDP from fixed ESRI CSV

Each function updates the provided ``status`` dict with (STATE, INFO) where
STATE ∈ {OK, ERROR, SKIP}. Returns a quarterly (freq Q) DataFrame or empty.

Prefer newer modules: ``fetch_code05_fred_us_mbs_gdp``, ``fetch_code06_esri_jp_gdp`` (manual),
``fetch_code07_imf_jp_gdp`` (primary JP GDP source).
"""
from __future__ import annotations
import pandas as pd, warnings, io, urllib.request
from typing import Dict

FREQ_Q_END = 'Q'

def _safe_read_fred(url: str, id_col: str) -> pd.DataFrame:
    """Fetch a simple FRED CSV (DATE + value) and normalize it.

    Returns DataFrame with columns DATE & id_col or empty on failure.
    Warnings emitted instead of exceptions to keep legacy tolerance.
    """
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            raw = r.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:  # pragma: no cover
        warnings.warn(f"FRED fetch failed {url}: {e}")
        return pd.DataFrame()
    if 'DATE' not in df.columns:
        alt = [c for c in df.columns if 'DATE' in c.upper()]
        if alt:
            df = df.rename(columns={alt[0]: 'DATE'})
        else:
            warnings.warn(f"DATE column not found (cols={df.columns.tolist()})")
            return pd.DataFrame()
    if id_col not in df.columns:
        data_cols = [c for c in df.columns if c != 'DATE']
        if not data_cols:
            warnings.warn(f"No value column: {url}")
            return pd.DataFrame()
        df = df.rename(columns={data_cols[0]: id_col})
    try:
        df['DATE'] = pd.to_datetime(df['DATE'])
    except Exception as e:
        warnings.warn(f"DATE parse failed: {e}")
        return pd.DataFrame()
    return df

def fetch_us_mbs(status: Dict) -> pd.DataFrame:
    """US agency & GSE MBS outstanding (adds US_MBS_$Bn)."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=AGSEBMPTCMAHDFS"
    df = _safe_read_fred(url, "US_MBS_$Mil")
    if df.empty:
        status["us_mbs"] = ("ERROR", "FRED MBS fetch failed/empty")
        return pd.DataFrame()
    df["US_MBS_$Bn"] = pd.to_numeric(df["US_MBS_$Mil"], errors='coerce') / 1_000.0
    status["us_mbs"] = ("OK", f"rows={len(df)}")
    return df.set_index("DATE").asfreq(FREQ_Q_END)

def fetch_us_gdp(status: Dict) -> pd.DataFrame:
    """US nominal GDP SAAR (converted to quarterly level via /4)."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP"
    df = _safe_read_fred(url, "US_GDP_SAAR_$Bn")
    if df.empty:
        status["us_gdp"] = ("ERROR", "FRED GDP fetch failed/empty")
        return pd.DataFrame()
    df["US_GDP_$Bn"] = pd.to_numeric(df["US_GDP_SAAR_$Bn"], errors='coerce') / 4.0
    status["us_gdp"] = ("OK", f"rows={len(df)}")
    return df.set_index("DATE").asfreq(FREQ_Q_END)

def fetch_esri_jp_gdp(status: Dict, skip: bool=False) -> pd.DataFrame:
    """Japan nominal GDP via legacy fixed ESRI CSV (pattern row extraction).

    skip=True returns empty and records a SKIP status. Prefer IMF source.
    """
    if skip:
        status["esri_gdp"] = ("SKIP", "--no-remote-esri specified")
        return pd.DataFrame()
    url = "https://www.esri.cao.go.jp/jp/sna/data/sokuhou/files/2025/qe252_2/tables/gaku-mk2522.csv"
    try:
        raw = pd.read_csv(url, header=None)
    except Exception as e:
        status["esri_gdp"] = ("ERROR", f"fetch failed {e}")
        return pd.DataFrame()
    row_idx = raw[raw.astype(str).apply(lambda r: r.str.contains("Gross domestic product", case=False, na=False)).any(axis=1)].index
    if not len(row_idx):
        status["esri_gdp"] = ("ERROR", "GDP row not detected")
        return pd.DataFrame()
    row = row_idx[0]
    series = raw.iloc[row].reset_index(drop=True)
    q_cols = [c for c in range(len(series)) if isinstance(series[c], str) and "Q" in series[c]]
    dates = pd.PeriodIndex(series[q_cols], freq=FREQ_Q_END).to_timestamp(FREQ_Q_END)
    values = raw.iloc[row+1, q_cols].astype(float)
    df = pd.DataFrame({"DATE": dates, "JP_GDP": values}).set_index("DATE").asfreq(FREQ_Q_END)
    status["esri_gdp"] = ("OK", f"rows={len(df)}")
    return df

__all__ = [
    'fetch_us_mbs','fetch_us_gdp','fetch_esri_jp_gdp'
]
