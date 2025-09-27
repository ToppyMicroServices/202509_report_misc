"""fetch_code08_fred_series

Generic FRED single‑series fetch utility.

Goals:
    - Decouple network retrieval logic from figure scripts (e.g. ``fig_code10``).
    - Provide a tiny API: ``fetch_fred_series(series_id, out_dir, do_download)``.

Features:
    - Downloads ``fredgraph.csv?id=SERIES_ID`` (auto-detect DATE & value column).
    - Normalizes index to quarter end (Period 'Q' → timestamp 'Q') without deprecated resample codes.
    - Saves CSV in ``out_dir`` (uses ``safe_to_csv`` to avoid clobbering existing files).
    - On failure returns ``(empty DataFrame, None)`` instead of raising.

Return shape:
    ``(DataFrame(index=QuarterEnd, columns=[series_id]), saved_csv_path|None)``

Notes:
    - Set ``do_download=False`` to skip network call (dry run scenarios).
    - Network exceptions are caught and reported via ``print`` to remain dependency‑light.
"""
from __future__ import annotations
import io, time
import urllib.request, urllib.error
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from util_code01_lib_io import safe_to_csv

def _to_quarter_end(idx):
    return pd.to_datetime(idx).to_period('Q').to_timestamp('Q')

def _fred_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

def fetch_fred_series(series_id: str, out_dir: Path | str = Path("data_raw"), *, timeout: int = 20, do_download: bool = True, retries: int = 3, backoff: float = 1.5) -> Tuple[pd.DataFrame, Optional[Path]]:
    """Download one FRED series and return a quarter‑end indexed DataFrame.

    Parameters
    ----------
    series_id : str
        FRED Series ID (e.g. ``GDP``, ``AGSEBMPTCMAHDFS``).
    out_dir : Path | str
        Destination directory (created if missing).
    timeout : int
        HTTP timeout seconds.
    do_download : bool
        If False, skip network call and return ``(empty, None)``.

    Returns
    -------
    (pd.DataFrame, Path|None)
        On success: (DataFrame, saved CSV path)
        On failure/skip: (empty DataFrame, None)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not do_download:
        return pd.DataFrame(), None
    url = _fred_url(series_id)
    attempt = 0
    headers = {"User-Agent":"Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    while True:
        attempt += 1
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            # Heuristic: if starts with '<' treat as HTML error page
            if raw[:1] == b"<":
                raise ValueError("HTML response (likely blocked or error)")
            # Try reading CSV; tolerate odd encodings/newlines
            df = pd.read_csv(io.BytesIO(raw))
            break
        except Exception as e:  # pragma: no cover
            if attempt <= retries:
                sleep_for = backoff ** (attempt - 1)
                print(f"[retry {attempt}/{retries}] FRED {series_id} failed: {e} -> sleep {sleep_for:.1f}s")
                time.sleep(sleep_for)
                continue
            print(f"[WARN] FRED download failed {series_id}: {e}")
            return pd.DataFrame(), None
    # Normalize column names: accept 'observation_date' or first column as date if reasonable
    cols_lower = {c.lower(): c for c in df.columns}
    if 'date' in cols_lower:
        dcol = cols_lower['date']
    elif 'observation_date' in cols_lower:
        dcol = cols_lower['observation_date']
    else:
        # fallback to first column name
        dcol = df.columns[0]
    if dcol not in df.columns:
        print(f"[ERROR] could not identify DATE column for {series_id}; cols={list(df.columns)}")
        return pd.DataFrame(), None
    if series_id not in df.columns:
        val_cols = [c for c in df.columns if c != dcol]
        if not val_cols:
            print(f"[ERROR] value column missing {series_id}")
            return pd.DataFrame(), None
        df = df.rename(columns={val_cols[0]: series_id})
    # Build a clean frame with canonical columns
    df = df[[dcol, series_id]].copy()
    df.rename(columns={dcol: 'DATE'}, inplace=True)
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
    df = df.dropna(subset=['DATE']).set_index('DATE').sort_index()
    # Normalize to quarter-end without resample('Q') to avoid deprecation warnings
    df.index = _to_quarter_end(df.index)
    csv_path = out_dir / f"{series_id}.csv"
    saved = safe_to_csv(df.reset_index(), csv_path, index=False)
    return df, saved

__all__ = ["fetch_fred_series"]

if __name__ == '__main__':  # minimal ad-hoc CLI
    import argparse, sys
    ap = argparse.ArgumentParser(description='Fetch single FRED series.')
    ap.add_argument('series_id')
    ap.add_argument('--out-dir', default='data_raw')
    ap.add_argument('--no-download', action='store_true')
    args = ap.parse_args()
    df, path = fetch_fred_series(args.series_id, args.out_dir, do_download=not args.no_download)
    if df.empty:
        print('[INFO] empty (skip or failure)')
        sys.exit(1)
    print(f"[INFO] rows={len(df)} saved={path}")
    sys.exit(0)
