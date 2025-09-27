"""fetch_code05_fred_us_mbs_gdp

Fetch helper functions for key US macro / housing finance series from FRED.

Baseline (legacy behaviour):
    * AGSEBMPTCMAHDFS  (Agency & GSE MBS outstanding, $Mil)
    * GDP (SAAR)       (used historically; converted to level by /4)

Enhancement (2025-09): also download and persist the exact raw series files
needed by downstream ratio builders and figure scripts:
    * NGDPSAXDCUSQ (Nominal GDP, quarterly, not SAAR)  -> data_raw/NGDPSAXDCUSQ.csv
    * AGSEBMPTCMAHDFS (raw copy)                       -> data_raw/AGSEBMPTCMAHDFS.csv
    * MORTGAGE30US (30-year fixed mortgage rate)       -> data_raw/MORTGAGE30US.csv (context / not used in ratio)
    * BOGZ1FL673065105Q (Home mortgages liability Lvl) -> data_raw/BOGZ1FL673065105Q.csv

For each series we attempt an HTTP fetch; on failure we emit a MANUAL instruction
with an URL you can visit and save the CSV into data_raw/ using the indicated
filename so that later processing scripts (e.g. proc_code05) can operate offline.
"""
from __future__ import annotations
import pandas as pd
import urllib.request, io, warnings, argparse, logging
from pathlib import Path
from typing import Dict

FREQ_Q_END = 'QE'  # quarter-end label (avoid deprecated plain 'Q' in asfreq)

def _to_quarter_end(idx):
    # Use Period('Q') then to_timestamp which is stable, independent of 'QE' label usage.
    return pd.to_datetime(idx).to_period('Q').to_timestamp('Q')

def _safe_read_fred(url: str, id_col: str, *, timeout: int = 15) -> pd.DataFrame:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            raw = r.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        warnings.warn(f"FRED fetch failed {url}: {e}")
        print(f"[MANUAL] FREDデータ取得に失敗しました。以下URLをブラウザで開き、CSVをダウンロードして data_raw/ に保存してください:\n  {url}\n  ファイル名例: AGSEBMPTCMAHDFS.csv, GDP.csv")
        return pd.DataFrame()
    if 'DATE' not in df.columns:
        alt = [c for c in df.columns if 'DATE' in c.upper()]
        if alt:
            df = df.rename(columns={alt[0]: 'DATE'})
        else:
            warnings.warn(f"DATE column missing for {url}")
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
        warnings.warn(f"DATE parse fail: {e}")
        return pd.DataFrame()
    return df

def _fetch_and_save_raw(series_id: str, status: Dict, raw_dir: Path, timeout: int = 15):
    """Fetch a FRED series and save raw two-column CSV (DATE, value) into raw_dir.

    Writes file named <series_id>.csv. On failure prints manual instructions.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = _safe_read_fred(url, series_id, timeout=timeout)
    out_path = raw_dir/f"{series_id}.csv"
    if df.empty:
        print(f"[MANUAL] Download {url} and save as {out_path}")
        status[series_id.lower()] = ("ERROR", "empty")
        return
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        df[['DATE', series_id]].to_csv(out_path, index=False)
        status[series_id.lower()] = ("OK", f"saved {out_path.name} rows={len(df)}")
    except Exception as e:
        status[series_id.lower()] = ("ERROR", f"save fail: {e}")
        print(f"[ERROR] could not save {out_path}: {e}")

logger = logging.getLogger(__name__)

def fetch_us_mbs(status: Dict) -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=AGSEBMPTCMAHDFS"
    df = _safe_read_fred(url, "US_MBS_$Mil")
    if df.empty:
        status["us_mbs"] = ("ERROR", "empty")
        return pd.DataFrame()
    df["US_MBS_$Bn"] = pd.to_numeric(df["US_MBS_$Mil"], errors='coerce') / 1_000.0
    status["us_mbs"] = ("OK", f"rows={len(df)}")
    return df.set_index("DATE").pipe(lambda d: d.set_index(_to_quarter_end(d.index)))

def fetch_us_gdp(status: Dict) -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP"
    df = _safe_read_fred(url, "US_GDP_SAAR_$Bn")
    if df.empty:
        status["us_gdp"] = ("ERROR", "empty")
        return pd.DataFrame()
    df["US_GDP_$Bn"] = pd.to_numeric(df["US_GDP_SAAR_$Bn"], errors='coerce') / 4.0
    status["us_gdp"] = ("OK", f"rows={len(df)}")
    return df.set_index("DATE").pipe(lambda d: d.set_index(_to_quarter_end(d.index)))

def _ensure_prefixed_default(out_path: str) -> str:
    """If caller did not override --out (uses default) ensure fetch_code05_ prefix.

    Backward compatibility: if user explicitly supplies an *unprefixed* path we
    respect it. Only transform when the value equals the historical default or
    resides in data_processed with the legacy basename.
    """
    legacy = 'data_processed/US_MBS_GDP_quarterly.csv'
    pref  = 'data_processed/fetch_code05_US_MBS_GDP_quarterly.csv'
    if out_path == legacy:
        return pref
    return out_path

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data_processed/fetch_code05_US_MBS_GDP_quarterly.csv')
    ap.add_argument('--raw-dir', default='data_raw', help='Directory to write individual raw series CSVs.')
    ap.add_argument('--no-extra', action='store_true', help='Skip downloading auxiliary raw series (NGDPSAXDCUSQ, MORTGAGE30US).')
    ap.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(levelname)s %(message)s')
    status: Dict[str, tuple[str,str]] = {}
    raw_dir = Path(args.raw_dir)

    # Always fetch and persist the canonical outstanding MBS series & nominal GDP (non-SAAR) for downstream scripts.
    _fetch_and_save_raw('AGSEBMPTCMAHDFS', status, raw_dir)
    _fetch_and_save_raw('NGDPSAXDCUSQ', status, raw_dir)
    _fetch_and_save_raw('BOGZ1FL673065105Q', status, raw_dir)
    if not args.no_extra:
        _fetch_and_save_raw('MORTGAGE30US', status, raw_dir)
    mbs = fetch_us_mbs(status)
    gdp = fetch_us_gdp(status)
    if mbs.empty or gdp.empty:
        logger.error('one or both series empty (mbs=%s gdp=%s)', mbs.empty, gdp.empty)
        return 1
    merged = mbs.join(gdp, how='outer')
    out = Path(_ensure_prefixed_default(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out)
    logger.info('wrote %s rows=%d', out, len(merged))
    # Summary status line for automation / logs
    try:
        summary = ' '.join([f"{k}={v[0]}" for k,v in status.items()])
        logger.info('status %s', summary)
    except Exception:  # pragma: no cover
        pass
    return 0

if __name__ == '__main__':  # pragma: no cover
    import sys
    sys.exit(main())
