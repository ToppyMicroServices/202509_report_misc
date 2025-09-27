"""fetch_code07_imf_jp_gdp

Fetch Japan nominal quarterly GDP (current prices) from IMF IFS SDMX JSON API.
Default indicator: NGDP (Gross Domestic Product, current prices, national currency) – quarterly.

Example:
  python fetch_code07_imf_jp_gdp.py --start 2010 --end 2025 --out data_processed/JP_IMF_GDP_quarterly.csv

If network blocked or you prefer a preview, use --dry-run to see the constructed URL.

References:
  IMF SDMX JSON: https://dataservices.imf.org/REST/SDMX_JSON.svc/
  CompactData pattern: /CompactData/{DB}/{FREQ}.{REF_AREA}.{INDICATOR}?startPeriod=YYYY&endPeriod=YYYY
"""
from __future__ import annotations
import argparse, logging, sys, datetime, json
from pathlib import Path
from typing import List, Dict, Any
import re

try:
    import requests  # type: ignore
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    print("[FATAL] Missing dependencies: requests, pandas", e)
    raise

BASE = "https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS"

logger = logging.getLogger(__name__)

QUARTER_RE = re.compile(r"^(\d{4})(?:-?Q([1-4]))$")


def build_url(freq: str, country: str, indicator: str, start: str, end: str) -> str:
    key = f"{freq}.{country}.{indicator}"
    return f"{BASE}/{key}?startPeriod={start}&endPeriod={end}"


def _parse_obs_list(series: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Series may contain 'Obs' list; each has 'TimePeriod' & 'ObsValue' (or '@OBS_VALUE') depending on IMF version.
    obs = series.get('Obs') or []
    out: List[Dict[str, Any]] = []
    for o in obs:
        v = None
        if isinstance(o, dict):
            v = o.get('ObsValue') or o.get('@OBS_VALUE')
            if isinstance(v, dict):
                v = v.get('@value')
            tp = o.get('TimePeriod') or o.get('@TIME_PERIOD')
            if tp is None:
                continue
            out.append({'time': tp, 'value': v})
    return out


def _to_quarter_end(ts: str):
    m = QUARTER_RE.match(ts.strip())
    if not m:
        return None
    year = int(m.group(1))
    q = int(m.group(2))
    month = q * 3
    # Use last day of quarter month
    if month in (3, 6, 9, 12):
        # Determine last day quickly: next month first day minus one.
        if month == 12:
            end_day = 31
        elif month == 9:
            end_day = 30
        elif month == 6:
            end_day = 30
        else:
            end_day = 31
        return datetime.date(year, month, end_day)
    return None


def _make_session(retries: int, backoff: float):
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry  # type: ignore
    except Exception:  # pragma: no cover
        Retry = None  # type: ignore
    sess = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Accept": "application/json",
        # IMF endpoint can reset pooled keep-alive connections; force fresh connections per request
        "Connection": "close",
    }
    sess.headers.update(headers)
    if Retry is not None:
        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=backoff,
            status_forcelist=(408, 429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        # Keep pool small to reduce chance of stale keep-alive reuse
        adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=2)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
    return sess


def fetch_imf_gdp(freq: str, country: str, indicator: str, start: str, end: str, *, timeout: int = 30, retries: int = 3, backoff: float = 1.5, raw_out: Path | None = None) -> pd.DataFrame:
    url = build_url(freq, country, indicator, start, end)
    logger.info("IMF URL: %s", url)
    sess = _make_session(retries, backoff)
    r = sess.get(url, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    # Navigate JSON (CompactData structure)
    data = js.get('CompactData', {}).get('DataSet', {}).get('Series')
    if data is None:
        raise ValueError('Series not found in JSON response')
    # Series may be list or dict
    series_list: List[Dict[str, Any]]
    if isinstance(data, list):
        series_list = data
    else:
        series_list = [data]
    if not series_list:
        raise ValueError('Empty series list')
    obs_all: List[Dict[str, Any]] = []
    for s in series_list:
        obs_all.extend(_parse_obs_list(s))
    if not obs_all:
        raise ValueError('No observations parsed')
    rows = []
    for o in obs_all:
        tp = str(o['time'])
        val = o['value']
        if val in (None, '', 'NA'):
            continue
        dt = _to_quarter_end(tp)
        if not dt:
            continue
        try:
            fv = float(val)
        except ValueError:
            continue
        rows.append((dt, fv))
    if not rows:
        raise ValueError('No valid quarterly rows')
    df = pd.DataFrame(rows, columns=['DATE', f'{country}_{indicator}_NC'])
    df = df.sort_values('DATE').reset_index(drop=True)
    return df


def fetch_imf_gdp_chunked(freq: str, country: str, indicator: str, start: str, end: str, *, timeout: int = 90, retries: int = 6, backoff: float = 2.0, chunk_years: int = 8) -> pd.DataFrame:
    """Fetch IMF data by splitting the time range into smaller year windows.

    This reduces payload and helps avoid server-side connection resets.
    """
    import pandas as pd
    s_year, e_year = int(start), int(end)
    if s_year > e_year:
        raise ValueError("start must be <= end")
    sess = _make_session(retries, backoff)
    frames = []
    y = s_year
    while y <= e_year:
        w_start = y
        w_end = min(y + max(1, chunk_years) - 1, e_year)
        url = build_url(freq, country, indicator, str(w_start), str(w_end))
        logger.info("IMF URL (chunk %d-%d): %s", w_start, w_end, url)
        r = sess.get(url, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        dfw = _df_from_compact_json(js, country, indicator)
        frames.append(dfw)
        y = w_end + 1
    if not frames:
        raise ValueError('No data frames fetched')
    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df.drop_duplicates(subset=['DATE']).sort_values('DATE').reset_index(drop=True)
    return df


def _df_from_compact_json(js: Dict[str, Any], country: str, indicator: str):
    data = js.get('CompactData', {}).get('DataSet', {}).get('Series')
    if data is None:
        raise ValueError('Series not found in JSON response')
    series_list: List[Dict[str, Any]]
    if isinstance(data, list):
        series_list = data
    else:
        series_list = [data]
    if not series_list:
        raise ValueError('Empty series list')
    obs_all: List[Dict[str, Any]] = []
    for s in series_list:
        obs_all.extend(_parse_obs_list(s))
    if not obs_all:
        raise ValueError('No observations parsed')
    rows = []
    for o in obs_all:
        tp = str(o['time'])
        val = o['value']
        if val in (None, '', 'NA'):
            continue
        dt = _to_quarter_end(tp)
        if not dt:
            continue
        try:
            fv = float(val)
        except ValueError:
            continue
        rows.append((dt, fv))
    if not rows:
        raise ValueError('No valid quarterly rows')
    import pandas as pd  # local import to keep type hints above
    df = pd.DataFrame(rows, columns=['DATE', f'{country}_{indicator}_NC'])
    df = df.sort_values('DATE').reset_index(drop=True)
    return df


def _try_load_local_raw(country: str, indicator: str) -> pd.DataFrame | None:
    """Try to load a manually saved IMF response from data_raw/.

    Supported file names:
      - data_raw/JP_IMF_GDP_quarterly.json (IMF CompactData JSON)
      - data_raw/JP_IMF_GDP_quarterly.csv  (CSV with DATE, value)
    """
    from pathlib import Path
    import pandas as pd
    raw_json = Path('data_raw/JP_IMF_GDP_quarterly.json')
    raw_csv = Path('data_raw/JP_IMF_GDP_quarterly.csv')
    if raw_json.exists():
        try:
            js = json.loads(raw_json.read_text(encoding='utf-8'))
            return _df_from_compact_json(js, country, indicator)
        except Exception as e:  # pragma: no cover
            logger.warning('failed to parse local JSON %s: %s', raw_json, e)
    if raw_csv.exists():
        try:
            df = pd.read_csv(raw_csv)
            # Expect columns: DATE, VALUE (or second column)
            if 'DATE' in df.columns and df.shape[1] >= 2:
                val_col = [c for c in df.columns if c != 'DATE'][0]
                out = df[['DATE', val_col]].copy()
                out['DATE'] = pd.to_datetime(out['DATE']).dt.date
                out = out.rename(columns={val_col: f'{country}_{indicator}_NC'})
                out = out.sort_values('DATE').reset_index(drop=True)
                return out
        except Exception as e:  # pragma: no cover
            logger.warning('failed to parse local CSV %s: %s', raw_csv, e)
    return None


def write_manifest(out_path: Path, url: str, count: int):
    manifest = {
        'source': 'IMF IFS',
        'url': url,
        'indicator': out_path.stem,
        'rows': count,
        'generated_at': datetime.datetime.utcnow().isoformat() + 'Z'
    }
    (out_path.parent / (out_path.stem + '_manifest.json')).write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def _ensure_prefixed_default(path: str) -> str:
    """Promote legacy default to prefixed name if user did not override.

    If path equals historical 'data_processed/JP_IMF_GDP_quarterly.csv' replace
    with 'data_processed/fetch_code07_JP_IMF_GDP_quarterly.csv'. Otherwise keep.
    """
    legacy = 'data_processed/JP_IMF_GDP_quarterly.csv'
    pref = 'data_processed/fetch_code07_JP_IMF_GDP_quarterly.csv'
    if path == legacy:
        return pref
    return path

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--country', default='JP')
    ap.add_argument('--indicator', default='NGDP', help='IFS indicator (e.g. NGDP nominal, NGDP_R real)')
    ap.add_argument('--freq', default='Q', choices=['Q'])
    ap.add_argument('--start', default='2010')
    ap.add_argument('--end', default='2025')
    ap.add_argument('--out', default='data_processed/fetch_code07_JP_IMF_GDP_quarterly.csv')
    ap.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    ap.add_argument('--timeout', type=int, default=90)
    ap.add_argument('--retries', type=int, default=6, help='HTTP retry attempts')
    ap.add_argument('--backoff', type=float, default=2.0, help='Exponential backoff factor between retries')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--chunk-years', type=int, default=8, help='Fetch in windows of N years (reduces server resets)')
    ap.add_argument('--no-manifest', action='store_true')
    ap.add_argument('--raw-out', help='Optional path to save raw IMF response body (JSON)')
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format='%(levelname)s %(message)s')

    url = build_url(args.freq, args.country, args.indicator, args.start, args.end)
    if args.dry_run:
        logger.info('(dry-run) would GET %s', url)
        return 0

    try:
        raw_path = Path(args.raw_out) if args.raw_out else None
        if args.chunk_years and int(args.chunk_years) > 0:
            df = fetch_imf_gdp_chunked(
                args.freq,
                args.country,
                args.indicator,
                args.start,
                args.end,
                timeout=args.timeout,
                retries=args.retries,
                backoff=args.backoff,
                chunk_years=int(args.chunk_years),
            )
        else:
            df = fetch_imf_gdp(
                args.freq,
                args.country,
                args.indicator,
                args.start,
                args.end,
                timeout=args.timeout,
                retries=args.retries,
                backoff=args.backoff,
                raw_out=raw_path,
            )
        if raw_path is not None:
            try:
                # Save raw JSON for reproducibility
                url = build_url(args.freq, args.country, args.indicator, args.start, args.end)
                sess = _make_session(args.retries, args.backoff)
                r = sess.get(url, timeout=args.timeout)
                r.raise_for_status()
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(r.text, encoding='utf-8')
                logger.info('saved raw JSON -> %s', raw_path)
            except Exception as e:  # pragma: no cover
                logger.warning('saving raw JSON failed: %s', e)
    except Exception as e:
        logger.error('fetch failed: %s', e)
        # Try local fallback from data_raw/
        df = _try_load_local_raw(args.country, args.indicator)
        if df is None:
            url = build_url(args.freq, args.country, args.indicator, args.start, args.end)
            print(
                "[MANUAL] IMF data download failed. Please open the URL below in a browser, then save the response to data_raw/:\n"
                f"  {url}\n"
                "Suggested filename: JP_IMF_GDP_quarterly.json (or .csv if you export as CSV)\n"
                "Tips: increase --timeout/--retries/--backoff, or run with --dry-run to just print the URL."
            )
            return 1
        else:
            logger.warning('using local fallback from data_raw/* due to network failure')

    out_path = Path(_ensure_prefixed_default(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info('wrote %s rows=%d', out_path, len(df))
    if not args.no_manifest:
        write_manifest(out_path, url, len(df))
    return 0

if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
