#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fetch_code01_capital_k_data

Downloads core datasets for CET1/RWA proxies and mortgage rates/terms (FRED).
HMDA is NOT fetched by default (manual browser download is recommended).
Example:
        # FRED only (default)
        python fetch_code01_capital_k_data.py
        # Enable automatic HMDA fetch (legacy-compatible)
        python fetch_code01_capital_k_data.py --enable-hmda-fetch --years 2018-2024 --hmda-auto-trim

Notes:
                - Public endpoints only. JHF Excel must be added manually.
                - HMDA is currently expected to be downloaded manually via browser: visit
                    https://ffiec.cfpb.gov/data-publication/ and export CSV; place it as
                    data_raw/HMDA_YYYY_originated_nationwide.csv. To use auto fetch again,
                    pass --enable-hmda-fetch.
                - Improved retry with exponential backoff (handles 408/429/5xx).
"""
from __future__ import annotations
import argparse, sys, time, json, gzip, io, random, logging, datetime
from pathlib import Path
from urllib.parse import urlencode
import requests
from requests import HTTPError, Response
from typing import List, Dict, Sequence, Iterable, Union
import pandas as pd  # post-processing

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
# Default FRED series (filename -> metadata)
SERIES: Dict[str, Dict[str, str]] = {
    # Mortgage rate (weekly) - Freddie Mac PMMS via FRED
    "MORTGAGE30US.csv": {"id": "MORTGAGE30US"},
    # World Bank GFDD.SI.05 via FRED (regulatory capital to RWA, %), annual
    "CET1_RWA_US_WB.csv": {"id": "DDSI05USA156NWDB"},
    "CET1_RWA_JP_WB.csv": {"id": "DDSI05JPA156NWDB"},
    # Merged from fetch_code10 defaults (loans on bank balance sheet, agency/nonagency pools)
    # RREACBM027SBOG: Real Estate Loans: Residential Real Estate Loans, All Commercial Banks
    "RREACBM027SBOG.csv": {"id": "RREACBM027SBOG"},
    # AGSEBMPTCMAHDFS: Agency- and GSE-Backed Mortgage Pools; Total Cash Mortgage Assets Held by Domestic Financial Sectors
    "AGSEBMPTCMAHDFS.csv": {"id": "AGSEBMPTCMAHDFS"},
}

def fred_csv(series_id: str) -> str:
    return f"{FRED_CSV}?{urlencode({'id': series_id})}"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Accept": "text/csv,application/json,application/text;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}

def fetch(
    url: str,
    out: Path,
    *,
    headers: Dict[str,str]|None = None,
    retries: int = 3,
    backoff: float = 2.0,
    timeout: int = 60,
    verbose: bool = True,
    return_content: bool = False,
    session: requests.Session|None = None,
) -> Union[Path, bytes]:
    out.parent.mkdir(parents=True, exist_ok=True)
    _headers = DEFAULT_HEADERS.copy()
    if headers:
        _headers.update(headers)
    attempt = 0
    sess = session or requests.Session()
    while True:
        attempt += 1
        try:
            r: Response = sess.get(url, headers=_headers, timeout=timeout)
            status = r.status_code
            if status == 403:
                r.raise_for_status()
            r.raise_for_status()
            content = r.content
            if return_content:
                return content
            out.write_bytes(content)
            return out
        except HTTPError as e:
            code = getattr(e.response, "status_code", None)
            transient = (code is None) or (500 <= code < 600) or code in (408, 429)
            if code == 403:
                if verbose:
                    print(
                        f"[WARN] 403 Forbidden for {url}\n       -> Try:"
                        "\n          - Open the URL once in a browser"
                        "\n          - Increase --hmda-sleep (e.g. 3-5s)"
                        "\n          - Use --hmda-continue to skip blocked years"
                        "\n          - Different network (VPN off)"
                    )
                raise
            if (attempt <= retries) and transient:
                jitter = random.uniform(0.25,0.75)
                sleep_for = (backoff * (2 ** (attempt - 1))) * jitter
                if verbose:
                    print(f"[retry {attempt}/{retries}] {url} (status {code}); sleeping {sleep_for:.1f}s...")
                time.sleep(sleep_for)
                continue
            raise
        except requests.RequestException as e:
            if attempt <= retries:
                jitter = random.uniform(0.25,0.75)
                sleep_for = (backoff * (2 ** (attempt - 1))) * jitter
                if verbose:
                    print(f"[retry {attempt}/{retries}] network error {e}; sleeping {sleep_for:.1f}s...")
                time.sleep(sleep_for)
                continue
            raise

def fetch_fred(outdir: Path, mapping: Dict[str, Dict[str,str]]) -> List[str]:
    """Download a mapping of filename -> {id: FRED_SERIES_ID}."""
    saved: List[str] = []
    for fn, meta in mapping.items():
        sid = meta.get("id")
        if not sid:
            logging.warning("[FRED] missing id for %s - skip", fn)
            continue
        url = fred_csv(sid)
        path = outdir / fn
        try:
            fetch(url, path)
        except Exception as e:
            logging.error("[FRED] failed %s (%s): %s", sid, fn, e)
            raise
        saved.append(str(path))
    return saved

# ---------------------------------------------------------------------------
# CET1/RWA post-processing utilities
# ---------------------------------------------------------------------------
def _read_fred_single(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logging.warning("read fail %s: %s", path, e)
        return pd.DataFrame()
    if 'DATE' not in df.columns:
        return pd.DataFrame()
    val_cols = [c for c in df.columns if c != 'DATE']
    if not val_cols:
        return pd.DataFrame()
    v = val_cols[0]
    df = df[['DATE', v]].copy()
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['DATE'])
    df[v] = pd.to_numeric(df[v], errors='coerce')
    return df.dropna(subset=[v])

def normalize_cet1_rwa(raw_dir: Path, processed_dir: Path) -> Path|None:
    us_path = raw_dir / 'CET1_RWA_US_WB.csv'
    jp_path = raw_dir / 'CET1_RWA_JP_WB.csv'
    if not us_path.exists() or not jp_path.exists():
        logging.warning("[POST] CET1 files missing -> skip")
        return None
    us = _read_fred_single(us_path)
    jp = _read_fred_single(jp_path)
    if us.empty or jp.empty:
        logging.warning("[POST] CET1 data empty after load -> skip")
        return None
    us = us.rename(columns={us.columns[1]: 'US_CET1_RWA_pct'})
    jp = jp.rename(columns={jp.columns[1]: 'JP_CET1_RWA_pct'})
    merged = pd.merge(us, jp, on='DATE', how='outer').sort_values('DATE')
    processed_dir.mkdir(parents=True, exist_ok=True)
    # Historical name (unprefixed): CET1_RWA_WB_combined.csv
    # New standardized name (prefixed): fetch_code01_CET1_RWA_WB_combined.csv
    out = processed_dir / 'fetch_code01_CET1_RWA_WB_combined.csv'
    merged.to_csv(out, index=False)
    logging.info("[POST] wrote %s rows=%d", out, len(merged))
    return out

def write_manifest(raw_files: List[str], outdir: Path) -> Path:
    """Write a manifest for raw acquisitions (formerly downloads_manifest)."""
    manifest = {
        'generated_at': datetime.datetime.utcnow().isoformat() + 'Z',
        'count': len(raw_files),
        'files': raw_files,
    }
    path = outdir / 'raw_manifest.json'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    logging.info("[INFO] manifest written %s", path)
    return path

def fetch_hmda(
    outdir: Path,
    years: Sequence[int],
    *,
    retries: int = 2,
    sleep_between: float = 1.0,
    continue_on_error: bool = False,
    loan_purpose: str|None = None,
    dwelling: str|None = None,
    auto_trim: bool = False,
    save_gzip: bool = False,
    debug: bool = False,
    max_filters: int = 2,
) -> List[str]:
    """
    HMDA Data Browser API (CSV) – nationwide subset.
    Docs: https://ffiec.cfpb.gov/documentation/api/data-browser/

    The API restricts number of filter criteria (aside from years) for some aggregation endpoints.
    """
    saved = []
    base = "https://ffiec.cfpb.gov/v2/data-browser-api/view/nationwide/csv"
    base_params = {"actions_taken": "1"}
    base_optional: List[tuple[str,str]] = []
    if loan_purpose:
        base_optional.append(("loan_purposes", loan_purpose))
    if dwelling:
        base_optional.append(("dwelling_categories", dwelling))

    def build_param_dict(opts: Iterable[tuple[str,str]]):
        p = dict(base_params)
        for k,v in opts:
            p[k] = v
        return p

    def trim_filters(opts: List[tuple[str,str]]):
        if not auto_trim:
            return opts
        work = opts.copy()
        while len(work) + len(base_params) > max_filters and work:
            dropped = work.pop()
            if debug:
                print(f"[HMDA][auto-trim] dropped {dropped[0]}={dropped[1]}")
        return work

    for y in years:
        working = trim_filters(base_optional)
        params_full = build_param_dict(working)
        if len(params_full) > max_filters:
            msg = f"[HMDA] too many filters ({len(params_full)}) > {max_filters}. Use --hmda-auto-trim or remove filters."
            print(msg)
            if continue_on_error:
                continue
            return saved
        q = params_full.copy()
        q["years"] = str(y)
        url = base + "?" + urlencode(q, doseq=True)
        out_csv = outdir / f"HMDA_{y}_originated_nationwide.csv"
        try:
            raw = fetch(url, out_csv, retries=retries, return_content=True)
            if raw[:2] == b"\x1f\x8b":
                if debug:
                    print(f"[HMDA] year {y} gzip-compressed ({len(raw)} bytes)")
                if save_gzip:
                    (out_csv.with_suffix(out_csv.suffix + ".gz")).write_bytes(raw)
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                    decompressed = gz.read()
            else:
                decompressed = raw
            text = decompressed.decode('utf-8', 'replace')
            first_char = text.lstrip()[:1]
            if first_char == '{':
                try:
                    js = json.loads(text)
                except json.JSONDecodeError:
                    js = None
                if js and ('errorType' in js or 'message' in js):
                    err_path = outdir / f"HMDA_{y}_error.json"
                    err_path.write_text(json.dumps(js, indent=2))
                    print(f"[HMDA] API error year {y}: {js.get('errorType')} - {js.get('message')}. Saved {err_path}")
                    msg_txt = (js.get('message') or '')
                    if auto_trim and 'two or less' in msg_txt.lower() and base_optional:
                        if debug:
                            print("[HMDA][auto-trim] API complained; trimming one filter and retrying this year")
                        base_optional = base_optional[:-1]
                        continue
                    if continue_on_error:
                        continue
                    else:
                        return saved
            if not text.startswith('as_of_year'):
                inspect_path = outdir / f"HMDA_{y}_unexpected.txt"
                inspect_path.write_text(text[:5000])
                print(f"[HMDA] unexpected content year {y} (saved first 5KB to {inspect_path}); skipping parse")
                if continue_on_error:
                    continue
                else:
                    return saved
            out_csv.write_text(text)
            saved.append(str(out_csv))
            if sleep_between:
                time.sleep(sleep_between)
        except Exception as e:
            msg = f"[HMDA] failed year {y}: {e.__class__.__name__}: {e}"
            if continue_on_error:
                print(msg + " (continuing)")
                continue
            else:
                print(msg)
                raise
    return saved

def parse_years(expr: str) -> List[int]:
    if '-' in expr:
        a,b = expr.split('-',1)
        return list(range(int(a), int(b)+1))
    return [int(x) for x in expr.split(',') if x.strip()]

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data_raw", help="output folder (default: data_raw; previously downloads)")
    p.add_argument("--outdir", help="alias for --out (if provided overrides --out)")
    p.add_argument("--fred-series", help="Comma list of custom FRED series IDs (replaces default set)")
    p.add_argument("--fred-out", help="Output filename for single custom FRED series (e.g. mortgage_rate.csv)")
    p.add_argument("--years", default="2018-2024", help="HMDA years (e.g., 2018-2024 or 2018,2019,2020)")
    # HMDA: disabled by default (manual download guidance). Enable with --enable-hmda-fetch for legacy behavior.
    p.add_argument("--enable-hmda-fetch", action="store_true", help="(Not recommended) Automatically fetch HMDA (manual browser download preferred)")
    p.add_argument("--no-hmda", action="store_true", help="(Compatibility flag) Ignored / will be removed: HMDA is already disabled by default")
    p.add_argument("--hmda-retries", type=int, default=2, help="Retry attempts per HMDA year (default: 2)")
    p.add_argument("--hmda-sleep", type=float, default=1.0, help="Seconds sleep between HMDA requests")
    p.add_argument("--hmda-continue", action="store_true", help="Continue on HMDA year failure (partial set)")
    p.add_argument("--hmda-loan-purpose", help="HMDA loan_purposes filter (e.g. 1)")
    p.add_argument("--hmda-dwelling", help="HMDA dwelling_categories filter (exact API text)")
    p.add_argument("--hmda-auto-trim", action="store_true", help="Auto drop extra filters to satisfy API (<=2 filters)")
    p.add_argument("--hmda-save-gzip", action="store_true", help="Save raw gzip payload alongside CSV")
    p.add_argument("--hmda-debug", action="store_true", help="Verbose debug for HMDA fetch")
    p.add_argument("--hmda-max-filters", type=int, default=2, help="Maximum allowed filters before trimming (API constraint)")
    p.add_argument("--dry-run", action="store_true", help="List planned downloads only")
    p.add_argument("--postprocess", action="store_true", help="Run CET1/RWA normalization after download")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Logging level")
    args = p.parse_args(argv)

    # Resolve outdir (alias precedence)
    effective_out = args.outdir if args.outdir else args.out
    outdir = Path(effective_out)
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(levelname)s %(message)s')
    outdir.mkdir(parents=True, exist_ok=True)

    # Build FRED series mapping (default or custom)
    fred_mapping: Dict[str, Dict[str,str]]
    if args.fred_series:
        series_ids = [s.strip() for s in args.fred_series.split(',') if s.strip()]
        if not series_ids:
            logging.error("--fred-series provided but no valid IDs parsed")
            return 1
        if args.fred_out and len(series_ids) > 1:
            logging.error("--fred-out can only be used with a single --fred-series ID")
            return 1
        fred_mapping = {}
        for sid in series_ids:
            if args.fred_out and len(series_ids) == 1:
                fname = args.fred_out
            else:
                fname = f"{sid}.csv"
            fred_mapping[fname] = {"id": sid}
        logging.info("[FRED] custom series: %s", ", ".join(series_ids))
    else:
        fred_mapping = dict(SERIES)  # copy default
        logging.info(
            "[FRED] downloading default set (CET1, mortgage rate, bank loans, agency/GSE pools) [%d files]...",
            len(fred_mapping),
        )

    saved: List[str] = []
    if args.dry_run:
        for fn, meta in fred_mapping.items():
            logging.info("(dry-run) %s -> %s", meta['id'], outdir/fn)
    else:
        saved = fetch_fred(outdir, fred_mapping)
        for s in saved:
            logging.info("saved %s", s)

    if args.enable_hmda_fetch:
        if args.no_hmda:
            logging.warning("--no-hmda was ignored: --enable-hmda-fetch takes precedence")
        years = parse_years(args.years)
        logging.info("[HMDA][auto] downloading nationwide originated subset (one file per year)... (manual download is recommended)")
        if args.dry_run:
            logging.info("(dry-run) HMDA years %s", years)
        else:
            hmda_saved = fetch_hmda(
                outdir,
                years,
                retries=args.hmda_retries,
                sleep_between=args.hmda_sleep,
                continue_on_error=args.hmda_continue,
                loan_purpose=args.hmda_loan_purpose,
                dwelling=args.hmda_dwelling,
                auto_trim=args.hmda_auto_trim,
                save_gzip=args.hmda_save_gzip,
                debug=args.hmda_debug,
                max_filters=args.hmda_max_filters,
            )
            for s in hmda_saved:
                logging.info("saved %s", s)
            saved.extend(hmda_saved)
    else:
        logging.info(
            "[HMDA][manual] Auto-fetch is disabled. Steps:\n"
            " 1) Open https://ffiec.cfpb.gov/data-publication/ in your browser\n"
            " 2) In Data Browser, select filters such as 'Originated' and export CSV\n"
            " 3) Rename the file to HMDA_YYYY_originated_nationwide.csv and place it under data_raw/\n"
            " 4) After CET1/RWA normalization, k-/phi-building steps become available\n"
            "  (To use legacy auto-fetch: pass --enable-hmda-fetch)"
        )

    if not args.dry_run and saved:
        write_manifest(saved, outdir)
    if args.postprocess and not args.dry_run:
        normalize_cet1_rwa(outdir, Path('data_processed'))
    logging.info("[Manual] Download JHF 'Factors and Other Monthly Data' Excel (place in data_raw/jhf_factors/) then build k_JP.")
    return 0

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
