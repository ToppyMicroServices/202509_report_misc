"""fetch_code02_convert_jhf_xls_to_xlsx

Batch convert legacy JHF factor workbook files (.xls) to .xlsx for consistent downstream parsing.
This script is intentionally minimal: schema validation is deferred to downstream loaders that know specific sheet layouts.
"""
from __future__ import annotations
import argparse, logging, re
from pathlib import Path
import pandas as pd
import requests
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

def convert_file(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    # Preserve original filename stem but add fetch_code02_ prefix for provenance.
    out = dst_dir / (f"fetch_code02_{src.stem}.xlsx")
    try:
        ext = src.suffix.lower()
        engine_kwargs = {}
        if ext == '.xls':
            engine_kwargs = {'engine': 'xlrd'}  # requires xlrd>=2.0.1
        elif ext == '.xlsx':
            engine_kwargs = {'engine': 'openpyxl'}
        df = pd.read_excel(src, **engine_kwargs)
        df.to_excel(out, index=False)
        logger.info("converted %s -> %s", src.name, out.name)
    except Exception as e:
        msg = str(e)
        if src.suffix.lower() == '.xls' and 'xlrd' in msg.lower():
            logger.error("convert fail %s: %s (install dependency: pip install xlrd>=2.0.1)", src.name, e)
        else:
            logger.error("convert fail %s: %s", src.name, e)
    return out

def _download_jhf(page_url: str, outdir: Path, *, timeout: int = 20, overwrite: bool = False) -> list[Path]:
    """Download JHF monthly workbook files (.xls/.xlsx) from the given page URL.

    Returns list of downloaded (or existing, if skipped) file paths.
    Patterns: primary (shell-equivalent) and a generic fallback.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(page_url, headers=headers, timeout=timeout)
    r.raise_for_status()
    html = r.text
    # Primary pattern (matches the shell script behavior)
    pat_primary = re.compile(r'https://www\.jhf\.go\.jp/files/topics/5880_ext_99_[^"]*\.xls[x]?', re.I)
    # Fallback, more generic topics file matcher
    pat_generic = re.compile(r'https://www\.jhf\.go\.jp/files/topics/[^\"]+\.xls[x]?', re.I)
    urls = sorted(set(pat_primary.findall(html)) or set(pat_generic.findall(html)))
    # Fallback: extract any href ending with .xls/.xlsx (supports relative links)
    if not urls:
        href_pat = re.compile(r'''href=["']([^"']*\.xls[x]?)["']''', re.I)
        rels = [m.group(1) for m in href_pat.finditer(html)]
        urls = []
        for u in rels:
            try:
                absu = urljoin(page_url, u)
                # Restrict to jhf.go.jp domain to avoid accidental captures
                if 'jhf.go.jp' in absu:
                    urls.append(absu)
            except Exception:
                continue
        urls = sorted(set(urls))
    if not urls:
        logger.warning("no JHF XLS/XLSX links detected at %s (try --page-url to a page that lists monthly files)", page_url)
        return []
    logger.info("found %d JHF files", len(urls))
    saved: list[Path] = []
    with requests.Session() as sess:
        sess.headers.update(headers)
        for u in urls:
            name = u.split("/")[-1]
            dest = outdir / name
            if dest.exists() and not overwrite:
                logger.info("exists (skip) %s", dest.name)
                saved.append(dest)
                continue
            try:
                with sess.get(u, timeout=timeout, stream=True) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1 << 15):
                            if chunk:
                                f.write(chunk)
                logger.info("downloaded %s", dest.name)
                saved.append(dest)
            except requests.RequestException as e:
                logger.error("download fail %s: %s", name, e)
    return saved

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-dir', default='data_raw/jhf_factors', help='Source directory containing legacy JHF .xls/.xlsx files')
    ap.add_argument('--dst-dir', default='data_raw/jhf_factors_xlsx', help='Destination directory for converted .xlsx files')
    ap.add_argument('--glob', default='*.xls*', help='Glob pattern for source files (default: *.xls*)')
    # Integrated downloader options (merged from fetch_code04_jhf_downloader.sh)
    ap.add_argument('--download', action='store_true', default=True, help='Fetch JHF monthly XLS/XLSX files into --src-dir before conversion (default: on)')
    ap.add_argument('--no-download', dest='download', action='store_false', help='Do not fetch; convert existing files only')
    ap.add_argument('--page-url', default='https://www.jhf.go.jp/english/mbs_m_f_monthly.html', help='JHF listing page URL')
    ap.add_argument('--timeout', type=int, default=20, help='HTTP timeout seconds for --download')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing files when downloading')
    ap.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], help='Logging level')
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(levelname)s %(message)s')
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    # Optional: download first
    if args.download:
        try:
            files = _download_jhf(args.page_url, src_dir, timeout=args.timeout, overwrite=args.overwrite)
            if files:
                logger.info('downloaded/kept %d files into %s', len(files), src_dir)
        except Exception as e:
            logger.error('download step failed: %s', e)
    files = sorted(src_dir.glob(args.glob))
    if not files:
        logger.warning('No files matching pattern %s in %s', args.glob, src_dir)
    for f in files:
        convert_file(f, dst_dir)
    return 0

if __name__ == '__main__':  # pragma: no cover
    import sys
    sys.exit(main())
