"""fetch_code06_esri_manual_jp_gdp (legacy optional)

NOTE:
    IMF IFS (see fetch_code07_imf_jp_gdp.py) is now the primary source for JP quarterly nominal GDP.
    This module remains ONLY for manual / fallback comparison with ESRI (速報/確報) when a CSV URL is known.
    Auto-discovery routines are retained but may be removed in future. Prefer supplying --url explicitly.
"""
from __future__ import annotations
import pandas as pd, argparse, logging, requests
from typing import Dict, Optional
from pathlib import Path
import itertools

FREQ_Q_END = 'Q'
ESRI_CSV_URL = "https://www.esri.cao.go.jp/jp/sna/data/sokuhou/files/2025/qe252_2/tables/gaku-mk2522.csv"  # default (will age)

logger = logging.getLogger(__name__)

def fetch_esri_jp_gdp(status: Dict, skip: bool=False, *, pattern: str = "Gross domestic product", raw_save: Path|None = None, url: str = ESRI_CSV_URL) -> pd.DataFrame:
    if skip:
        status["esri_gdp"] = ("SKIP", "--skip specified")
        return pd.DataFrame()
    try:
        raw = pd.read_csv(url, header=None)
    except Exception as e:
        status["esri_gdp"] = ("ERROR", f"fetch fail {e}")
        return pd.DataFrame()
    if raw_save:
        try:
            raw.to_csv(raw_save, index=False)
            logger.debug("saved raw ESRI CSV -> %s", raw_save)
        except Exception as e:
            logger.debug("failed raw save: %s", e)
    row_idx = raw[raw.astype(str).apply(lambda r: r.str.contains(pattern, case=False, na=False)).any(axis=1)].index
    if not len(row_idx):
        # Capture a small sample for diagnostics
        sample_lines = []
        for i in range(min(15, len(raw))):
            sample_lines.append("|".join(str(x) for x in raw.iloc[i].tolist()[:8]))
        status["esri_gdp"] = ("ERROR", f"pattern not found: '{pattern}' sample_rows=\n" + "\n".join(sample_lines))
        return pd.DataFrame()
    row = row_idx[0]
    series = raw.iloc[row].reset_index(drop=True)
    q_cols = [c for c in range(len(series)) if isinstance(series[c], str) and "Q" in series[c]]
    dates = pd.PeriodIndex(series[q_cols], freq=FREQ_Q_END).to_timestamp(FREQ_Q_END)
    values = raw.iloc[row+1, q_cols].astype(float)
    df = pd.DataFrame({"DATE": dates, "JP_GDP": values}).set_index("DATE").asfreq(FREQ_Q_END)
    status["esri_gdp"] = ("OK", f"rows={len(df)}")
    return df

def _auto_locate(year: int, *, q_min: int = 230, q_max: int = 300, max_revision: int = 4, timeout: float = 5.0, attempts: Optional[list] = None) -> Optional[str]:
    """Heuristic search for latest ESRI GDP CSV.

    Pattern observed: files/2025/qe252_2/tables/gaku-mk2522.csv
      directory: qe{Q}_{R}
      file:      gaku-mk{Q}{R}.csv

    We iterate Q downward (q_max -> q_min), revision descending.
    Records attempted URLs if *attempts* list provided.
    Returns first 200 OK URL.
    """
    base = f"https://www.esri.cao.go.jp/jp/sna/data/sokuhou/files/{year}"
    session = requests.Session()
    headers = {"User-Agent":"Mozilla/5.0"}
    for q, r in itertools.product(range(q_max, q_min-1, -1), range(max_revision, 0, -1)):
        directory = f"qe{q}_{r}"
        fname = f"gaku-mk{q}{r}.csv"
        url = f"{base}/{directory}/tables/{fname}"
        if attempts is not None:
            attempts.append(url)
        try:
            resp = session.head(url, timeout=timeout, headers=headers, allow_redirects=True)
            if resp.status_code == 200 and resp.headers.get('Content-Type','').lower().startswith('text'):
                return url
        except requests.RequestException:
            continue
    return None

def _html_discover(year: int, *, timeout: float = 8.0) -> Optional[str]:
    """HTML を GET して gaku-mk#####.csv パターンを直接抽出し最大 (Q,rev) を返す。

    戻り値: 最も大きい (mk 番号) の完全 URL か None。
    注意: ディレクトリは qe{Q}_{R}/tables/gaku-mk{Q}{R}.csv という従来規則を仮定。
    """
    base = f"https://www.esri.cao.go.jp/jp/sna/data/sokuhou/files/{year}"
    try:
        resp = requests.get(base + '/', timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        if resp.status_code >= 400:
            return None
        html = resp.text
    except requests.RequestException:
        return None
    import re
    # 例: gaku-mk2522.csv のように mk + Q(3桁前後) + revision(1桁)
    pattern = re.compile(r"gaku-mk(\d{3,4})(\d)\.csv")
    candidates = []
    for m in pattern.finditer(html):
        qrev = m.group(1)  # 例 252
        rev = int(m.group(2))
        try:
            q_num = int(qrev)
        except ValueError:
            continue
        candidates.append((q_num, rev))
    if not candidates:
        return None
    # 最大 Q, その中で最大 rev
    candidates.sort(reverse=True)
    q_num, rev = candidates[0]
    directory = f"qe{q_num}_{rev}"
    fname = f"gaku-mk{q_num}{rev}.csv"
    return f"{base}/{directory}/tables/{fname}"

def _html_deep_discover(year: int, *, timeout: float = 8.0, log_samples: int = 12) -> Optional[str]:
    """Deep crawl: year root -> find qeXXX_Y dirs -> fetch each /tables/ listing -> collect gaku-mk####.csv.
    Return newest (max Q, then max rev)."""
    import re
    base = f"https://www.esri.cao.go.jp/jp/sna/data/sokuhou/files/{year}"
    try:
        root_html = requests.get(base + '/', timeout=timeout, headers={"User-Agent":"Mozilla/5.0"}).text
    except requests.RequestException:
        return None
    dir_pat = re.compile(r"qe(\d{3,4})_(\d)")
    dirs = sorted({m.group(0) for m in dir_pat.finditer(root_html)}, reverse=True)
    if not dirs:
        return None
    file_pat = re.compile(r"gaku-mk(\d{3,4})(\d)\.csv")
    found: list[tuple[int,int,str]] = []
    for d in dirs:
        tables_url = f"{base}/{d}/tables/"
        try:
            html = requests.get(tables_url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"}).text
        except requests.RequestException:
            continue
        for m in file_pat.finditer(html):
            try:
                q_num = int(m.group(1))
                rev = int(m.group(2))
            except ValueError:
                continue
            fname = f"gaku-mk{q_num}{rev}.csv"
            found.append((q_num, rev, f"{tables_url}{fname}"))
    if not found:
        return None
    found.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for s in found[:log_samples]:
        logger.debug('deep-candidate Q=%d rev=%d %s', s[0], s[1], s[2])
    return found[0][2]

def _ensure_prefixed_default(path: str) -> str:
    legacy = 'data_processed/JP_ESRI_GDP_quarterly.csv'
    pref = 'data_processed/fetch_code06_JP_ESRI_GDP_quarterly.csv'
    if path == legacy:
        return pref
    return path

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data_processed/fetch_code06_JP_ESRI_GDP_quarterly.csv')
    ap.add_argument('--skip', action='store_true')
    ap.add_argument('--pattern', default='Gross domestic product', help='Row match phrase to locate GDP line')
    ap.add_argument('--raw-save', help='Optional path to save raw downloaded CSV')
    ap.add_argument('--url', help='Explicit ESRI GDP CSV URL (overrides default & auto-locate)')
    ap.add_argument('--auto-locate', action='store_true', help='Attempt heuristic discovery of newest CSV (HEAD scan)')
    ap.add_argument('--auto-html', action='store_true', help='Parse year root HTML to find newest gaku-mk*.csv')
    ap.add_argument('--auto-deep', action='store_true', help='Deep crawl subdirectories for newest CSV')
    ap.add_argument('--year', type=int, default=2025, help='Target year directory for --auto-locate')
    ap.add_argument('--locate-q-min', type=int, default=230, help='Search lower bound Q code (inclusive)')
    ap.add_argument('--locate-q-max', type=int, default=400, help='Search upper bound Q code (inclusive)')
    ap.add_argument('--locate-max-revision', type=int, default=4, help='Max revision number to try (descending)')
    ap.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(levelname)s %(message)s')
    status: Dict[str, tuple[str,str]] = {}
    raw_save_path = Path(args.raw_save) if args.raw_save else None
    target_url = args.url or ESRI_CSV_URL
    # 0) Deep crawl (highest coverage)
    if args.auto_deep and not args.url:
        deep = _html_deep_discover(args.year)
        if deep:
            logger.info('html-deep success: %s', deep)
            target_url = deep
        else:
            logger.warning('html-deep failed (continuing fallback chain)')
    # 1) Shallow HTML discover
    if args.auto_html and not args.url and target_url == ESRI_CSV_URL:
        discovered = _html_discover(args.year)
        if discovered:
            logger.info('html-discover success: %s', discovered)
            target_url = discovered
        else:
            logger.warning('html-discover failed (continuing fallback chain)')
    # 2) HEAD probing (heuristic sequential URL scan)
    if args.auto_locate and not args.url and (target_url == ESRI_CSV_URL or args.auto_html):
        attempts: list[str] = []
        located = _auto_locate(
            args.year,
            q_min=args.locate_q_min,
            q_max=args.locate_q_max,
            max_revision=args.locate_max_revision,
            attempts=attempts,
        )
        if located:
            logger.info('auto-locate success: %s', located)
            target_url = located
        else:
            logger.warning('auto-locate failed; falling back to default URL %s', target_url)
            if attempts:
                sample = attempts[:8]
                logger.info('Sample of attempted URLs (first up to 8):\n  %s', '\n  '.join(sample))
            # Provide a browser guidance message
            guess_next = f"https://www.esri.cao.go.jp/jp/sna/data/sokuhou/files/{args.year}/qe{args.locate_q_max+1}_1/tables/gaku-mk{args.locate_q_max+1}1.csv"
            logger.info('Newest file might sit outside the established pattern right after a site update. Try opening in a browser and navigate upward: \n  %s', guess_next)
            logger.info('Once confirmed in browser, rerun with --url to fetch explicitly.')
    logger.debug('using ESRI url: %s', target_url)
    df = fetch_esri_jp_gdp(status, skip=args.skip, pattern=args.pattern, raw_save=raw_save_path, url=target_url)
    if df.empty:
        logger.error('ESRI GDP empty or skipped (%s)', status.get('esri_gdp'))
        base = f"https://www.esri.cao.go.jp/jp/sna/data/sokuhou/files/{args.year}/"
        prev_year = args.year - 1 if isinstance(args.year, int) else 2024
        print(
            "[MANUAL] ESRI data download failed. Quick options:\n"
            "\n"
            "Fastest path:\n"
            f"  1) Open {base} in your browser\n"
            "  2) Find directory 'qe{Q}_{R}/tables/' and file 'gaku-mk{Q}{R}.csv'\n"
            "  3) Re-run with explicit URL, e.g.:\n"
            "     python code/fetch_code06_esri_jp_gdp.py --url '<paste-found-url>'\n"
            "\n"
            "Alternative (previous year auto-discovery):\n"
            f"  python code/fetch_code06_esri_jp_gdp.py --auto-locate --auto-html --auto-deep --year {prev_year}\n"
            "\n"
            "Practical note: JP nominal GDP primary source is IMF (fetch_code07).\n"
            "If IMF works for you, ESRI is optional."
        )
        return 1
    out = Path(_ensure_prefixed_default(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    logger.info('wrote %s rows=%d', out, len(df))
    return 0

if __name__ == '__main__':  # pragma: no cover
    import sys
    sys.exit(main())
