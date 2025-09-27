"""fetch_code03_extract_jsda_rmbs_components

Parse JSDA appendix (semiannual) RMBS component breakdown table from a pre-fetched
CSV. Canonical input is ``data_processed/JP_RMBS_components_semiannual_JSDA_2012_2025.csv``.
Minimal implementation: no line-wrap/repair heuristics (raw file is assumed structurally
correct: 4 columns).
"""
from __future__ import annotations
import argparse, re, logging
from pathlib import Path
import pandas as pd
from datetime import datetime

def _default_out_name(base: str) -> str:
    """Ensure output filename starts with fetch_code03_ for provenance.

    If user passed a path that already has the prefix, leave unchanged.
    If they passed a directory, append default filename.
    """
    p = Path(base)
    if p.is_dir():
        return str(p / 'fetch_code03_JSDA_RMBS_components.csv')
    if p.name.startswith('fetch_code03_'):
        return str(p)
    return str(p.with_name('fetch_code03_' + p.name))

logger = logging.getLogger(__name__)

def parse_jsda_table(csv_path: Path, *, debug: bool = False) -> pd.DataFrame:
    """Parse the semiannual JSDA/JHF RMBS component CSV.

    Robust to header variations:
      - Accepts >4 columns and detects columns by tokens: date/rmbs/gdp/pct|ratio
      - If ratio column missing but RMBS & NGDP present, computes MBS_to_GDP_pct automatically.
      - Standardizes to columns: ['Date','JP_RMBS_Bn','NGDP_Bn','MBS_to_GDP_pct'] when possible.
    """
    from io import StringIO
    raw = csv_path.read_text(encoding='utf-8', errors='ignore')
    try:
        df = pd.read_csv(StringIO(raw))
    except Exception:
        # Fallback direct path read (should be equivalent unless encoding issues)
        df = pd.read_csv(csv_path)
    if debug:
        logger.info("raw columns: %s", list(df.columns))
    # Standard expected columns
    exp = ['Date','JP_RMBS_Bn','NGDP_Bn','MBS_to_GDP_pct']
    # Token-based detection
    date_col = None
    rmbs_col = None
    gdp_col = None
    ratio_col = None
    for c in df.columns:
        lc = str(c).lower()
        if date_col is None and ('date' in lc or '期' in lc or lc in {'time','quarter','period'}):
            date_col = c
        if rmbs_col is None and ('rmbs' in lc or 'jhf' in lc and 'rmbs' in lc):
            rmbs_col = c
        if gdp_col is None and 'gdp' in lc:
            gdp_col = c
        # detect percentage/ratio column; allow tokens 'pct', 'ratio', or a literal '%'
        if ratio_col is None and ('pct' in lc or 'ratio' in lc or '%' in lc):
            ratio_col = c
    # Heuristic: if exactly 4 columns, try simple renaming
    if len(df.columns) == 4 and not all(c in df.columns for c in exp):
        new_cols = []
        for c in df.columns:
            lc = str(c).lower()
            if 'date' in lc:
                new_cols.append('Date')
            elif 'rmbs' in lc:
                new_cols.append('JP_RMBS_Bn')
            elif 'gdp' in lc:
                new_cols.append('NGDP_Bn')
            elif 'pct' in lc or 'ratio' in lc or '%' in lc:
                new_cols.append('MBS_to_GDP_pct')
            else:
                new_cols.append(str(c))
        if len(set(new_cols)) == 4:
            df.columns = new_cols
            date_col, rmbs_col, gdp_col, ratio_col = 'Date','JP_RMBS_Bn','NGDP_Bn','MBS_to_GDP_pct'
    # Build standardized output when possible
    if date_col is not None and rmbs_col is not None and gdp_col is not None:
        out = pd.DataFrame({
            'Date': pd.to_datetime(df[date_col], errors='coerce'),
            'JP_RMBS_Bn': pd.to_numeric(df[rmbs_col], errors='coerce'),
            'NGDP_Bn': pd.to_numeric(df[gdp_col], errors='coerce'),
        })
        out = out.dropna(subset=['Date']).sort_values('Date')
        if ratio_col is not None and ratio_col in df.columns:
            out['MBS_to_GDP_pct'] = pd.to_numeric(df[ratio_col], errors='coerce')
        else:
            with pd.option_context('mode.use_inf_as_na', True):
                out['MBS_to_GDP_pct'] = (out['JP_RMBS_Bn'] / out['NGDP_Bn']) * 100.0
        if debug:
            logger.info("mapped columns -> %s", out.columns.tolist())
        return out[exp]
    # Fallback: try to coerce a Date column name variant and pass-through
    for c in df.columns:
        if str(c).upper() == 'DATE':
            df = df.rename(columns={c:'Date'})
            break
    return df

def main(argv=None):
    from util_code01_lib_io import find
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data_processed/JP_RMBS_components_semiannual_JSDA_2012_2025.csv', help='Canonical JSDA components CSV')
    ap.add_argument('--out', default='data_processed/fetch_code03_JSDA_RMBS_components.csv')
    ap.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    ap.add_argument('--strict', action='store_true', help='Exit with code 1 if expected columns cannot be standardized')
    ap.add_argument('--debug', action='store_true', help='Verbose parsing diagnostics')
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(levelname)s %(message)s')
    inp = Path(args.input)
    if not inp.exists():
        # Auto-discover from common names (canonical first, then legacy)
        cands = find([
            'JP_RMBS_components_semiannual_JSDA_2012_2025.csv',
            'fetch_code03_JSDA_RMBS_components.csv',
            'JP_JHF_RMBS_vs_GDP*.csv',  # legacy merged export (fallback)
        ]) or []
        if cands:
            inp = Path(cands[0])
            logger.info("auto-detected input: %s", inp)
        else:
            logger.error("input not found: %s (no candidates detected)", args.input)
            return 1
    out = Path(_default_out_name(args.out))
    df = parse_jsda_table(inp, debug=args.debug)
    # If strict and we failed to standardize, enforce presence of expected columns
    exp = ['Date','JP_RMBS_Bn','NGDP_Bn','MBS_to_GDP_pct']
    if args.strict and not all(c in df.columns for c in exp):
        logger.error("expected columns missing under --strict: %s (got %s)", exp, list(df.columns))
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    # Always write without overwrite (timestamp suffix if exists)
    if out.exists():
        stem, suf = out.stem, out.suffix
        ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        out = out.with_name(f"{stem}_{ts}{suf}")
    df.to_csv(out, index=False)
    logger.info("wrote %s rows=%d", out, len(df))
    return 0

if __name__ == '__main__':  # pragma: no cover
    import sys
    sys.exit(main())
