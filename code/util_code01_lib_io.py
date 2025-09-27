"""util_code01_lib_io

Core IO / filesystem / figure helpers (formerly in lib_io).
All modules should import from this canonical name now.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve()
_CODE_DIR = _HERE.parent
_REPO_ROOT = _CODE_DIR.parent

_BASE_DIRS = [
    # relative to current working dir
    Path("data_raw"),
    Path("data_processed"),
    Path("."),
    Path("code"),
    Path("..")/"data_raw",
    Path("..")/"data_processed",
    # absolute repo-root candidates
    _REPO_ROOT/"data_raw",
    _REPO_ROOT/"data_processed",
    _REPO_ROOT,
    _CODE_DIR,
]

def find(pathnames):
    """Search glob patterns in common project folders and return absolute Paths.

    Search order includes (relative to cwd and repo root): data_raw, data_processed, '.', 'code', '../data_raw', '../data_processed'.
    Returns a list of unique, existing files as absolute Path objects.
    """
    results = []
    seen = set()
    for pat in pathnames:
        for base in _BASE_DIRS:
            if not base.exists():
                continue
            for p in base.glob(pat):
                if p.is_file():
                    rp = p.resolve()
                    if rp not in seen:
                        results.append(rp)
                        seen.add(rp)
    return results

def read_fred_two_col(csv_path, date_col="DATE"):
    """Read a typical FRED CSV with DATE + one value column.

    - Detects the date column (DATE or first column), coerces types, drops invalid rows,
      sorts by date, and returns a DataFrame indexed by datetime with the value column
      named as the file stem.
    """
    df = pd.read_csv(csv_path)
    # DATE detection
    dcol = None
    for c in df.columns:
        if str(c).upper() == "DATE":
            dcol = c
            break
    if dcol is None:
        # Fallback: first column
        dcol = df.columns[0]
    vcols = [c for c in df.columns if c != dcol]
    if not vcols:
        raise ValueError(f"No value column in {csv_path}")
    v = vcols[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df[v] = pd.to_numeric(df[v], errors="coerce")
    df = df.dropna(subset=[dcol, v]).set_index(dcol).sort_index()
    return df.rename(columns={v: Path(csv_path).stem})

def to_annual_from_quarterly_sum(df, colname):
    """Quarterly flow level to annual sum (approx for NGDP).

    Uses calendar year-end (Dec): 'YE-DEC'.
    """
    # Pandas 2.2+ deprecation: 'A-DEC' -> 'YE-DEC'
    a = df[colname].resample("YE-DEC").sum(min_count=1)
    return a.to_frame(name=colname)

def to_annual_from_q4(df, colname):
    """Quarterly stock level to annual year-end (Q4) level.

    Uses calendar year-end (Dec): 'YE-DEC'.
    """
    # 'A-DEC' -> 'YE-DEC' (year-end: December)
    a = df[colname].resample("YE-DEC").last()
    return a.to_frame(name=colname)

def safe_savefig(fig, path):
    """Save a Matplotlib figure safely (no overwrite) and return the saved Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure unique to avoid overwrite
    path = ensure_unique(path)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path

def figure_name_with_code(script_path: str|Path, original: Path) -> Path:
    """Return figure Path ensuring filename begins with the fig_code## identifier.

    Detects pattern 'fig_code<digits>' at start of script filename.
    If original already starts with that token, returns unchanged.
    """
    import re
    sp = Path(script_path).name  # e.g. fig_code12_dsr_....py
    m = re.match(r'(fig_code\d+)', sp)
    if not m:
        return original
    code_id = m.group(1)  # fig_code12
    stem = original.name
    if stem.startswith(code_id + '_'):
        return original
    return original.with_name(f"{code_id}_{stem}")

    def has_expected_prefix(script_path: str|Path, produced: Path) -> bool:
        """Return True if produced filename starts with the figure code derived from script_path.

        If script_path does not match fig_code<digits>, always returns True (no constraint).
        """
        import re
        sp = Path(script_path).name
        m = re.match(r'(fig_code\d+)', sp)
        if not m:
            return True
        return produced.name.startswith(m.group(1) + '_')


def ensure_unique(path: Path) -> Path:
    """Return a non-existing path by appending _1, _2, ... if needed."""
    path = Path(path)
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    i = 1
    while True:
        cand = path.with_name(f"{stem}_{i}{suffix}")
        if not cand.exists():
            return cand
        i += 1

def safe_to_csv(df: pd.DataFrame, path, index=False, **kwargs) -> Path:
    """Write CSV safely with suffixing to avoid overwrite. Returns final Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p = ensure_unique(p)
    df.to_csv(p, index=index, **kwargs)
    return p
