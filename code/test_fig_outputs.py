import os
import sys
import subprocess
from pathlib import Path
import re
import pytest
from util_code01_lib_io import figure_name_with_code, has_expected_prefix

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT.parent / 'figures'
PYTHON = sys.executable

# Helper to check prerequisite data exists; otherwise skip to keep CI fast/stable.
REQUIRED_DATA = [
    # Prefer prefixed proc_code03 outputs; tests skip if neither prefixed nor legacy exist.
    Path('data_processed/proc_code03_US_DSR_CreditGrowth_panel.csv'),
    Path('data_processed/proc_code03_JP_DSR_CreditGrowth_panel.csv'),
]

LEGACY_ALTERNATIVES = [
    Path('data_processed/US_DSR_CreditGrowth_panel.csv'),
    Path('data_processed/JP_DSR_CreditGrowth_panel.csv'),
]

def _have_required():
    # True if prefixed exist OR (all legacy exist).
    if all(p.exists() for p in REQUIRED_DATA):
        return True
    # fallback: both legacy
    return all(p.exists() for p in LEGACY_ALTERNATIVES)

@pytest.mark.skipif(not _have_required(), reason='required processed panel data missing')
def test_fig_code12_batch_outputs_created():
    prefix = FIG_DIR / 'TEST_AUTOBATCH'
    cmd = [PYTHON, 'code/fig_code12_dsr_creditgrowth_us_jp_dualpanels.py', '--batch', '--prefix', str(prefix)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT.parent)
    assert r.returncode == 0, f'process failed: {r.stderr}\nSTDOUT={r.stdout}'
    # Expected 4 variants
    expected_suffixes = ['DUAL_SHARED.png', 'DUAL_INDEPAX.png', 'US.png', 'JP.png']
    created = []
    for suf in expected_suffixes:
        f = FIG_DIR / f'fig_code12_{prefix.name}_{suf}'
        assert f.exists(), f'missing {f}'
        assert f.stat().st_size > 1000, f'file too small {f}'
        created.append(f)
    # Optionally store list for debugging
    print('Created (code12 batch):', created)

@pytest.mark.skipif(not _have_required(), reason='required processed panel data missing')
def test_fig_code13_timeseries_output_created():
    out = FIG_DIR / 'TMP_TS_AUTOTEST.png'
    if out.exists():
        out.unlink()
    cmd = [PYTHON, 'code/fig_code13_dsr_creditgrowth_us_jp_timeseries.py', '--out', str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT.parent)
    assert r.returncode == 0, f'process failed: {r.stderr}\nSTDOUT={r.stdout}'
    # Ensure prefix applied (function renames on save); locate actual file
    # It should become fig_code13_TMP_TS_AUTOTEST.png (or with numeric suffix if existed)
    pattern = re.compile(r'^fig_code13_TMP_TS_AUTOTEST.*\.png$')
    matches = [p for p in FIG_DIR.glob('fig_code13_TMP_TS_AUTOTEST*.png') if pattern.match(p.name)]
    assert matches, 'expected prefixed timeseries figure not found'
    # Use the newest file
    latest = max(matches, key=lambda p: p.stat().st_mtime)
    assert latest.stat().st_size > 1000, 'timeseries figure too small'
    print('Created (code13 ts):', latest)

def _recent_fig_files(limit=100):
    files = sorted(FIG_DIR.glob('*.png')) + sorted(FIG_DIR.glob('*.csv'))
    return files[-limit:]

def test_all_recent_outputs_have_prefix():
    # Heuristic: every file whose root matches fig_codeXX script output should start with that prefix
    # We only assert for figure patterns we know (fig_code\d+_)
    bad = []
    for f in _recent_fig_files():
        name = f.name
        if re.match(r'^fig_code\d+_', name):
            continue  # already good
        # Allow some exclusion list (non-figure assets) if necessary
        if name.lower().startswith(('readme','manifest')):
            continue
        # Flag as bad
        bad.append(name)
    assert not bad, f"Files missing prefix: {bad}"
