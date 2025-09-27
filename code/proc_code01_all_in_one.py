#!/usr/bin/env python3
"""proc_code01_all_in_one

One-shot runner for the entire proc_code01 flow with fixed paths and no options:
  1) Build JP WAC/WAM quarterly CSV if needed (from firstrow extract)
  2) Build k decomposition for US (rate-only) and JP (rate/term) with auto fallbacks
  3) Trim JP k decomposition (remove trailing all-zero block if any)

Fixed paths:
  - RAW US rate: data_raw/MORTGAGE30US.csv
  - JP firstrow: data_processed/jhf_wac_wam_firstrow_extract.csv
  - JP WAC/WAM : data_processed/jp_wac_wam_quarterly_estimated.csv
  - OUT US     : data_processed/proc_code01_k_decomposition_US_quarterly_RATE_ONLY.csv
  - OUT JP     : data_processed/proc_code01_k_decomposition_JP_quarterly.csv

This script does not accept CLI options; it is idempotent.
"""
from __future__ import annotations
from pathlib import Path
import sys, subprocess
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
RAW  = ROOT / 'data_raw'
PROC = ROOT / 'data_processed'

RAW_US = RAW / 'MORTGAGE30US.csv'
JP_FIRSTROW = PROC / 'jhf_wac_wam_firstrow_extract.csv'
JP_WACWAM   = PROC / 'jp_wac_wam_quarterly_estimated.csv'

OUT_US = PROC / 'proc_code01_k_decomposition_US_quarterly_RATE_ONLY.csv'
OUT_JP = PROC / 'proc_code01_k_decomposition_JP_quarterly.csv'

def ensure_wac_wam():
    """Create JP WAC/WAM if missing and firstrow is available."""
    if JP_WACWAM.exists():
        print(f"[STEP1][SKIP] WAC/WAM exists: {JP_WACWAM}")
        return True
    if not JP_FIRSTROW.exists():
        print(f"[STEP1][INFO] firstrow not found: {JP_FIRSTROW} (WAC/WAM build skipped)")
        return False
    # Import builder and run with fixed paths
    sys.path.insert(0, str(CODE))
    try:
        import proc_code01_build_jp_wac_wam_quarterly as wacwam
        ok = wacwam.build(JP_FIRSTROW, JP_WACWAM, include_t=False)
        if ok:
            print(f"[STEP1][OK] WAC/WAM built -> {JP_WACWAM}")
            return True
        else:
            print(f"[STEP1][WARN] import-build returned False; trying subprocess fallback")
    except Exception as e:
        print(f"[STEP1][WARN] import/build failed: {e}; trying subprocess fallback")
    # Fallback: run as a subprocess with fixed paths
    cmd = [sys.executable, str(CODE/'proc_code01_build_jp_wac_wam_quarterly.py')]
    cp = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if cp.returncode == 0 and JP_WACWAM.exists():
        print(f"[STEP1][OK] WAC/WAM built via subprocess -> {JP_WACWAM}")
        return True
    else:
        if cp.stdout.strip():
            print(cp.stdout.strip())
        if cp.stderr.strip():
            print(cp.stderr.strip())
        print(f"[STEP1][ERR] subprocess build failed rc={cp.returncode}")
        return False

def run_k_decomposition():
    """Run US/JP k decomposition via import first (robust), fallback to subprocess if needed."""
    # Try import-first execution
    try:
        sys.path.insert(0, str(CODE))
        import proc_code01_build_k_decomposition as kbuild  # type: ignore
        rc = kbuild.main()
        if rc == 0:
            print("[STEP2][OK] k decomposition completed (import)")
            return True
        else:
            print(f"[STEP2][WARN] import run rc={rc}; trying subprocess")
    except Exception as e:
        print(f"[STEP2][WARN] import failed: {e}; trying subprocess")
    # Fallback 1: run as a subprocess
    target = CODE / 'proc_code01_build_k_decomposition.py'
    if target.exists():
        cmd = [sys.executable, str(target)]
        cp = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if cp.returncode == 0:
            print("[STEP2][OK] k decomposition completed (subprocess)")
            if cp.stdout.strip():
                print(cp.stdout.strip())
            return True
        else:
            if cp.stdout.strip():
                print(cp.stdout.strip())
            if cp.stderr.strip():
                print(cp.stderr.strip())
            print(f"[STEP2][WARN] subprocess rc={cp.returncode}; trying importlib loader")
            # Fallback 2: load module directly from file path
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location('kbuild_mod', str(target))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                    rc = mod.main()
                    if rc == 0:
                        print("[STEP2][OK] k decomposition completed (importlib)")
                        return True
                    else:
                        print(f"[STEP2][ERR] importlib rc={rc}")
                        return False
                else:
                    print("[STEP2][ERR] importlib spec/loader not available")
                    return False
            except Exception as e:
                print(f"[STEP2][ERR] importlib loader failed: {e}")
                return False
    else:
        print(f"[STEP2][ERR] missing script: {target}")
        return False

def trim_jp(eps: float = 0.0):
    """Trim trailing all-zero block from JP k decomposition (if any)."""
    if not OUT_JP.exists():
        print(f"[STEP3][SKIP] JP output not found: {OUT_JP}")
        return False
    try:
        df = pd.read_csv(OUT_JP)
        # Consider zeros on these columns
        cols = [c for c in ('dk_rate','dk_term','dk_total') if c in df.columns]
        if not cols:
            print("[STEP3][SKIP] no dk_* columns to trim")
            return False
        def is_zero_row(row):
            return all(abs(row[c]) <= eps for c in cols)
        mask_nonzero = ~df.apply(is_zero_row, axis=1)
        if mask_nonzero.any():
            last_nz = mask_nonzero[::-1].idxmax()
            trimmed = df.iloc[:last_nz+1]
            if len(trimmed) < len(df):
                trimmed.to_csv(OUT_JP, index=False)
                print(f"[STEP3][OK] trimmed JP rows {len(df)} -> {len(trimmed)} (eps={eps})")
                return True
        print("[STEP3][INFO] no trailing zero block detected")
        return True
    except Exception as e:
        print(f"[STEP3][ERR] trim failed: {e}")
        return False

def main():
    PROC.mkdir(parents=True, exist_ok=True)
    # Step1: WAC/WAM (best-effort)
    ensure_wac_wam()
    # Step2: k decomposition (handles US+JP and JP fallbacks internally)
    ok2 = run_k_decomposition()
    # Step3: JP trim
    ok3 = trim_jp(eps=0.0)
    return 0 if ok2 else 1

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
