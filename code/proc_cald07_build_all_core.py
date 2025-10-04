#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
proc_cald07_build_all_core.py

Alias wrapper for the unified core build orchestrator.
User requested an additional entry-point name (likely for backward
compatibility or external automation) pointing to
`proc_code07_build_all_core_series.py`.

Behavior:
  - Simply imports the real orchestrator and delegates to its main().
  - All CLI arguments are passed through unchanged.

Usage:
  python code/proc_cald07_build_all_core.py --quiet

This file is intentionally lightweight so future logic should be added only
in the canonical script `proc_code07_build_all_core_series.py`.
"""
from __future__ import annotations
import sys

try:
    from proc_code07_build_all_core_series import main as _core_main
except Exception as e:  # pragma: no cover
    print(f"[proc_cald07][ERR] failed to import orchestrator: {e}")
    sys.exit(1)

if __name__ == '__main__':  # pragma: no cover
    sys.exit(_core_main())
