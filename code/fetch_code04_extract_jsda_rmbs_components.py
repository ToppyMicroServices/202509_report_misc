#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy alias wrapper for JSDA RMBS component extraction.

This delegates to `fetch_code03_extract_jsda_rmbs_components.py` to preserve
older invocation patterns:

  python code/fetch_code04_extract_jsda_rmbs_components.py --input <csv> --out <csv>

Use --help on the underlying module for full options.
"""
from __future__ import annotations
import sys

def main(argv=None) -> int:
    try:
        import fetch_code03_extract_jsda_rmbs_components as m03
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[ERROR] cannot import fetch_code03_extract_jsda_rmbs_components: {e}\n")
        return 1
    return m03.main(argv)

if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
