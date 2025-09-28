"""
Deprecated: Use fig_code02_mbs_to_gdp.py

This script previously rendered a dual-axis figure from an already-assembled CSV.
As of 2025-09-28, fig_code02 handles both ETL and plotting with consistent labels/formatting.
Keeping this file only to avoid breaking callers; it prints a deprecation message and exits.
"""
import sys

def main():
    print("[DEPRECATED] fig_code03_mbs_to_gdp_dual_axis.py -> Use code/fig_code02_mbs_to_gdp.py instead.")
    print("[DEPRECATED] Skipping execution. The orchestrator also skips fig_code03.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
