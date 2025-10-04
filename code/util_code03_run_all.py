"""util_code03_run_all

Minimal batch orchestrator for selected ``fig_codeXX`` scripts.
Used mainly for quick regeneration during iterative drafting. For the full
paper run (including prefix / integrity checks) prefer the shell orchestrator
introduced later (see ``fig_code08_graph.sh``).
"""
from __future__ import annotations
import subprocess, sys

FIG_SCRIPTS = [
    'fig_code01_jsda_components.py',
    'fig_code02_mbs_to_gdp.py',
    'fig_code04_phi_offbalance.py',
    'fig_code05_phi_offbalance_simple.py',
    'fig_code06_scatter_dsr_creditgrowth.py',
    'fig_code07_k_decomp_generic.py',
    'fig_code09_jp_jhf_rmbs_vs_gdp.py',
    'fig_code10_verify_us_phi_sources.py',
]

def main(argv=None):
    failures = 0
    for script in FIG_SCRIPTS:
        print(f"[RUN] {script}")
        r = subprocess.run([sys.executable, script])
        if r.returncode != 0:
            print(f"[FAIL] {script} rc={r.returncode}")
            failures += 1
    print(f"[DONE] Figures complete (failures={failures})")
    return 0 if failures == 0 else 1

if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
