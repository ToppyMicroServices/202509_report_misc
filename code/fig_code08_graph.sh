#!/usr/bin/env bash
# fig_code08_graph.sh
# Master orchestrator: generate ALL figure outputs (fig_code01–13) with sensible defaults.
# Safe to re-run; individual Python scripts already avoid overwriting (ensure_unique / figure_name_with_code).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # release root
CODE_DIR="$ROOT_DIR/code"
FIG_DIR="$ROOT_DIR/figures"
PROC_DIR="$ROOT_DIR/data_processed"
RAW_DIR="$ROOT_DIR/data_raw"

mkdir -p "$FIG_DIR"

log(){ printf "\n[FIG08][%s] %s\n" "$1" "$2"; }
run(){ log RUN "$*"; if ! "$@"; then log FAIL "$*"; return 1; fi; }

SUMMARY_OK=()
SUMMARY_SKIP=()
SUMMARY_ERR=()
# Execution log (robust against any array / subshell quirks)
EXEC_LOG="$(mktemp 2>/dev/null || echo /tmp/fig_exec_log_$$)"
# Defensive: ensure arrays are treated as existing even if empty when expanding with @ under set -u
:${SUMMARY_OK[@]:-}
:${SUMMARY_SKIP[@]:-}
:${SUMMARY_ERR[@]:-}

mark_ok(){ SUMMARY_OK+=("$1"); echo "$1" >>"$EXEC_LOG"; }
mark_skip(){ SUMMARY_SKIP+=("$1: $2"); echo "$1" >>"$EXEC_LOG"; }
mark_err(){ SUMMARY_ERR+=("$1: $2"); echo "$1" >>"$EXEC_LOG"; }

exists(){ [[ -f "$1" ]]; }

########## fig_code01 ##########
if run python "$CODE_DIR/fig_code01_jsda_components.py"; then mark_ok fig_code01; else mark_err fig_code01 run_failed; fi

########## fig_code02 ##########
if run python "$CODE_DIR/fig_code02_mbs_to_gdp.py"; then mark_ok fig_code02; else mark_err fig_code02 run_failed; fi

########## fig_code03 ##########
if run python "$CODE_DIR/fig_code03_mbs_to_gdp_dual_axis.py"; then mark_ok fig_code03; else mark_err fig_code03 run_failed; fi

########## build annual MBS/RMBS ratios (proc_code05) ##########
if run python "$CODE_DIR/proc_code05_build_annual_mbs_rmbs_ratios.py"; then
  :
else
  log WARN "proc_code05_build_annual_mbs_rmbs_ratios failed (legacy annual ratio CSVs may remain)"
fi

########## build JP JHF RMBS vs GDP (proc_code06) ##########
if run python "$CODE_DIR/proc_code06_build_jp_jhf_rmbs_vs_gdp.py"; then
  :
else
  log WARN "proc_code06_build_jp_jhf_rmbs_vs_gdp failed (fig_code09 may fallback)"
fi

########## build phi quarterly (proc_code02) ##########
if run python "$CODE_DIR/proc_code02_build_phi_quarterly.py"; then
  :
else
  log WARN "proc_code02_build_phi_quarterly failed"
fi

########## fig_code04 ##########
PHI_Q_CSV="$PROC_DIR/proc_code02_JP_phi_quarterly.csv"
if ! exists "$PHI_Q_CSV" && exists "$PROC_DIR/JP_phi_quarterly.csv"; then
  PHI_Q_CSV="$PROC_DIR/JP_phi_quarterly.csv"  # legacy fallback
  log WARN "Using legacy JP_phi_quarterly.csv (deprecated; prefer proc_code02_JP_phi_quarterly.csv; removal target 2025Q4)"
fi
if exists "$PHI_Q_CSV"; then
  # fig_code04 (minimal) uses --jp / --us ; provide explicit JP path
  if run python "$CODE_DIR/fig_code04_phi_offbalance.py" --jp "$PHI_Q_CSV"; then mark_ok fig_code04; else mark_err fig_code04 run_failed; fi
else
  mark_skip fig_code04 "missing proc_code02_JP_phi_quarterly.csv (and legacy JP_phi_quarterly.csv)"
fi

########## fig_code05 (needs CSV + out) ##########
# Auto-detect a multi-country phi CSV (>=2 phi columns). Fallback to JP_phi_quarterly.
detect_phi_csv(){
  python - <<'PY'
import os, pandas as pd, json
base='data_processed'
best=None
for fn in sorted(os.listdir(base)):
    if 'phi' not in fn.lower() or not fn.lower().endswith('.csv'): continue
    path=os.path.join(base, fn)
    try:
        df=pd.read_csv(path, nrows=3)
    except Exception:
        continue
    cols=[c for c in df.columns if 'phi' in c.lower()]
    if len(cols)>=2 and ('date' in [c.lower() for c in df.columns]):
        best=path; break
if not best and os.path.exists(f'{base}/proc_code02_JP_phi_quarterly.csv'): best=f'{base}/proc_code02_JP_phi_quarterly.csv'
if not best and os.path.exists(f'{base}/JP_phi_quarterly.csv'): best=f'{base}/JP_phi_quarterly.csv'
print(best or '')
PY
}
PHI_CSV="$(detect_phi_csv)"
if [[ -n "$PHI_CSV" ]]; then
  case "$(basename "$PHI_CSV")" in
    JP_phi_quarterly.csv|phi_US_quarterly.csv|US_phi_quarterly.csv|JP_phi_from_JSDA_cleaned_quarterly_DEDUP.csv)
      log WARN "Using legacy phi file $(basename "$PHI_CSV") (deprecated; prefer proc_code02_/proc_code04_ prefixed equivalents; removal target 2025Q4)" ;;
  esac
fi
if [[ -n "$PHI_CSV" ]]; then
  if run python "$CODE_DIR/fig_code05_phi_offbalance_simple.py" --csv "$PHI_CSV" --out "$FIG_DIR/JP_US_phi_simple.png"; then mark_ok fig_code05; else mark_err fig_code05 run_failed; fi
else
  mark_skip fig_code05 "no phi csv detected"
fi

########## fig_code06 ##########
## Build DSR panels (proc_code03) to ensure prefixed panel CSVs exist
if run python "$CODE_DIR/proc_code03_build_dsr_creditgrowth_panels.py"; then
  :
else
  log WARN "proc_code03_build_dsr_creditgrowth_panels failed (will rely on legacy panels if present)"
fi
# Prefer prefixed US panel when present; fig_code06 auto-fallback remains inside.
if run python "$CODE_DIR/fig_code06_scatter_dsr_creditgrowth.py" --panel-csv "$PROC_DIR/proc_code03_US_DSR_CreditGrowth_panel.csv" --panel; then mark_ok fig_code06; else mark_err fig_code06 run_failed; fi

########## fig_code07 (needs k_decomposition CSV) ##########
K_CSV_PREF="$PROC_DIR/proc_code01_k_decomposition_JP_quarterly.csv"
K_CSV_LEG="$PROC_DIR/k_decomposition_JP_quarterly.csv"
if ! exists "$K_CSV_PREF" && exists "$K_CSV_LEG"; then
  # adopt legacy as working copy name if prefixed missing
  cp "$K_CSV_LEG" "$K_CSV_PREF" || true
fi
if exists "$K_CSV_PREF"; then
  if run python "$CODE_DIR/proc_code01_trim_k_decomp.py" --csv "$K_CSV_PREF" --eps 0; then :; else log WARN "k_decomposition trim step failed (continuing)"; fi
  if run python "$CODE_DIR/fig_code07_k_decomp_generic.py" \
      --csv "$K_CSV_PREF" \
      --out "$FIG_DIR/K_DECOMP_JP_generic.png" \
      --end-year 2025 --end-quarter 2 --debug-save-trimmed-csv; then
    mark_ok fig_code07
  else
    mark_err fig_code07 run_failed
  fi
else
  mark_skip fig_code07 "missing proc_code01_k_decomposition_JP_quarterly.csv (and legacy)"
fi

########## fig_code08 (this orchestrator) ##########
mark_ok fig_code08

########## fig_code09 ##########
# Prefer new proc_code06 output; fallback to legacy expectations
JHF_PROC06="$PROC_DIR/proc_code06_JP_JHF_RMBS_vs_GDP_quarterly.csv"
LEGACY_JHF="$PROC_DIR/JP_JHF_RMBS_vs_GDP.csv"
if exists "$JHF_PROC06"; then
  if run python "$CODE_DIR/fig_code09_jp_jhf_rmbs_vs_gdp.py"; then mark_ok fig_code09; else mark_err fig_code09 run_failed; fi
elif exists "$LEGACY_JHF" && exists "$RAW_DIR/JPNNGDP.csv"; then
  if run python "$CODE_DIR/fig_code09_jp_jhf_rmbs_vs_gdp.py"; then mark_ok fig_code09; else mark_err fig_code09 run_failed; fi
else
  mark_skip fig_code09 "missing proc_code06_JP_JHF_RMBS_vs_GDP_quarterly.csv (or legacy + JPNNGDP.csv)"
fi

########## fig_code10 ##########
# Build US phi prefixed processed (bank_off) before verification figure
if run python "$CODE_DIR/proc_code04_build_us_phi_series.py"; then :; else log WARN "proc_code04_build_us_phi_series failed"; fi
# Pre-check presence of required raw series for fig_code10 to give clearer UX
if [[ ! -f "$RAW_DIR/AGSEBMPTCMAHDFS.csv" || ! -f "$RAW_DIR/HHMSDODNS.csv" ]]; then
  log WARN "fig_code10 prerequisites missing (need data_raw/AGSEBMPTCMAHDFS.csv and HHMSDODNS.csv); will skip gracefully."
fi
if OUTPUT=$(python "$CODE_DIR/fig_code10_verify_us_phi_sources.py" --no-download 2>&1); then
  if grep -q "\[SKIP\]" <<<"$OUTPUT"; then
    mark_skip fig_code10 'no local series (skip)'
    printf "[FIG08][INFO] fig_code10 skipped (no local data)\n"
  else
    mark_ok fig_code10
  fi
  printf "%s\n" "$OUTPUT"
else
  printf "%s\n" "$OUTPUT"
  mark_err fig_code10 run_failed
fi

########## (Former) fig_code11 scripts ##########
# Deprecated: composite off-balance φ and experimental JHF RMBS vs GDP.
# Replaced by:
#   - fig_code04_phi_offbalance_min.py  (final 2-series φ figure)
#   - fig_code09_jp_jhf_rmbs_vs_gdp.py (official JHF RMBS vs GDP figure)
mark_skip fig_code11 "deprecated (use fig_code04 min + fig_code09)"

########## fig_code12 (auto batch when no args) ##########
US_PANEL="$PROC_DIR/proc_code03_US_DSR_CreditGrowth_panel.csv"
JP_PANEL="$PROC_DIR/proc_code03_JP_DSR_CreditGrowth_panel.csv"
if [[ ! -f "$US_PANEL" && -f "$PROC_DIR/US_DSR_CreditGrowth_panel.csv" ]]; then US_PANEL="$PROC_DIR/US_DSR_CreditGrowth_panel.csv"; fi
if [[ ! -f "$JP_PANEL" && -f "$PROC_DIR/JP_DSR_CreditGrowth_panel.csv" ]]; then JP_PANEL="$PROC_DIR/JP_DSR_CreditGrowth_panel.csv"; fi
if run python "$CODE_DIR/fig_code12_dsr_creditgrowth_us_jp_dualpanels.py" --us-csv "$US_PANEL" --jp-csv "$JP_PANEL"; then mark_ok fig_code12; else mark_err fig_code12 run_failed; fi

########## fig_code13 (supply a dummy arg to suppress auto show) ##########
if run python "$CODE_DIR/fig_code13_dsr_creditgrowth_us_jp_timeseries.py" --us-csv "$US_PANEL" --jp-csv "$JP_PANEL" --start 1990; then mark_ok fig_code13; else mark_err fig_code13 run_failed; fi

########## fig_code16 ##########
# Requires either FRED_API_KEY for online fetch or local raw CSVs under data_raw/ for WALCL/WSHOMCB and BIS series.
if run python "$CODE_DIR/fig_code16_fred.py"; then mark_ok fig_code16; else mark_err fig_code16 run_failed; fi

echo "\n================ FIGURE GENERATION SUMMARY ================"
printf "OK   : %s\n" "${SUMMARY_OK[*]:-}" || true
if [ ${#SUMMARY_SKIP[@]:-0} -gt 0 ] 2>/dev/null; then
  printf "SKIP :\n"; for s in "${SUMMARY_SKIP[@]}"; do echo "  - $s"; done
fi
if [ ${#SUMMARY_ERR[@]:-0} -gt 0 ] 2>/dev/null; then
  printf "ERRORS:\n"; for s in "${SUMMARY_ERR[@]}"; do echo "  - $s"; done
  exit 1
fi
echo "\n[CHECK] Verifying fig_code prefix on generated PNG/CSV outputs..."
prefix_issues=0
while IFS= read -r f; do
  base="$(basename "$f")"
  # Allow a legacy diagnostic file (kept temporarily) without prefix to avoid noisy warning
  if [[ "$base" == JP_US_phi_offbalance_quarterly_* ]]; then continue; fi
  [[ "$base" =~ ^fig_code[0-9]+_ ]] || { echo "  [WARN] missing prefix: $base"; prefix_issues=$((prefix_issues+1)); }
done < <(find "$FIG_DIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.csv' \) -mtime -1)
if (( prefix_issues > 0 )); then
  echo "[CHECK] Prefix issues detected: $prefix_issues (see warnings above)"; else echo "[CHECK] All recent files have expected prefix."; fi

echo "[CHECK] Verifying no figure scripts were omitted..."
# Build a space-delimited set with leading/trailing spaces for exact membership checks.
exec_ids=""
for id in "${SUMMARY_OK[@]:-}"; do exec_ids+=" $id"; done
for s in "${SUMMARY_SKIP[@]:-}"; do exec_ids+=" ${s%%:*}"; done
for s in "${SUMMARY_ERR[@]:-}"; do exec_ids+=" ${s%%:*}"; done
exec_ids+=" "
echo "[DEBUG] Executed IDs:$exec_ids"
missing=0
for path in "$CODE_DIR"/fig_code0*.py; do
  [ -e "$path" ] || continue
  bn="$(basename "$path")"
  base_id=""
  if [[ $bn =~ ^(fig_code[0-9]+)_ ]]; then
    base_id="${BASH_REMATCH[1]}"
  fi
  [ -z "$base_id" ] && continue
  [ "$base_id" = "fig_code08" ] && continue
  case "$exec_ids" in
    *" $base_id "*) : ;;  # present
    *) echo "  [WARN] figure script not executed by orchestrator: $bn"; missing=$((missing+1));;
  esac
done
rm -f "$EXEC_LOG" 2>/dev/null || true
if [ $missing -eq 0 ]; then
  echo "[CHECK] All fig_code scripts (excluding fig_code08) were invoked (or explicitly skipped)."; else echo "[CHECK] Missing figure scripts detected: $missing"; fi
echo "All requested figure scripts processed."

exit 0
