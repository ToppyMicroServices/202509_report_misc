# Release Package (20250914_1100) – US–JP Credit Externalization Study (English Only)

This repository contains the reproducible code, processed data, and figure outputs used in the US–Japan credit externalization analysis.

## 1. Directory Overview
- `code/` – fetch / processing / figure / verification scripts.
- `data_raw/` – raw source files (re-downloadable). Missing files can be restored under the same name.
- `data_processed/` – deterministic derived series (panels, ratios, decompositions).
- `figures/` – final figures (some auto-suffixed to avoid overwrite).
- `fig_code16/` – outputs for code16 figures (US credit creation shift, Fed support, NBFI parallel, φ vs Fed MBS/Total).
- `manifests/` – machine-readable integrity / provenance metadata.
- `docs/` – documentation (this README, method notes if any).
- `downloads*/` – transient download staging (regenerable).

## 2. Reproducibility & Provenance
- All figures are programmatically generated (Python: pandas / numpy / matplotlib).
- **No generative AI was used to create or alter any images.**
- Re-run scripts after restoring any missing raw file under the original filename.
- Period covered (current release window): quarterly 2012Q1–2025Q?. Trailing zero-pad rows (if any) are trimmed at build.

## 3. Recent Key Changes (2025Q3 Refresh)
| Item | Change |
|------|--------|
| DSR panels | `proc_code03_build_dsr_creditgrowth_panels.py` now rebuilds JP/US panels and wide forms: `proc_code03_US_DSR_CreditGrowth_panel.csv`, `proc_code03_JP_DSR_CreditGrowth_panel.csv`, plus `proc_code03_bis_DSR_wide.csv`, `proc_code03_bis_DSR_JP_US_panel.csv` |
| Annual MBS/RMBS ratios | `proc_code05_...` outputs stable annual ratio filenames (with legacy fallback) |
| Raw FRED caching | `fetch_code05_fred_us_mbs_gdp.py` explicitly saves `AGSEBMPTCMAHDFS`, `NGDPSAXDCUSQ`, `MORTGAGE30US` raw CSVs |
| US φ verification | `fig_code10_verify_us_phi_sources.py` produces raw vs adjusted series + diagnostic report |
| Orchestrator robustness | `fig_code08_graph.sh` omission detection fixed (regex base-id extraction) |
| φ auto-detect messaging | `fig_code05_phi_offbalance_simple.py` now prints column pattern guidance when US φ not found |
| Date column flexibility | Scripts accept both `DATE` and `observation_date` for time series inputs |

## 4. Naming / Prefix Conventions (2025Q3+)
| Type | Prefix | Example | Meaning |
|------|--------|---------|---------|
| Fetch scripts | `fetch_codeNN_` | `fetch_code07_imf_jp_gdp.py` | Acquisition (IMF / FRED / JSDA / JHF) |
| Processing scripts | `proc_codeNN_` | `proc_code03_build_dsr_creditgrowth_panels.py` | Deterministic pipeline stages |
| Figure scripts | `fig_codeNN_` | `fig_code06_scatter_dsr_creditgrowth.py` | Plot generation |
| Verification scripts | `fig_code10_...` etc. | `fig_code10_verify_us_phi_sources.py` | Diagnostic / verification (not primary figures) |
| Outputs | Same prefix as generator | `proc_code05_US_MBS_to_NGDP_annual.csv` | Traceable back to its script |

Rules:
1. Every output name begins with the generating script's prefix for traceability.
2. During migration, scripts attempt prefixed filename first, then legacy fallback if present.
3. Processed CSVs use stable names (no numeric suffix); figures may add suffixes (timestamps or labels).
4. First column in processed panels is explicit quarter-end timestamp (`to_period('Q').to_timestamp('Q')`).

## 5. DSR Panel Construction Logic (Summary)
JP DSR priority: (1) BIS PNFS → (2) legacy adoption → (3) φ-based proxy as a last resort. If a legacy panel exists, its CreditGrowth_ppYoY column is aligned and inherited; otherwise a 0.0 placeholder is used. US panel: BIS DSR + 4Q change using `HHMSDODNS`. `MetaVersion` and `ProxyFlag` trace the chosen path.

Run (always rebuilds):
```
python code/proc_code03_build_dsr_creditgrowth_panels.py
```

## 6. Data Fetch Overview
- JP nominal quarterly GDP (primary): `fetch_code07_imf_jp_gdp.py` (IMF IFS NGDP via SDMX) e.g. `--start 2012 --end 2025`.
- US MBS + GDP + raw caching: `fetch_code05_fred_us_mbs_gdp.py`.
- Generic single FRED series: `fetch_code08_fred_series.py --series <FRED_ID> --out <path>`.
- JSDA RMBS components: `fetch_code03_extract_jsda_rmbs_components.py`.
- JHF XLS→XLSX conversion: `fetch_code02_convert_jhf_xls_to_xlsx.py`.

## 7. Figure / Script Index
| # | Script | Description |
|---|--------|-------------|
| 01 | fig_code01_jsda_components.py | JSDA semiannual RMBS components |
| 02 | fig_code02_mbs_to_gdp.py | US vs JP MBS/RMBS to GDP (single axis) |
| 03 | fig_code03_mbs_to_gdp_dual_axis.py | Deprecated — use fig_code02_mbs_to_gdp.py |
| 04 | fig_code04_phi_offbalance.py | Detailed φ off-balance analysis |
| 05 | fig_code05_phi_offbalance_simple.py | Simplified φ comparison (US auto-detect guidance) |
| 06 | fig_code06_scatter_dsr_creditgrowth.py | DSR vs Credit Growth scatter + OLS |
| 07 | fig_code07_k_decomp_generic.py | Generic k decomposition plot |
| 08 | fig_code08_graph.sh | Batch quality / orchestrator script |
| 09 | fig_code09_jp_jhf_rmbs_vs_gdp.py | JP JHF RMBS vs GDP ratio |
| 10 | fig_code10_verify_us_phi_sources.py | US φ raw vs adjusted diagnostics |
| 11 | fig_code11_jp_jhf_rmbs_vs_gdp_experimental.py | Experimental extended variant |
| 12 | fig_code12_dsr_creditgrowth_us_jp_dualpanels.py | US/JP dual-panel DSR+CreditGrowth |
| 13 | fig_code13_dsr_creditgrowth_us_jp_timeseries.py | Time series comparison |
| 14 | fig_code14_scatter_phi_capital_thinness.py | φ vs capital thinness scatter |
| 15 | fig_code15_dsr_k_quantity_stack.py | DSR k-quantity stack visualization |
| 16 | fig_code16_fred.py | US bank-share (BIS), Fed balance sheet (nominal/normalized/share), NBFI parallel, and US φ vs Fed MBS/Total |

## 8. US φ Auto-Detection Guidance (fig_code05)
If the script finds a candidate file but no recognizable US φ column, it logs:
- Column pattern examples: `phi_US`, `phi_us`, `phi_US_adj`, `phi_us_pctgdp`, `phi_US_bank_off_raw`.
Resolution: (1) rename a column to one of the patterns, or (2) pass `--us-col <column_name>`.

Mandatory naming rule (enforced in-memory during plotting): φ series columns must contain an explicit country code token (`US` or `JP`). If a selected column lacks the token the script will duplicate-rename it in-memory (e.g. `phi` → `phi_US`) to prevent label inversion.

## 9. Dependencies / Environment
See `requirements.txt` (pandas, numpy, matplotlib, requests, openpyxl, fredapi). Minimal setup:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 10. Quick Reproduction Steps
```
# (Optional) fetch core GDP / MBS inputs
python code/fetch_code07_imf_jp_gdp.py --start 2012 --end 2025
python code/fetch_code05_fred_us_mbs_gdp.py

# Build core processed datasets
python code/proc_code03_build_dsr_creditgrowth_panels.py
python code/proc_code05_build_annual_mbs_rmbs_ratios.py

# Generate selected figures (examples)
python code/fig_code02_mbs_to_gdp.py
python code/fig_code05_phi_offbalance_simple.py
python code/fig_code16_fred.py  # writes under ./fig_code16 (requires FRED_API_KEY)

# Batch QC (runs multiple scripts and checks prefixes)
bash code/fig_code08_graph.sh
```

## 11. Quality / Integrity Checks
- `fig_code08_graph.sh` summarizes executed vs expected scripts and flags omissions.
- Use manifest hashes (if provided) in `manifests/` to confirm file integrity.
- Replace any missing raw file and rerun relevant processing scripts.

## 12. k Decomposition Notes
US k decomposition: weighted average maturity (WAM) held constant at 360 months; term contribution forced to 0 (see code comments). Units appear explicitly in filenames (`_pct`, `%`, or context labels).

## 13. Time Series Date Handling
Quarter alignment uses `to_period('Q').to_timestamp('Q')`. Scripts accept `DATE` or `observation_date` columns interchangeably and normalize them.

## 14. Legacy / Deprecated
- `scratch_code99_lab_cli.py` – lab CLI (scheduled for removal after 2025Q4).
- `code_old/` – historical reference only; no reliability guarantee.

## 15. FAQ (Selected)
Q: US φ column not detected? → Check warning list; rename a column or pass `--us-col`.
Q: Missing BIS DSR for JP? → Fallback sequence: legacy → φ proxy. For US, legacy/proxy path currently disabled (supply inputs then rerun).
Q: Extend the time window? → Re-fetch with updated `--end`, then rerun processing scripts.

## 16. data_raw/ Key Files and Sources
| File / Pattern | Source / API | Acquisition Method | Purpose |
|----------------|-------------|--------------------|---------|
| `AGSEBMPTCMAHDFS.csv` | FRED (AGSEBMPTCMAHDFS) | Auto-saved by `fetch_code05_fred_us_mbs_gdp.py` | US MBS Outstanding |
| `NGDPSAXDCUSQ.csv` | FRED (NGDPSAXDCUSQ) | Auto-saved (`fetch_code05`) | US Nominal GDP (quarterly) |
| `MORTGAGE30US.csv` | FRED (MORTGAGE30US) | Auto-saved (`fetch_code05`) | 30Y mortgage rate (context) |
| `HHMSDODNS.csv` | FRED (HHMSDODNS) | `fetch_code08_fred_series.py --series HHMSDODNS` | Household mortgage debt outstanding (US) |
| `RREACBM027NBOG.csv` / `RREACBM027SBOG.csv` | FRED | Same single-series fetch | US real estate loan components |
| `QJPPAM770A.csv` / `QUSPAM770A.csv` | FRED | Same | Price-to-income type ratios (if used) |
| `BOGZ1FL763065105Q.csv` | FRED | Same | Flow of Funds related series |
| `JPNNGDP.csv` | IMF IFS (NGDP) or ESRI fallback | `fetch_code07_imf_jp_gdp.py` (SDMX) | JP nominal GDP |
| `bis_dp_search_export_*.csv` | BIS Data Portal | Manual web export (retain timestamped filenames) | BIS DSR & debt service metrics |
| `JP_JHF_RMBS_vs_GDP.csv` | Internal / JSDA derived | Intermediate to `proc_code06...` | JHF RMBS ratio staging |
| `JP_TOTAL_RMBS_vs_GDP.xlsx` | JSDA publication | Manual download (no rename) | JP total RMBS outstanding |
| `nme_R031.*.csv` / `2024doukou*.xls` / `2025zandaka*.xls` | JHF | Manual XLS; converted by `fetch_code02_...` | JHF factor / balance inputs |
| `jhf_factors/*.xls` | JHF | Bulk manual save | φ construction raw factors |
| `jhf_factors_xlsx/*.xlsx` | Converted outputs | `fetch_code02_convert_jhf_xls_to_xlsx.py` | Normalized factor sheets |
| `CET1_RWA_JP_WB*.csv` / `CET1_RWA_US_WB.csv` | World Bank + manual curation | Combined via `fetch_code01_capital_k_data.py` | CET1 / RWA comparative inputs |
| `HMDA_2018_originated_nationwide.csv` | CFPB HMDA | Download national file (zip → extract) | Origination context |
| `raw_manifest.json` | Internal | Script-generated enumeration | Raw audit meta |
| `test_mortgage.csv` | Internal test | Manual placeholder | Local testing |

### Acquisition Notes
- FRED API (single series pattern): `https://api.stlouisfed.org/fred/series/observations?series_id=<ID>&api_key=<KEY>&file_type=json` (wrapped by fetch scripts → CSV).
- IMF IFS SDMX handled inside `fetch_code07_imf_jp_gdp.py` (no API key required).
- BIS portal exports: use GUI, keep original timestamped filename(s).
- JSDA / JHF: download official XLS exactly as published; conversion script handles naming collisions.

### Missing File Remediation
1. Re-download under the identical filename into `data_raw/`.
2. For API sources (FRED / IMF) re-run the relevant fetch script.
3. For BIS, re-export via portal (duplicates allowed).

---
README reflects repository state as of 2025-09-28. Future adjustments will appear in git history.

## 17. φ definition, construction, and structural break adjustment (US bank_off)

### 17.1 Definitions
JP φ (off‑balance share):
	JP φ = (JHF securitized pools + Private‑label RMBS pools) / (BoJ on‑balance + JHF securitized pools + Private‑label RMBS pools)
	i.e., (JHF + private) / (BoJ + JHF + private)

US φ (bank_off, “observed proxy”):
	US φ = Pools / (Loans_on + Pools)
	where Pools = securitized mortgage pools off bank balance sheets (FRED: AGSEBMPTCMAHDFS),
		  Loans_on = real‑estate loans on bank balance sheets (RREACBM027SBOG; fallback RREACBM027NBOG).

Both φ series are shares in [0,1]. Columns that look like φ but are ratios to GDP (…to_gdp, pctgdp, etc.) are automatically excluded with a warning and are not used in figures.

### 17.2 US bank_off construction (proc_code04)
1. Aggregate monthly FRED series to quarter‑end (last observation per quarter).
2. Normalize units to billions of USD via simple heuristics.
3. Compute φ_raw = Pools / (Pools + Loans_on).
4. Detect level breaks (structural shifts): search the largest |Δφ| after 1990‑01‑01; flag a break if
	   abs(jump) > max(0.05, 5 × median(|Δφ| over previous 8 quarters)).
5. If a break is flagged: compute scale = φ(break)/φ(prev) using the value just before and at the break, and multiply all pre‑break values by this scale to splice levels.
6. Output the adjusted series as `phi_US_bank_off_adj` alongside the raw (raw is preserved).

### 17.3 Adjustment metadata (sidecar)
If you pass `--break-meta-out <path>` to `proc_code04_build_us_phi_series.py`, the script writes a small text summary with:
	break_date, prev_date, jump, scale_pre_to_post, median_prev_abs_diff, applied=bool.
If no break is detected, applied=False and scale is NA.

### 17.4 Legend note in figures
`fig_code05_phi_offbalance_simple.py` annotates “adj” in the legend when the US series used is adjusted.

### 17.5 Reproducibility note
The adjustment is a simple level splice (multiplicative scale on the entire pre‑break segment). Because the diff statistics depend on the current raw series, rerunning after re‑fetching the same Pools/Loans sources reproduces the same scale and break_date in the sidecar meta.

### 17.6 Alternate φ (household share) and column naming
By request, in addition to the bank_off φ based on bank balance sheet + recently securitized flows, we also output a system‑wide off‑balance share using household mortgage totals (FRED: HHMSDODNS) as denominator.

Definition (household share):
	φ_US_household = Pools (+ PLMBS, currently ≈0) / HH_Mortgages
After unit normalization to billions of USD, we compute the same adjustment (scale) as bank_off for an aligned `*_adj` column.

Example columns (from proc_code04_US_phi_series_raw.csv):
	phi_US_bank_off_raw, phi_US_bank_off_adj,
	phi_US_household_share_raw, phi_US_household_share_adj,
	Pools_bilUSD, Loans_on_bilUSD, HH_mortgages_bilUSD

Important: the household share is conceptually “off‑balance portion / total household‑origin mortgages”. Since its denominator is larger than bank_off’s (Loans_on + Pools), the level is lower (≈0.20–0.30). Do not mix these definitions. If both are plotted on the same axis, make the bank_off series the headline and label the household series explicitly (e.g., “(household)”). `fig_code04_phi_offbalance_fixed.py` accepts `--us-household-col` to co‑plot both.

### 17.7 New figure (code16)
`fig_code16_fred.py` also produces “US: φ vs Fed MBS / Total credit (two‑axis)” under `fig_code16/us_phi_vs_fedmbs_totalcredit.png`.
Left axis is US φ; right axis is Fed MBS / total private credit. The script auto‑detects US φ from processed files (`proc_code04_US_phi_series_*.csv`) or accepts a simple `data_processed/us_phi.csv` with columns `[date, phi_us]`.


## 18. Zenodo linking (DOI)

This repository is prepared for Zenodo archiving to mint a DOI.

- Metadata files included:
	- `.zenodo.json` (title, creators, keywords, related identifiers)
	- `CITATION.cff` (citation metadata for GitHub UI and tools)

Steps to enable DOI (once the paper is ready or for a software-only DOI):
1. Sign in to Zenodo and connect your GitHub account (https://zenodo.org/account/settings/github/).
2. Toggle archiving for this repository: `ToppyMicroServices/202509_report_misc`.
3. Create a GitHub Release (e.g., `v2025.09`). Zenodo will archive that release and mint a DOI.
4. Copy the DOI badge from Zenodo and paste it at the top of this README.

Notes:
- If/when the paper DOI is available, add it into `.zenodo.json` under `related_identifiers` with relation `isSupplementTo` (or `isPartOf`/`isDocumentedBy` as appropriate).
- Avoid including huge raw datasets in the archived artifact; this repo already ignores large HMDA CSVs. Releases should contain code and small, deterministic processed data.


