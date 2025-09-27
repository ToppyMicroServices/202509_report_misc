"""fig_code02_mbs_to_gdp

Construct annual MBS-to-GDP (US) and RMBS-to-GDP (Japan) ratios from raw / semi-processed
inputs and plot them on a dual y-axis chart (legacy computational version).

Responsibilities:
    1. Attempt to build US ratio from raw FRED series (MBS outstanding + nominal GDP).
    2. Attempt to build JP ratio from JSDA semiannual components + nominal GDP.
    3. If raw reconstruction fails, fall back to pre-computed processed CSVs.
    4. Restrict sample to 2012–2021 for comparability with publication figures.
    5. Produce a single dual-axis figure (US left axis, JP right axis) and save with
         fig_code02_ prefix ensured (via figure_name_with_code).

Notable differences vs fig_code03:
    - This script performs data assembly (ETL) whereas fig_code03 assumes a ready-made CSV.
    - No CLI yet (runs on import as a script) – could be extended similarly if needed.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from util_code01_lib_io import find, read_fred_two_col, to_annual_from_quarterly_sum, to_annual_from_q4, safe_savefig, ensure_unique, figure_name_with_code

OUTDIR = Path("figures")

# Preferred processed annual ratio CSVs (proc_code05) for optional shortcut / verification.
PREF_US_ANNUAL = Path('data_processed/proc_code05_US_MBS_to_NGDP_annual.csv')
PREF_JP_ANNUAL = Path('data_processed/proc_code05_JP_RMBS_to_NGDP_annual.csv')

# Flags to annotate legend labels with provenance if fast-path was used
US_SOURCE_PROC05 = False
JP_SOURCE_PROC05 = False

def build_us():
    """Build US annual MBS/GDP ratio from quarterly raw sources.

    Returns (DataFrame|None, error_message|None).
    DataFrame index: year-end (Dec 31) dates; columns: US_MBS_$bn, US_NGDP_$bn, US_MBS_to_NGDP_%.
    """
    # Fast-path: use prefixed processed if both ratio inputs absent (skip rebuild logic if available)
    global US_SOURCE_PROC05
    if PREF_US_ANNUAL.exists():
        try:
            dfp = pd.read_csv(PREF_US_ANNUAL)
            dcol = dfp.columns[0]
            dfp[dcol] = pd.to_datetime(dfp[dcol], errors='coerce')
            dfp = dfp.dropna(subset=[dcol]).set_index(dcol).sort_index()
            # Accept either existing ratio col or reconstruct if components present
            ratio_col = None
            for c in dfp.columns:
                if 'to' in c.lower() and 'gdp' in c.lower():
                    ratio_col = c; break
            if ratio_col:
                US_SOURCE_PROC05 = True
                return dfp.rename(columns={ratio_col:'US_MBS_to_NGDP_%'}), None
        except Exception:
            pass  # fall back to raw build
    mbs_files = find(["AGSEBMPTCMAHDFS.csv"])
    gdp_files = find(["NGDPSAXDCUSQ.csv","NGDPSAXDCUSQ*.csv"])
    if not mbs_files or not gdp_files:
        return None, "US inputs missing"
    mbs = read_fred_two_col(mbs_files[0])
    col_m = mbs.columns[0]
    mbs[col_m] = mbs[col_m] / 1_000.0
    gdp = read_fred_two_col(gdp_files[0])
    col_g = gdp.columns[0]
    gdp_a = to_annual_from_quarterly_sum(gdp, col_g).rename(columns={col_g:"US_NGDP_$bn"})
    mbs_a = to_annual_from_q4(mbs, col_m).rename(columns={col_m:"US_MBS_$bn"})
    df = mbs_a.join(gdp_a, how="inner")
    if df.empty: return None, "no US overlap"
    df["US_MBS_to_NGDP_%"] = 100.0 * df["US_MBS_$bn"] / df["US_NGDP_$bn"]
    return df, None

def _read_processed_ratio(file_path: Path, col_hint: str, new_col: str):
    """Generic helper for fallback processed ratio CSV.

    Tries to locate a column containing col_hint (case-insensitive); otherwise applies heuristics
    preferring columns with 'pct'. Returns a tidy DateTime-indexed DataFrame with new_col.
    """
    df = pd.read_csv(file_path)
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    value_cols = [c for c in df.columns if c != date_col]
    pick = None
    for c in value_cols:
        if col_hint.lower() in c.lower():
            pick = c; break
    if pick is None:
        if len(value_cols) == 1:
            pick = value_cols[0]
        else:
            pct_cols = [c for c in value_cols if "pct" in c.lower()]
            pick = pct_cols[0] if pct_cols else value_cols[0]
    s = pd.to_numeric(df[pick], errors="coerce")
    out = pd.DataFrame({new_col: s})
    out[date_col] = df[date_col]
    out = out.dropna(subset=[date_col]).set_index(date_col).sort_index()
    out = out.dropna(subset=[new_col])
    return out

def build_us_fallback_processed():
    """Fallback: read pre-computed US ratio if raw assembly failed."""
    files = find(["US_MBS_to_AnnualGDP.csv"])
    if not files:
        return None, "no US processed"
    try:
        df = _read_processed_ratio(files[0], col_hint="US_MBS_to_GDP", new_col="US_MBS_to_NGDP_%")
        return df, None
    except Exception as e:
        return None, f"US processed read error: {e}"


def build_jp():
    """Build JP annual RMBS/GDP ratio from JSDA semiannual components + nominal GDP.

    Steps:
      - Read JSDA components (semiannual) and identify either a total column or sum JHF + Private.
      - Convert units (hundred million yen -> billion yen) via *0.1.
      - Resample to quarter-end then to annual (Dec) last value for stocks; sum GDP for annual flow.
      - Compute JP_RMBS_to_NGDP_%.
    Returns (DataFrame|None, error_message|None).
    """
    # Fast-path: use prefixed processed if present
    global JP_SOURCE_PROC05
    if PREF_JP_ANNUAL.exists():
        try:
            dfp = pd.read_csv(PREF_JP_ANNUAL)
            dcol = dfp.columns[0]
            dfp[dcol] = pd.to_datetime(dfp[dcol], errors='coerce')
            dfp = dfp.dropna(subset=[dcol]).set_index(dcol).sort_index()
            ratio_col = None
            for c in dfp.columns:
                if 'to' in c.lower() and 'gdp' in c.lower():
                    ratio_col = c; break
            if ratio_col:
                JP_SOURCE_PROC05 = True
                return dfp.rename(columns={ratio_col:'JP_RMBS_to_NGDP_%'}), None
        except Exception:
            pass
    jsda_files = find(["JP_RMBS_components_semiannual_JSDA_2012_2025.csv"])
    gdp_files = find(["JPNNGDP.csv"])
    if not jsda_files or not gdp_files:
        return None, "JP inputs missing"
    jsda = pd.read_csv(jsda_files[0])
    jsda.columns = [c.strip().lower() for c in jsda.columns]
    dtc = [c for c in jsda.columns if "date" in c or "month" in c]
    if not dtc:
        return None, "no date col in JSDA"
    dtc = dtc[0]
    cand_total = [c for c in jsda.columns if "total" in c and ("100" in c or "yen" in c)]
    if cand_total:
        total_col = cand_total[0]
        jsda_total = jsda[[dtc, total_col]].copy().rename(columns={total_col: "rmbs_total_100m_yen"})
    else:
        jhf = [c for c in jsda.columns if "jhf" in c]
        prv = [c for c in jsda.columns if "priv" in c]
        if not jhf or not prv:
            return None, "no JHF/Private cols"
        jsda["rmbs_total_100m_yen"] = jsda[jhf[0]] + jsda[prv[0]]
        jsda_total = jsda[[dtc, "rmbs_total_100m_yen"]]
    jsda_total[dtc] = pd.to_datetime(jsda_total[dtc], errors="coerce")
    jsda_total = jsda_total.dropna(subset=[dtc]).set_index(dtc).sort_index()
    rmbs_bny = (jsda_total["rmbs_total_100m_yen"] * 0.1).to_frame(name="JP_RMBS_bny")
    rmbs_q = rmbs_bny.resample("QE-DEC").ffill()
    jp_gdp_q = pd.read_csv(gdp_files[0])
    jd = [c for c in jp_gdp_q.columns if "date" in c.lower()][0]
    val = [c for c in jp_gdp_q.columns if c != jd][0]
    jp_gdp_q[jd] = pd.to_datetime(jp_gdp_q[jd], errors="coerce")
    jp_gdp_q = jp_gdp_q.dropna(subset=[jd]).rename(columns={jd: "DATE", val: "JP_NGDP_bny"}).set_index("DATE").sort_index()
    gdp_a = jp_gdp_q["JP_NGDP_bny"].resample("YE-DEC").sum(min_count=1).to_frame()
    rmbs_a = rmbs_q.resample("YE-DEC").last()
    df = rmbs_a.join(gdp_a, how="inner")
    if df.empty:
        return None, "no JP overlap"
    df["JP_RMBS_to_NGDP_%"] = 100.0 * df["JP_RMBS_bny"] / df["JP_NGDP_bny"]
    return df, None

def build_jp_fallback_processed():
    """Fallback: read pre-computed JP ratio if raw assembly failed."""
    files = find(["JP_RMBS_to_AnnualGDP.csv"])
    if not files:
        return None, "no JP processed"
    try:
        df = _read_processed_ratio(files[0], col_hint="JP_RMBS_to_GDP", new_col="JP_RMBS_to_NGDP_%")
        return df, None
    except Exception as e:
        return None, f"JP processed read error: {e}"

def make():
    """Orchestrate build steps and produce the dual-axis figure.

    Handles fallback logic, sample truncation, plotting, and safe file save.
    Prints status messages describing which data source path was used.
    """
    us, e1 = build_us()
    jp, e2 = build_jp()
    if us is None:
        us, e1b = build_us_fallback_processed()
        if us is None:
            print("[MBS->GDP] skip:", e1, e1b); return
        else:
            print("[MBS->GDP] use processed US ratio")
    if jp is None:
        jp, e2b = build_jp_fallback_processed()
        if jp is None:
            print("[MBS->GDP] skip:", e2, e2b); return
        else:
            print("[MBS->GDP] use processed JP ratio")
    df = us.join(jp, how="inner")
    if not df.empty:
        df = df[(df.index.year>=2012) & (df.index.year<=2021)]
    if df.empty:
        print("[MBS->GDP] skip: no overlap 2012-2021"); return
    fig, ax1 = plt.subplots(figsize=(7,4))
    us_label = "US MBS/GDP (%)" + (" (proc_code05)" if US_SOURCE_PROC05 else "")
    ax1.plot(df.index.year, df["US_MBS_to_NGDP_%"], marker="o", lw=2.0, label=us_label, color="#E69F00")
    ax1.set_ylabel("US: % of GDP")
    ax2 = ax1.twinx()
    jp_label = "Japan RMBS/GDP (%)" + (" (proc_code05)" if JP_SOURCE_PROC05 else "")
    ax2.plot(df.index.year, df["JP_RMBS_to_NGDP_%"], marker="s", lw=2.0, label=jp_label, color="#56B4E9")
    ax2.set_ylabel("JP: % of GDP")
    ax1.set_title("MBS-to-GDP (US) vs RMBS-to-GDP (JP), annual")
    ax1.grid(True, alpha=0.3)
    lines = ax1.get_lines()+ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    target = figure_name_with_code(__file__, OUTDIR/"JP_US_MBS_to_AnnualGDP_2012_2021_DUAL_AXIS.png")
    unique = ensure_unique(target)
    safe_savefig(fig, unique)
    print("WROTE:", unique)

if __name__=="__main__":
    make()
