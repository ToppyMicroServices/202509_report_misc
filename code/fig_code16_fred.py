"""
US Credit Creation Shift: Banks vs Nonbanks (evidence figures)

What this script makes (PNG under ./fig_code16):
    1) us_bank_share_private_credit.png
         = (BIS: Credit to private non‑financial sector by banks) / (BIS: Total credit to private non‑financial sector)
         + vertical line at 2010Q1 (ASC 860 effective) for reference

    2) us_jp_bank_share_compare.png
         = US vs Japan bank‑share of private credit (same construction as #1)

    3) fed_support_qe_mbs.png
         = Fed total assets (WALCL) and MBS holdings (WSHOMCB), normalized
    3b) fed_support_qe_mbs_nominal.png
        = Nominal levels (USD trillions)
    3c) fed_mbs_share_of_total.png
        = MBS as share of total Fed assets

    4) us_bankshare_vs_nbficommit.png
        = US bank share vs NBFI commitments (step/ffill)

    5) us_phi_vs_fedmbs_totalcredit.png
        = US φ vs Fed MBS / Total credit (two-axis)

Dependencies:
    pip install fredapi pandas matplotlib python-dateutil
    # optional: pip install seaborn  (not used)

FRED API key:
    export FRED_API_KEY=YOUR_KEY
    (Key is free: https://fred.stlouisfed.org/docs/api/api_key.html)

Data sources (series codes, preferred in USD level for numerator/denominator consistency):
    - US total private credit (BIS via FRED):      QUSPAMUSDA (preferred), fallback: CRDQUSAPABIS
    - US bank credit to private sector (BIS/FRED): QUSPBMUSDA (preferred), fallback: QUSPBM770A
    - JP total private credit (BIS/FRED):          QJPPAMUSDA
    - JP bank credit to private sector:            QJPPBMUSDA
    - Fed total assets (H.4.1):                    WALCL (weekly)
    - Fed MBS holdings (H.4.1):                    WSHOMCB (weekly)

Notes:
    * We compute bank_share = (bank_credit_private) / (total_private_credit) in levels.
    * Quarterly BIS series are not seasonally adjusted; we align by period end.
    * H.4.1 series are weekly; we resample to quarter‑end mean for comparability.
    * ASC 860 (2010Q1) line is a visual reference only.
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from matplotlib.ticker import ScalarFormatter, PercentFormatter

# ---------- Setup ----------
FRED_API_KEY = os.getenv("FRED_API_KEY")
fred = None
if FRED_API_KEY:
    fred = Fred(api_key=FRED_API_KEY)



OUTDIR = "fig_code16"
os.makedirs(OUTDIR, exist_ok=True)

# Where to store/load raw FRED CSVs (DATE,VALUE). Prefer 'raw_data' if present; otherwise 'data_raw'.
RAW_OUTDIR = "raw_data" if os.path.isdir("raw_data") else "data_raw"
os.makedirs(RAW_OUTDIR, exist_ok=True)

# Save a raw FRED series as DATE,VALUE CSV (no resampling)
def save_fred_raw(series_id: str, outdir: str = RAW_OUTDIR):
    if fred is None:
        return
    try:
        s = fred.get_series(series_id)
    except Exception:
        return None
    df = pd.DataFrame({
        'DATE': pd.DatetimeIndex(s.index),
        'VALUE': pd.to_numeric(pd.Series(s.values), errors='coerce'),
    })
    df = df[['DATE','VALUE']]
    out_path = os.path.join(outdir, f"{series_id}.csv")
    df.to_csv(out_path, index=False)
    return out_path

def save_fred_raw_batch(series_ids):
    saved = []
    for sid in series_ids:
        p = save_fred_raw(sid, RAW_OUTDIR)
        if p:
            saved.append(p)
    if saved:
        print("Saved raw series to:")
        for p in saved:
            print("  ", p)

# --- Plot caption helper ---
def add_caption(ax, text: str, *, loc: str = 'top-left', fontsize: int = 7):
    """Add a small caption inside the axes with a subtle background.

    loc options: 'top-left', 'bottom-left'
    """
    if loc == 'bottom-left':
        x, y = 0.01, 0.02
        ha, va = 'left', 'bottom'
    else:  # default 'top-left'
        x, y = 0.01, 0.98
        ha, va = 'left', 'top'
    return ax.text(
        x, y, text,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha=ha, va=va,
        alpha=0.9,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.65, edgecolor='none'),
        wrap=True,
    )

# Helpers that try multiple candidate series IDs (for consistency of units)
def fred_q_any(series_ids: list[str]) -> pd.Series:
    if fred is None:
        raise RuntimeError("fred_q_any called without FRED_API_KEY.")
    last_err = None
    for sid in series_ids:
        try:
            s = fred.get_series(sid)
            s.index = pd.DatetimeIndex(s.index)
            s = s.resample('QE-DEC').last()
            s.index = s.index.to_period('Q-DEC')
            s.name = sid
            return s
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("No series matched in fred_q_any")

def read_quarterly_first(candidate_ids: list[str]) -> pd.Series:
    for sid in candidate_ids:
        p = os.path.join(RAW_OUTDIR, f"{sid}.csv")
        if os.path.exists(p):
            return read_fred_csv_quarterly(p, sid)
    raise FileNotFoundError(f"None of candidate CSVs found in {RAW_OUTDIR}: {candidate_ids}")

# --- Utilities ---
def normalize_to_base(q_series: pd.Series, base_period: pd.Period, *, eps: float = 1e-12, min_base: float | None = None) -> pd.Series:
    """Normalize a quarterly Series to 1.0 at base_period.

    If the base value is missing or too small, fall back to the first finite
    value meeting a minimum absolute threshold (min_base). If min_base is None,
    uses eps as the threshold.
    """
    if q_series is None or len(q_series) == 0:
        return q_series
    # Sanitize input series (replace infs with NaN to avoid deprecated options)
    ser = pd.to_numeric(q_series, errors='coerce').astype(float)
    ser = ser.replace([np.inf, -np.inf], np.nan)
    threshold = float(min_base) if (min_base is not None) else float(eps)
    base_val = None
    if base_period in ser.index:
        base_val = ser.loc[base_period]
    if pd.isna(base_val) or (base_val is not None and abs(float(base_val)) < threshold):
        # fallback to first finite value >= threshold
        mask = ser.apply(lambda x: np.isfinite(x) and abs(float(x)) >= threshold)
        if mask.any():
            base_val = ser[mask].iloc[0]
        else:
            return ser
    if not np.isfinite(float(base_val)) or abs(float(base_val)) < threshold:
        return ser  # avoid division by zero; return unnormalized as last resort
    return (ser / float(base_val))

# Helper to fetch a FRED series as pandas Series with PeriodIndex (Q)
def fred_q(series_id: str) -> pd.Series:
    if fred is None:
        raise RuntimeError("fred_q called without FRED_API_KEY. Provide CSV fallback in data_raw/")
    s = fred.get_series(series_id)
    s.index = pd.DatetimeIndex(s.index)
    # Pandas 2.2+: use 'QE-DEC' for quarter-end (Dec) instead of deprecated 'Q'
    s = s.resample('QE-DEC').last()
    s.index = s.index.to_period('Q-DEC')
    s.name = series_id
    return s

# Helper for weekly to quarterly (mean within quarter)
def fred_w_to_q(series_id: str) -> pd.Series:
    if fred is None:
        raise RuntimeError("fred_w_to_q called without FRED_API_KEY. Provide CSV fallback in data_raw/")
    s = fred.get_series(series_id)
    s.index = pd.DatetimeIndex(s.index)
    # Weekly -> quarterly mean, with explicit year-end calendar
    s = s.resample('QE-DEC').mean()
    s.index = s.index.to_period('Q-DEC')
    s.name = series_id
    return s

# CSV fallback readers (FRED CSV: DATE, VALUE)
def read_fred_csv_quarterly(path: str, series_id: str) -> pd.Series:
    df = pd.read_csv(path)
    # handle FRED format: DATE, VALUE, with '.' as missing
    df['DATE'] = pd.to_datetime(df['DATE'])
    # Replace '.' with NaN then to float
    df['VALUE'] = pd.to_numeric(df['VALUE'].replace('.', np.nan))
    s = pd.Series(df['VALUE'].values, index=pd.DatetimeIndex(df['DATE']), name=series_id)
    s = s.resample('QE-DEC').last()
    s.index = s.index.to_period('Q-DEC')
    return s

def read_fred_csv_weekly_to_quarterly(path: str, series_id: str) -> pd.Series:
    df = pd.read_csv(path)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['VALUE'] = pd.to_numeric(df['VALUE'].replace('.', np.nan))
    s = pd.Series(df['VALUE'].values, index=pd.DatetimeIndex(df['DATE']), name=series_id)
    s = s.resample('QE-DEC').mean()
    s.index = s.index.to_period('Q-DEC')
    return s

# ---------- Load BIS (via FRED) quarterly series ----------
if fred is not None:
    # Save raw inputs (DATE,VALUE) to RAW_OUTDIR for reproducibility
    save_fred_raw_batch([
        # US preferred (USD level), then legacy/index variants
        'QUSPAMUSDA','QUSPBMUSDA', 'CRDQUSAPABIS','QUSPBM770A',
        # JP (USD variants)
        'QJPPAMUSDA','QJPPBMUSDA',
        # Fed weekly
        'WALCL','WSHOMCB'
    ])

    # US (use USD-level variants for numerator/denominator consistency)
    us_total = fred_q_any(['QUSPAMUSDA','CRDQUSAPABIS'])
    us_bank  = fred_q_any(['QUSPBMUSDA','QUSPBM770A'])
    # JP (USD-denominated variants for cross-country unit consistency)
    jp_total = fred_q('QJPPAMUSDA')
    jp_bank  = fred_q('QJPPBMUSDA')
else:
    # CSV fallback: load from RAW_OUTDIR (DATE,VALUE format). Try preferred IDs first.
    us_total = read_quarterly_first(['QUSPAMUSDA','CRDQUSAPABIS'])
    us_bank  = read_quarterly_first(['QUSPBMUSDA','QUSPBM770A'])
    jp_total = read_quarterly_first(['QJPPAMUSDA'])
    jp_bank  = read_quarterly_first(['QJPPBMUSDA'])


# Align indices
common_q = us_total.dropna().index.union(us_bank.dropna().index).union(jp_total.dropna().index).union(jp_bank.dropna().index)
us_total = us_total.reindex(common_q)
us_bank  = us_bank.reindex(common_q)
jp_total = jp_total.reindex(common_q)
jp_bank  = jp_bank.reindex(common_q)

# Shares
us_bank_share = (us_bank / us_total).replace([np.inf, -np.inf], np.nan).rename('US bank share of private credit')
jp_bank_share = (jp_bank / jp_total).replace([np.inf, -np.inf], np.nan).rename('JP bank share of private credit')

# ---------- Figure 1: US bank share over time ----------
asc860_q = pd.Period('2010Q1', freq='Q-DEC')
fig1 = plt.figure(figsize=(8,4.5))
ax = plt.gca()
us_bank_share.plot(ax=ax)
ax.axvline(asc860_q.start_time, linestyle='--', linewidth=1)
ax.set_title('US: Bank share of private non‑financial credit (BIS via FRED)')
ax.set_ylabel('Share (banks / total)')
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Quarter')
ax.grid(True, alpha=0.3)
cap1 = f"US = {us_bank.name}/{us_total.name}"
add_caption(ax, cap1, loc='top-left')
plt.tight_layout()
fig1_path = os.path.join(OUTDIR, 'us_bank_share_private_credit.png')
plt.savefig(fig1_path, dpi=200)
plt.close(fig1)

# ---------- Figure 2: US vs Japan bank share ----------
fig2 = plt.figure(figsize=(8,4.5))
ax2 = plt.gca()
us_bank_share.plot(ax=ax2)
jp_bank_share.plot(ax=ax2)
ax2.axvline(asc860_q.start_time, linestyle='--', linewidth=1)
ax2.set_title('US vs JP: Bank share of private non‑financial credit')
ax2.set_ylabel('Share (banks / total)')
ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
ax2.set_ylim(0.0, 1.0)
ax2.legend()
ax2.grid(True, alpha=0.3)
cap2 = f"US={us_bank.name}/{us_total.name}; JP={jp_bank.name}/{jp_total.name}"
add_caption(ax2, cap2, loc='top-left')
plt.tight_layout()
fig2_path = os.path.join(OUTDIR, 'us_jp_bank_share_compare.png')
plt.savefig(fig2_path, dpi=200)
plt.close(fig2)

# ---------- Figure 3: Fed support (QE/MBS) ----------
if fred is not None:
    walcl_q = fred_w_to_q('WALCL')
    wshomcb_q = fred_w_to_q('WSHOMCB')
else:
    walcl_q = read_fred_csv_weekly_to_quarterly(os.path.join(RAW_OUTDIR,'WALCL.csv'), 'WALCL')
    wshomcb_q = read_fred_csv_weekly_to_quarterly(os.path.join(RAW_OUTDIR,'WSHOMCB.csv'), 'WSHOMCB')

# Normalize to 1.0 at 2008Q4 for readability (guard near-zero bases)
base = pd.Period('2008Q4', freq='Q-DEC')
walcl_n = normalize_to_base(walcl_q, base, min_base=1.0).rename('Fed total assets (norm)')
# WSHOMCB can be near-zero around 2008Q4; require at least $10B (10000 in millions)
wshomcb_n = normalize_to_base(wshomcb_q, base, min_base=10000.0).rename('Fed MBS (norm)')

fig3 = plt.figure(figsize=(8,4.5))
ax3 = plt.gca()
walcl_n.plot(ax=ax3)
wshomcb_n.plot(ax=ax3)
ax3.axvline(asc860_q.start_time, linestyle='--', linewidth=1)
ax3.set_title('Federal Reserve balance sheet & MBS holdings (normalized)')
ax3.set_ylabel('Index (base=1 at 2008Q4)')
# enforce plain number formatting to avoid scientific 1e6 display
ax3.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax3.ticklabel_format(style='plain', axis='y')
ax3.grid(True, alpha=0.3)
ax3.legend()
cap3 = "WALCL & WSHOMCB; base=2008Q4; weekly→quarterly mean"
add_caption(ax3, cap3, loc='top-left')
plt.tight_layout()
fig3_path = os.path.join(OUTDIR, 'fed_support_qe_mbs.png')
plt.savefig(fig3_path, dpi=200)
plt.close(fig3)

# --- Additional figures to avoid scale misinterpretation ---
# 3b) Nominal levels (USD trillions)
walcl_tril = (walcl_q / 1e6).rename('Fed total assets ($T)')
wshomcb_tril = (wshomcb_q / 1e6).rename('Fed MBS ($T)')
fig3b = plt.figure(figsize=(8,4.5))
ax3b = plt.gca()
walcl_tril.plot(ax=ax3b)
wshomcb_tril.plot(ax=ax3b)
ax3b.axvline(asc860_q.start_time, linestyle='--', linewidth=1)
ax3b.set_title('Federal Reserve balance sheet & MBS holdings (nominal)')
ax3b.set_ylabel('USD trillions')
ax3b.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax3b.ticklabel_format(style='plain', axis='y')
ax3b.grid(True, alpha=0.3)
ax3b.legend()
cap3b = "USD trillions; weekly→quarterly mean"
add_caption(ax3b, cap3b, loc='top-left')
plt.tight_layout()
fig3b_path = os.path.join(OUTDIR, 'fed_support_qe_mbs_nominal.png')
plt.savefig(fig3b_path, dpi=200)
plt.close(fig3b)

# 3c) Share: MBS / Total assets
mbs_share = (wshomcb_q / walcl_q).replace([np.inf, -np.inf], np.nan).rename('MBS share of Fed assets')
fig3c = plt.figure(figsize=(8,4.5))
ax3c = plt.gca()
mbs_share.plot(ax=ax3c)
ax3c.axvline(asc860_q.start_time, linestyle='--', linewidth=1)
ax3c.set_title('Federal Reserve: MBS as share of total assets')
ax3c.set_ylabel('Share')
ax3c.yaxis.set_major_formatter(PercentFormatter(1.0))
ax3c.set_ylim(0.0, 1.0)
ax3c.grid(True, alpha=0.3)
cap3c = "WSHOMCB/WALCL; weekly→quarterly mean"
add_caption(ax3c, cap3c, loc='top-left')
plt.tight_layout()
fig3c_path = os.path.join(OUTDIR, 'fed_mbs_share_of_total.png')
plt.savefig(fig3c_path, dpi=200)
plt.close(fig3c)

# ---------- Figure 4: US bank share vs NBFI commitments (parallel) ----------
# We rely on a user-provided CSV: data_raw/nbfi_commitments_us.csv
# Columns (example, USD billions):
#   date, bdc_commit, private_debt_commit, other_nbfi_commit
# Example row:
#   2013-03-31, 8.0, 0.0, 0.0
#   2024-12-31, 53.1, 39.4, 0.0
# You can add more categories; the script will sum all columns except 'date'.

RAW_CSV = os.path.join(RAW_OUTDIR, 'nbfi_commitments_us.csv')
if os.path.exists(RAW_CSV):
    df_nbfi = pd.read_csv(RAW_CSV)
    # Parse dates and set PeriodIndex (Q)
    df_nbfi['date'] = pd.to_datetime(df_nbfi['date'])
    # Coerce all non-date columns to numeric (handle '.', blanks, etc.)
    for col in df_nbfi.columns:
        if col != 'date':
            df_nbfi[col] = pd.to_numeric(df_nbfi[col], errors='coerce')
    df_nbfi = df_nbfi.set_index('date').sort_index()
    # Quarterly alignment (sum within quarter is fine for commitments snapshot; or use last if you record EoQ levels)
    # Here we assume the CSV holds EoQ levels; we take last value in each quarter.
    nbfi_q = df_nbfi.resample('QE-DEC').last()
    nbfi_q.index = nbfi_q.index.to_period('Q-DEC')
    # Total commitments = sum of all numeric columns
    nbfi_total = nbfi_q.select_dtypes(include=[np.number]).sum(axis=1)
    nbfi_total.name = 'NBFI commitments (USD bn)'

    # Align with US bank share (use bank-share full timeline; overlay NBFI where available)
    idx = us_bank_share.index
    x_full = idx.to_timestamp()
    # Forward-fill to represent interval-constant stock between updates
    nbfi_step = nbfi_total.reindex(idx).ffill()

    # Plot with twin y-axes (parallel view)
    fig4 = plt.figure(figsize=(8,4.8))
    ax4 = plt.gca()
    line1, = ax4.plot(x_full, us_bank_share.values, label='US bank share of private credit')
    ax4.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax4.set_ylabel('Bank share (banks/total)')
    ax4.set_xlabel('Quarter')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(asc860_q.start_time, linestyle='--', linewidth=1)

    ax4b = ax4.twinx()
    # Step line (interval-constant via ffill)
    line2 = ax4b.step(
        x_full, nbfi_step.values,
        where='post', label='NBFI commitments (USD bn, step/ffill)', color='tab:orange', zorder=2
    )
    # Also overlay actual observation points as markers if helpful
    obs = nbfi_total.dropna()
    if not obs.empty:
        ax4b.plot(obs.index.to_timestamp(), obs.values, linestyle='None', marker='o',
                  markersize=5, color='tab:orange', markeredgecolor='white', markeredgewidth=0.8, zorder=3)
    ymax = float(np.nanmax(nbfi_step.values)) if np.isfinite(np.nanmax(nbfi_step.values)) else None
    if ymax and ymax > 0:
        ax4b.set_ylim(0, ymax * 1.15)

    # If NBFI points exist, zoom x-axis around that window (e.g., start 5y before first point)
    if not obs.empty:
        tmin = pd.to_datetime(obs.index.min().to_timestamp())
        tmax = pd.to_datetime(x_full.max())
        xmin = tmin - pd.DateOffset(years=5)
        ax4.set_xlim(xmin, tmax)
        # Optimize left y-axis limits based on data within the visible window
        vis_mask = (x_full >= xmin) & (x_full <= tmax)
        s_vis = pd.Series(us_bank_share.values, index=x_full)[vis_mask].dropna()
        if not s_vis.empty:
            q5, q95 = s_vis.quantile(0.05), s_vis.quantile(0.95)
            pad = max(0.01, float(q95 - q5) * 0.08)
            lo = max(0.0, float(q5 - pad))
            hi = min(1.0, float(q95 + pad))
            if hi > lo:
                ax4.set_ylim(lo, hi)
    ax4b.set_ylabel('USD billions')

    # Build a combined legend
    lines = [line1, line2[0]]
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    ax4.set_title('US: Bank share vs NBFI commitments (parallel)')
    cap4 = (
        f"Bank share={us_bank.name}/{us_total.name}; NBFI=CSV sum; step=ffill"
    )
    add_caption(ax4, cap4, loc='top-left')
    plt.tight_layout()
    fig4_path = os.path.join(OUTDIR, 'us_bankshare_vs_nbficommit.png')
    plt.savefig(fig4_path, dpi=200)
    plt.close(fig4)

    print('Saved:')
    for p in [fig1_path, fig2_path, fig3_path, fig3b_path, fig3c_path, fig4_path]:
        print('  ', p)
else:
    print('Saved:')
    for p in [fig1_path, fig2_path, fig3_path, fig3b_path, fig3c_path]:
        print('  ', p)
    print('NOTE: To generate Figure 4 (us_bankshare_vs_nbficommit.png), provide a CSV at:', RAW_CSV)
    print('CSV columns example: date, bdc_commit, private_debt_commit, other_nbfi_commit  (values in USD billions)')

# ---------- Figure 5: US φ vs Fed MBS / Total credit (two-axis) ----------
# Prefer a simple CSV: data_processed/us_phi.csv with columns: date, phi_us (0-1)
# Fallbacks: data_processed/proc_code04_US_phi_series_[observed_post|observed_pre|raw].csv
def _load_phi_us_series() -> pd.Series | None:
    """Try to load a quarterly US φ series (PeriodIndex Q-DEC, values 0-1).

    Priority:
      1) data_processed/us_phi.csv (columns: date, phi_us or any 'phi' column)
      2) data_processed/proc_code04_US_phi_series_observed_post.csv
      3) data_processed/proc_code04_US_phi_series_observed_pre.csv
      4) data_processed/proc_code04_US_phi_series_raw.csv
    Column preference inside proc_code04_* files:
      'phi_US_adj' > 'phi_US_raw' > first column starting with 'phi_US' > first column containing 'phi'
    """
    # 1) Simple file
    phi_simple = os.path.join('data_processed', 'us_phi.csv')
    if os.path.exists(phi_simple):
        df = pd.read_csv(phi_simple)
        # Accept either 'date' or 'Date'
        date_col = 'date' if 'date' in df.columns else ('Date' if 'Date' in df.columns else None)
        if date_col is None:
            return None
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        # Choose phi column
        phi_col = 'phi_us' if 'phi_us' in df.columns else None
        if phi_col is None:
            # pick first col containing 'phi'
            for c in df.columns:
                if 'phi' in str(c).lower():
                    phi_col = c
                    break
        if phi_col is None:
            return None
        q = df[[phi_col]].resample('QE-DEC').last()
        q.index = q.index.to_period('Q-DEC')
        return pd.to_numeric(q[phi_col], errors='coerce').astype(float).rename('US φ (off-balance)')

    # 2) proc_code04 fallbacks
    candidates = [
        os.path.join('data_processed','proc_code04_US_phi_series_observed_post.csv'),
        os.path.join('data_processed','proc_code04_US_phi_series_observed_pre.csv'),
        os.path.join('data_processed','proc_code04_US_phi_series_raw.csv'),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            date_col = 'Date' if 'Date' in df.columns else ('date' if 'date' in df.columns else None)
            if date_col is None:
                continue
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
            # Column preference
            pref = ['phi_US_adj','phi_US_raw']
            phi_col = None
            for c in pref:
                if c in df.columns:
                    phi_col = c; break
            if phi_col is None:
                # first starting with 'phi_US'
                for c in df.columns:
                    if str(c).startswith('phi_US'):
                        phi_col = c; break
            if phi_col is None:
                for c in df.columns:
                    if 'phi' in str(c).lower():
                        phi_col = c; break
            if phi_col is None:
                continue
            q = df[[phi_col]].resample('QE-DEC').last()
            q.index = q.index.to_period('Q-DEC')
            label = 'US φ (off-balance)'
            if phi_col.endswith('_adj'): label = 'US φ (off-balance, adj)'
            if phi_col.endswith('_raw'): label = 'US φ (off-balance, raw)'
            return pd.to_numeric(q[phi_col], errors='coerce').astype(float).rename(label)
    return None

try:
    phi_us_series = _load_phi_us_series()
    if phi_us_series is not None:
        # Align units: WSHOMCB is millions of USD; convert to billions to match QUSPAMUSDA units
        fed_mbs_share_total = ((wshomcb_q / 1000.0) / us_total).replace([np.inf, -np.inf], np.nan).rename('Fed MBS / Total credit')

        df5 = pd.concat([phi_us_series, fed_mbs_share_total], axis=1).dropna()
        if not df5.empty:
            fig5 = plt.figure(figsize=(9,4.8))
            ax5 = plt.gca()
            l1, = ax5.plot(df5.index.to_timestamp(), df5[phi_us_series.name]*100, label=phi_us_series.name, color='tab:blue')
            ax5.set_ylabel('US φ (%)')
            ax5.set_xlabel('Quarter')
            ax5.grid(True, alpha=0.3)
            ax5.axvline(pd.Period('2010Q1', freq='Q-DEC').start_time, linestyle='--', linewidth=1, color='gray')

            ax5b = ax5.twinx()
            l2, = ax5b.plot(
                df5.index.to_timestamp(),
                df5['Fed MBS / Total credit']*100,
                label='Fed MBS / Total credit', linestyle='--', color='tab:orange'
            )
            ax5b.set_ylabel('Fed MBS / Total credit (%)')
            # Fix axis ranges as requested
            ax5.set_ylim(9.5, 30)  # φ: 9.5–30%
            ax5b.set_ylim(0, 7.0)  # Fed MBS/Total credit: 0–7.0%

            lines = [l1, l2]
            labels = [ln.get_label() for ln in lines]
            ax5.legend(lines, labels, loc='upper left')
            ax5.set_title('US: φ vs Fed MBS / Total credit (two-axis with ASC 860 marker)')
            add_caption(ax5, 'US φ vs Fed MBS/Total; quarterly (Q-DEC); MBS in $bn', loc='top-left')
            plt.tight_layout()
            fig5_path = os.path.join(OUTDIR, 'us_phi_vs_fedmbs_totalcredit.png')
            plt.savefig(fig5_path, dpi=200)
            plt.close(fig5)
            print('Also saved:')
            print('  ', fig5_path)
        else:
            print('NOTE: Figure 5 could not be generated (no overlapping quarters between φ and Fed/Total).')
    else:
        print('NOTE: To generate Figure 5 (us_phi_vs_fedmbs_totalcredit.png), provide φ CSV at data_processed/us_phi.csv')
        print('      or ensure a processed file exists like data_processed/proc_code04_US_phi_series_raw.csv')
        print('Expected columns include: Date and one of phi_US_adj, phi_US_raw, etc. (values in 0–1).')
except Exception as e:
    print('WARNING: Figure 5 generation failed:', e)