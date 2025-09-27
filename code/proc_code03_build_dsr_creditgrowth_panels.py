"""proc_code03_build_dsr_creditgrowth_panels

Purpose:
    Build standardized (prefixed) quarterly panels used by fig_code06 / fig_code12 / fig_code13.

Outputs (data_processed/):
    - proc_code03_US_DSR_CreditGrowth_panel.csv
    - proc_code03_JP_DSR_CreditGrowth_panel.csv

Schema:
    Date (quarter-end) as first column; columns: DSR_pct, CreditGrowth_ppYoY, ProxyFlag[, MetaVersion]

Required raw inputs in data_raw/ (must exist before running):
    1. bis_dp_search_export_*.csv   (BIS Debt Service Ratio export; contains Q.<ISO>.<sector>)
    2. HHMSDODNS.csv                (US household debt outstanding from FRED) for US credit growth
Optional raw/processed inputs:
    - data_processed/JP_DSR_CreditGrowth_panel.csv (legacy JP panel: used ONLY to inherit CreditGrowth_ppYoY)
    - data_processed_bk/JP_DSR_CreditGrowth_panel.csv (backup legacy)
    - data_processed/proc_code02_JP_phi_quarterly.csv (φ proxy; only used if BIS + legacy both absent)

Current Build Logic (2025Q3 consolidated):
    JP:
        - Parse BIS export (multi-file tolerant). Priority: PNFS -> HH -> NFC.
        - Set ProxyFlag=9 (official). MetaVersion=BIS_<sector>_v1 (+CG_legacy if credit growth inherited).
        - CreditGrowth_ppYoY: if legacy panel present, date-align and inherit that column; else fill 0.0.
        - Fallbacks (only if BIS missing): legacy (ProxyFlag=0, MetaVersion=legacy_adopted) → φ proxy (ProxyFlag=1, MetaVersion=proxy_phi).
    US:
        - BIS DSR (sector=P) + HHMSDODNS 4-quarter percent change * 100 -> CreditGrowth_ppYoY.
        - If BIS or credit series missing: attempt legacy (ProxyFlag=0); proxy path presently disabled (allow_proxy=False by design here).

CLI:
    python code/proc_code03_build_dsr_creditgrowth_panels.py
    (No options; always overwrites regenerated panels.)

ProxyFlag semantics:
    0 = adopted legacy / validated original
    1 = constructed proxy (φ-based, last-resort JP only)
    9 = official BIS source

MetaVersion examples:
    BIS_PNFS_v1+CG_legacy, legacy_adopted, proxy_phi

All quarter indices are normalized via .to_period('Q').to_timestamp('Q').
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import glob, os, re
from datetime import datetime

from util_code01_lib_io import find, safe_to_csv

DATA_PROCESSED = Path('data_processed')
DATA_PROCESSED_BK = Path('data_processed_bk')
US_OUT = DATA_PROCESSED / 'proc_code03_US_DSR_CreditGrowth_panel.csv'
JP_OUT = DATA_PROCESSED / 'proc_code03_JP_DSR_CreditGrowth_panel.csv'
LEGACY_US = DATA_PROCESSED / 'US_DSR_CreditGrowth_panel.csv'
LEGACY_JP = DATA_PROCESSED / 'JP_DSR_CreditGrowth_panel.csv'
LEGACY_US_BK = DATA_PROCESSED_BK / 'US_DSR_CreditGrowth_panel.csv'
LEGACY_JP_BK = DATA_PROCESSED_BK / 'JP_DSR_CreditGrowth_panel.csv'

# ---------------------------------------------------------------------------
# Preflight helper: advise user how to obtain required raw inputs if missing.
# ---------------------------------------------------------------------------
def _preflight_raw_inputs():
    missing_msgs = []
    # BIS DSR export
    bis_files = find(['bis_dp_search_export_*.csv'])
    if not bis_files:
        missing_msgs.append(
            '[MISSING] BIS DSR export (bis_dp_search_export_*.csv) not found in data_raw/.\n'
            '  Obtain via BIS Data Portal: https://data.bis.org/topics/dsr  -> Export (CSV)\n'
            '  Save the downloaded file into data_raw/ with its original filename (e.g. bis_dp_search_export_2025Q3.csv).')
    # US household debt (FRED series HHMSDODNS)
    fred_files = find(['HHMSDODNS.csv'])
    if not fred_files:
        missing_msgs.append(
            '[MISSING] FRED household debt series HHMSDODNS.csv not found in data_raw/.\n'
            '  Download CSV from: https://fred.stlouisfed.org/series/HHMSDODNS  (Download Data -> CSV).\n'
            '  Place as data_raw/HHMSDODNS.csv. Optionally use helper: \n'
            '    python code/fetch_code08_fred_series.py --series HHMSDODNS --outfile data_raw/HHMSDODNS.csv')
    if missing_msgs:
        print('\n'.join(['[PRE-FLIGHT] Required raw inputs missing:', *missing_msgs]))
    return not missing_msgs  # True if all present

def _load_bis_dsr(country_code: str, sector: str='P'):
    files = find(['bis_dp_search_export_*.csv'])
    if not files:
        return None, 'BIS export not found'
    files = sorted(files, key=lambda p: p.stat().st_size, reverse=True)
    try:
        df = pd.read_csv(files[0], skiprows=2)
    except Exception as e:
        return None, f'BIS read failed: {e}'
    df.columns = [c.strip() for c in df.columns]
    keycol = next((c for c in df.columns if 'KEY' in c), None)
    tcol = next((c for c in df.columns if 'TIME' in c), None)
    vcol = next((c for c in df.columns if 'OBS_VALUE' in c), None)
    if not (keycol and tcol and vcol):
        return None, 'BIS columns incomplete'
    pick = df[df[keycol].str.contains(f'Q.{country_code}.{sector}', na=False)].copy()
    if pick.empty:
        return None, f'No BIS DSR rows for {country_code}.{sector}'
    pick[tcol] = pd.to_datetime(pick[tcol], errors='coerce')
    pick[vcol] = pd.to_numeric(pick[vcol], errors='coerce')
    pick = pick.dropna(subset=[tcol, vcol]).set_index(tcol).sort_index()
    # 標準化: 四半期を quarter-end Timestamp に揃える
    try:
        q_idx = pick.index.to_period('Q').to_timestamp('Q')
        pick.index = q_idx
    except Exception:
        pass
    return pick[[vcol]].rename(columns={vcol: 'DSR_pct'}), None

def _load_us_credit_proxy(min_start_year: int = 1960):
    """Load HHMSDODNS (household debt) and compute 4q % change.
    Trims early sparse decades to avoid extreme early pct_change artefacts.
    """
    files = find(['HHMSDODNS.csv'])
    if not files:
        return None, 'HHMSDODNS.csv missing'
    s = pd.read_csv(files[0])
    dt = next(c for c in s.columns if 'DATE' in c.upper())
    val = next(c for c in s.columns if c != dt)
    s[dt] = pd.to_datetime(s[dt], errors='coerce')
    s[val] = pd.to_numeric(s[val], errors='coerce')
    s = s.dropna(subset=[dt, val]).set_index(dt).sort_index()
    # 四半期末揃え (FRED 四半期系列は期首日付の場合があるため)
    try:
        s.index = s.index.to_period('Q').to_timestamp('Q')
    except Exception:
        pass
    # Trim to start_year or first window with 8 consecutive non-null quarterly points (safety)
    if not s.empty:
        cand = s[s.index.year >= min_start_year]
        if len(cand) > 12:
            s = cand
    g = (s[val].pct_change(4) * 100).rename('CreditGrowth_ppYoY')
    return g.to_frame(), None

def _load_legacy(path_primary: Path, path_backup: Path):
    for p in (path_primary, path_backup):
        if p.exists():
            try:
                df = pd.read_csv(p, index_col=0)
                df.index = pd.to_datetime(df.index)
                sub = df[['DSR_pct','CreditGrowth_ppYoY']].dropna().copy()
                sub.index.name = ''
                return sub, f'legacy copied ({p.name})'
            except Exception as e:
                return None, f'legacy parse failed: {e}'
    return None, 'legacy not found'

def build_us_panel(allow_proxy: bool):
    dsr, e1 = _load_bis_dsr('US', 'P')
    credit, e2 = _load_us_credit_proxy()
    if dsr is not None and credit is not None:
        df = dsr.join(credit, how='inner').dropna()
        df.index = pd.to_datetime(df.index)
        df.index.name = ''
        df['ProxyFlag'] = 0
        return df, None
    # Try legacy before proxy
    legacy, lnote = _load_legacy(LEGACY_US, LEGACY_US_BK)
    if legacy is not None:
        legacy['ProxyFlag'] = 0
        return legacy, lnote + ' (US)'
    if not allow_proxy:
        return None, f'No BIS ({e1}) and no legacy; proxy disallowed (--allow-us-proxy to enable)'
    if credit is None:
        return None, f'Cannot form proxy: credit missing ({e2})'
    # Proxy: mean-reverting transform of credit growth (avoid cumulative drift)
    g = credit.copy()
    # Smooth volatility with 4-quarter rolling mean
    g_sm = g['CreditGrowth_ppYoY'].rolling(4, min_periods=1).mean()
    # Standardize (z-score) to scale dynamic amplitude
    std = g_sm.std()
    if std == 0 or pd.isna(std):
        std = 1.0
    z = (g_sm - g_sm.mean()) / std
    dsr_proxy = 9.0 + z * 1.5  # target band roughly 5.5 - 12.5
    dsr_proxy = dsr_proxy.clip(lower=4.0, upper=15.0)
    proxy = dsr_proxy.rename('DSR_pct').to_frame().join(g, how='inner').dropna()
    proxy.index = pd.to_datetime(proxy.index).to_period('Q').to_timestamp('Q')
    proxy.index.name = ''
    proxy['ProxyFlag'] = 1
    return proxy, f'proxy DSR constructed (US; no BIS: {e1}; no legacy)'

def _parse_bis_export_files(pattern: str='bis_dp_search_export'):
    raw_dir = Path('data_raw')
    paths = sorted(glob.glob(str(raw_dir / f"{pattern}*.csv"))) + \
            sorted(glob.glob(str(raw_dir / f"{pattern}*.xlsx")))
    if not paths:
        return None, 'no BIS export file in data_raw'
    merged = None
    def _parse_one(p: str):
        if p.lower().endswith('.xlsx'):
            df = pd.read_excel(p)
        else:
            df = pd.read_csv(p)
        if df.empty or df.shape[1] < 2:
            raise ValueError('format')
        first = df.columns[0]
        idx_tp = df.index[df[first].astype(str).str.contains('TIME_PERIOD', na=False)].tolist()
        if idx_tp:
            start_idx = idx_tp[0] + 1
        else:
            date_pat = re.compile(r'^\d{4}-\d{2}-\d{2}$')
            start_idx = None
            for i,v in enumerate(df[first].astype(str)):
                if date_pat.match(v):
                    start_idx = i; break
            if start_idx is None:
                raise ValueError('no time start')
        sector_map = {}
        if df.shape[0] >= 4:
            meta_row = 3
            for c in df.columns[1:4]:
                val = str(df.iloc[meta_row].get(c, ''))
                if ':' in val:
                    code = val.split(':')[0].strip()
                    if code in {'H','N','P'}:
                        sector_map[c] = {'H':'HH','N':'NFC','P':'PNFS'}[code]
        if not sector_map:
            order = ['HH','NFC','PNFS']
            for i,c in enumerate(df.columns[1:1+len(order)]):
                sector_map[c] = order[i]
        data = df.iloc[start_idx:].copy()
        data = data.rename(columns={first:'TIME_PERIOD', **sector_map})
        keep = ['TIME_PERIOD'] + [c for c in ['HH','NFC','PNFS'] if c in data.columns]
        data = data[keep]
        data['TIME_PERIOD'] = pd.to_datetime(data['TIME_PERIOD'], errors='coerce')
        for c in ['HH','NFC','PNFS']:
            if c in data.columns:
                data[c] = pd.to_numeric(data[c], errors='coerce')
        data = data.dropna(subset=['TIME_PERIOD']).sort_values('TIME_PERIOD')
        return data.reset_index(drop=True)
    for p in paths:
        try:
            d = _parse_one(p)
        except Exception:
            continue
        if merged is None:
            merged = d
        else:
            merged = pd.merge(merged, d, on='TIME_PERIOD', how='outer', suffixes=('', '_dup'))
    if merged is None or merged.empty:
        return None, 'parse failed for all BIS exports'
    out = merged[['TIME_PERIOD']].copy()
    for base in ['HH','NFC','PNFS']:
        cols = [c for c in merged.columns if c == base or c.startswith(base + '_dup')]
        if not cols:
            continue
        ser = None
        for c in cols:
            ser = merged[c] if ser is None else ser.where(ser.notna(), merged[c])
        out[base] = ser
    out = out.sort_values('TIME_PERIOD').reset_index(drop=True)
    return out, None

def _build_jp_from_bis(prefer: str='PNFS'):
    table, err = _parse_bis_export_files()
    if table is None:
        return None, err
    order = [prefer] + [c for c in ['PNFS','HH','NFC'] if c != prefer]
    pick_col, src = None, None
    for c in order:
        if c in table.columns and table[c].notna().any():
            pick_col = c; src = c; break
    if pick_col is None:
        return None, 'no usable BIS JP DSR column'
    dsr = table[['TIME_PERIOD', pick_col]].dropna().copy()
    dsr['Date'] = pd.to_datetime(dsr['TIME_PERIOD']).dt.to_period('Q').dt.to_timestamp('Q')
    dsr = dsr.drop_duplicates('Date').set_index('Date')
    panel = pd.DataFrame({'DSR_pct': dsr[pick_col]})
    # 初期は 0.0 プレースホルダ
    panel['CreditGrowth_ppYoY'] = 0.0
    panel['ProxyFlag'] = 9
    meta = f'BIS_{src}_v1'
    # legacy からの CreditGrowth 継承試行
    legacy, _ = _load_legacy(LEGACY_JP, LEGACY_JP_BK)
    if legacy is not None and 'CreditGrowth_ppYoY' in legacy.columns:
        try:
            cg = legacy['CreditGrowth_ppYoY'].copy()
            cg.index = pd.to_datetime(cg.index)
            cg.index = cg.index.to_period('Q').to_timestamp('Q')
            panel['CreditGrowth_ppYoY'] = cg.reindex(panel.index).fillna(0.0)
            meta += '+CG_legacy'
        except Exception:
            pass
    panel['MetaVersion'] = meta
    panel.index.name = ''
    return panel, f'BIS export used ({src})'

def build_jp_panel():
    # 1. BIS
    bis_df, note = _build_jp_from_bis()
    if bis_df is not None:
        return bis_df, note
    # 2. legacy fallback
    legacy, note_leg = _load_legacy(LEGACY_JP, LEGACY_JP_BK)
    if legacy is not None:
        legacy['ProxyFlag'] = 0
        if 'MetaVersion' not in legacy.columns:
            legacy['MetaVersion'] = 'legacy_adopted'
        return legacy, note_leg.replace('legacy copied', 'legacy copied (JP)')
    # 3. last-resort proxy (phi) — self-contained, only if file exists
    phi_path = Path('data_processed/proc_code02_JP_phi_quarterly.csv')
    if phi_path.exists():
        try:
            phi = pd.read_csv(phi_path, parse_dates=['Date'])
            if 'phi_JP' in phi.columns:
                work = phi[['Date','phi_JP']].dropna().copy()
                work['Date'] = work['Date'].dt.to_period('Q').dt.to_timestamp('Q')
                work = work.drop_duplicates(subset=['Date']).sort_values('Date').set_index('Date')
                centered = work['phi_JP'] - work['phi_JP'].mean()
                dsr = (8.0 + centered * 2.0).clip(lower=3.0, upper=18.0)
                cg = (work['phi_JP'] - work['phi_JP'].shift(4)) * 100.0
                proxy = pd.DataFrame({'DSR_pct': dsr, 'CreditGrowth_ppYoY': cg.fillna(0.0)})
                proxy['ProxyFlag'] = 1
                proxy['MetaVersion'] = 'proxy_phi'
                proxy.index.name = ''
                return proxy, 'proxy from phi_JP (no BIS/legacy)'
        except Exception:
            pass
    return None, 'no JP source (BIS+legacy+phi all unavailable)'

def _write(df: pd.DataFrame, path: Path, force: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        try:
            existing = pd.read_csv(path, index_col=0)
            existing.index = pd.to_datetime(existing.index)
            if existing.equals(df):
                print(f'[SKIP] {path.name} already up-to-date')
                return path
        except Exception:
            pass
    # Overwrite in place (processed canonical artifact) to keep stable filename
    df.to_csv(path, index=True)
    print('[OK] wrote', path)
    return path

def main(argv=None):
    force = True  # 常に再生成
    _preflight_raw_inputs()
    us_df, us_note = build_us_panel(allow_proxy=False)
    if us_df is not None:
        _write(us_df, US_OUT, force)
        if us_note:
            print('[INFO][US]', us_note)
    else:
        print('[WARN][US]', us_note)
    jp_df, jp_note = build_jp_panel()
    if jp_df is not None:
        _write(jp_df, JP_OUT, force)
        if jp_note:
            print('[INFO][JP]', jp_note)
    else:
        print('[WARN][JP]', jp_note)
    # Wide-format JP/US DSR panel output (always by default, immediately after panel output)
    try:
        jp = pd.read_csv(JP_OUT, index_col=None)
        us = pd.read_csv(US_OUT, index_col=None)
        if 'Date' not in jp.columns:
            jp = jp.reset_index().rename(columns={'index':'Date'})
        if 'Date' not in us.columns:
            us = us.reset_index().rename(columns={'index':'Date'})
        jp = jp.rename(columns={c: c+'_JP' for c in jp.columns if c!='Date'})
        us = us.rename(columns={c: c+'_US' for c in us.columns if c!='Date'})
        wide = pd.merge(jp, us, on='Date', how='outer').sort_values('Date')
        # proc_code03_bis_DSR_JP_US_panel.csv: minimal (Date, DSR_pct_JP, DSR_pct_US)
        panel_min = wide[['Date','DSR_pct_JP','DSR_pct_US']].copy()
        panel_min.to_csv(DATA_PROCESSED/'proc_code03_bis_DSR_JP_US_panel.csv', index=False)
        print(f"[OK] wrote proc_code03_bis_DSR_JP_US_panel.csv rows={len(panel_min)}")
        # proc_code03_bis_DSR_wide.csv: all columns
        wide.to_csv(DATA_PROCESSED/'proc_code03_bis_DSR_wide.csv', index=False)
        print(f"[OK] wrote proc_code03_bis_DSR_wide.csv rows={len(wide)}")
    except Exception as e:
        print(f"[ERROR] wide-format DSR panel output failed: {e}")
    if us_df is None and jp_df is None:
        return 1
    return 0

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
