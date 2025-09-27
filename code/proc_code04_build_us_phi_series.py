#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
proc_code04_build_us_phi_series.py  (bank_off φ builder)

コピー元: proc_code04_build_us_phi_series_integrated.py

機能:
  - FRED Pools (AGSEBMPTCMAHDFS) と Loans_on (RREACBM027SBOG / fallback RREACBM027NBOG) を読み込み
  - 月次→四半期化 (各Q末) / 単位自動判定 (Millions→Billions 正規化)
  - φ (bank_off) = Pools / (Pools + Loans_on)
  - 既存 `proc_code04_US_phi_series_raw.csv` へ列(Pools_bilUSD, Loans_on_bilUSD, phi_US_bank_off_raw) 追記/更新
  - 監査 sidecar `proc_code04_US_phi_unit_audit.txt` 出力

将来拡張メモ:
  - bank_on / total 分母 φ の併記
  - break splicing / adjusted 系列統合
"""
from __future__ import annotations
import argparse, math
import numpy as np
from pathlib import Path
import pandas as pd

DEF_POOLS = Path("data_raw/AGSEBMPTCMAHDFS.csv")
DEF_LOANS_PRIMARY   = Path("data_raw/RREACBM027SBOG.csv")  # SA monthly (preferred)
DEF_LOANS_FALLBACK  = Path("data_raw/RREACBM027NBOG.csv")  # NSA monthly (fallback)
DEF_OUT_RAW = Path("data_processed/proc_code04_US_phi_series_raw.csv")
DEF_AUDIT   = Path("data_processed/proc_code04_US_phi_unit_audit.txt")
DEF_HOUSEHOLD = Path("data_raw/HHMSDODNS.csv")  # Z.1 Home Mortgages (Households) – denominator alternative

def safe_read_optional(path: Path) -> pd.Series|None:
    if not path or not path.exists():
        return None
    try:
        return read_fred_series_q(path)
    except Exception as e:
        print(f"[WARN] failed reading optional series {path}: {e}")
        return None

def read_fred_series_q(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    dcol = None
    for c in df.columns:
        if c.lower() in ("date","observation_date"):
            dcol = c; break
    if dcol is None: dcol = df.columns[0]
    vcols = [c for c in df.columns if c != dcol]
    vcol = vcols[0] if vcols else df.columns[-1]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    s = pd.to_numeric(df[vcol], errors="coerce")
    s.index = df[dcol]
    s = s.dropna().sort_index()
    s.index = s.index.to_period("Q").to_timestamp("Q")
    s = s.groupby(level=0).last()
    s.name = path.stem
    return s

def detect_units_to_billions(s: pd.Series, hint: str|None=None) -> tuple[pd.Series,str]:
    """(series_in_billions, flag) を返却。ヒューリスティック:
       hint==millions -> /1000, hint==billions -> as-is
       それ以外: 中央値が 1e5~1e8 → millions 判定 / それ以外 billions
    """
    x = s.copy()
    if hint == "millions":
        return x/1000.0, "millions->billions (/1000)"
    if hint == "billions":
        return x, "billions (as-is)"
    med = float(x.median()) if len(x) else float("nan")
    if not math.isfinite(med):
        return x, "unknown (as-is)"
    if 1e5 <= med <= 1e8:
        return x/1000.0, "auto: millions->billions"
    return x, "auto: billions (as-is)"

def detect_break(series: pd.Series):
    if series.empty: return None
    diffs = series.diff().dropna()
    if diffs.empty: return None
    recent = diffs[diffs.index >= pd.Timestamp('1990-01-01')]
    target = recent if not recent.empty else diffs
    bk = target.abs().idxmax()
    jump = target.loc[bk]
    loc = series.index.get_loc(bk)
    if isinstance(loc, slice) or loc == 0:
        return None
    prev = series.index[loc-1]
    med_prev = series.diff().abs().loc[:prev].tail(8).median()
    if med_prev is None or math.isnan(med_prev):
        med_prev = 0.0
    if abs(jump) > max(0.05, 5*med_prev):
        scale = series.loc[bk] / series.loc[prev] if series.loc[prev] != 0 else np.nan
        return {
            'break_date': bk,
            'prev_date': prev,
            'jump': float(jump),
            'scale_pre_to_post': float(scale),
            'median_prev_abs_diff': float(med_prev)
        }
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(DEF_POOLS))
    ap.add_argument("--loans", default=str(DEF_LOANS_PRIMARY),
                    help="Prefer SA monthly (RREACBM027SBOG). If missing, fallback to NBOG.")
    ap.add_argument("--out", default=str(DEF_OUT_RAW))
    ap.add_argument("--pools-units", choices=["millions","billions"], default=None)
    ap.add_argument("--loans-units", choices=["millions","billions"], default=None)
    ap.add_argument("--break-meta-out", default=None, help="Optional path to write break detection meta (text)")
    ap.add_argument("--household", default=str(DEF_HOUSEHOLD), help="Household mortgages (HHMSDODNS) CSV (always required; used for household share)")
    ap.add_argument("--plmbs", default=None, help="Private-label MBS outstanding CSV (optional; if absent treated as 0)")
    ap.add_argument("--hh-units", choices=["millions","billions"], default=None, help="Override HHMSDODNS units (skip heuristic).")
    ap.add_argument("--auto-fix-hh", action="store_true", help="If share >1 and HH units heuristically scaled, retry with *1000 denominator")
    args = ap.parse_args()

    pools_path = Path(args.pools)
    loans_path = Path(args.loans)
    if not pools_path.exists():
        raise FileNotFoundError(f"Pools CSV not found: {pools_path}")
    if not loans_path.exists():
        if DEF_LOANS_FALLBACK.exists():
            print(f"[WARN] Loans_on primary missing; using fallback: {DEF_LOANS_FALLBACK}")
            loans_path = DEF_LOANS_FALLBACK
        else:
            raise FileNotFoundError(f"Loans CSV not found: {loans_path} (no fallback {DEF_LOANS_FALLBACK})")

    pools_q = read_fred_series_q(pools_path)
    loans_q = read_fred_series_q(loans_path)
    # Always require / load household series (dual-definition output is mandatory now)
    hh_q = safe_read_optional(Path(args.household))
    if hh_q is None:
        raise FileNotFoundError(f"Household mortgages CSV not readable: {args.household}")
    if args.plmbs:
        plmbs_q = safe_read_optional(Path(args.plmbs))
        if plmbs_q is None:
            print(f"[WARN] PLMBS series missing: {args.plmbs}; treating PLMBS=0")
    else:
        print("[INFO] no --plmbs provided; PLMBS assumed 0 for household share")

    pools_b, pools_flag = detect_units_to_billions(pools_q, args.pools_units)
    loans_b, loans_flag = detect_units_to_billions(loans_q, args.loans_units)
    hh_b, hh_flag = detect_units_to_billions(hh_q, args.hh_units)
    if args.hh_units is None:
        hh_b = hh_b * 1000.0
        hh_flag += " -> trillions_to_billions(*1000)"
    if 'plmbs_q' in locals() and plmbs_q is not None:
        plmbs_b, plmbs_flag = detect_units_to_billions(plmbs_q)
    else:
        plmbs_b, plmbs_flag = (pd.Series(dtype=float), "absent (0)")

    idx = pools_b.index.union(loans_b.index)
    idx = idx.union(hh_b.index)
    if not plmbs_b.empty:
        idx = idx.union(plmbs_b.index)
    pools_b = pools_b.reindex(idx)
    loans_b = loans_b.reindex(idx)
    # --- bank_off baseline ---
    denom_bank = loans_b + pools_b
    phi_bank = (pools_b / denom_bank).rename("phi_US_bank_off_raw").dropna()

    # Break detection on bank_off only
    info = detect_break(phi_bank)
    phi_bank_adj = pd.Series(dtype=float)
    scale_factor = None
    if info and math.isfinite(info['scale_pre_to_post']) and info['scale_pre_to_post']>0:
        scale_factor = info['scale_pre_to_post']
        phi_bank_adj = phi_bank.copy()
        phi_bank_adj.loc[phi_bank_adj.index <= info['prev_date']] = phi_bank_adj.loc[phi_bank_adj.index <= info['prev_date']]*scale_factor
        phi_bank_adj.name = 'phi_US_bank_off_adj'
        print(f"[INFO] bank_off break detected {info['break_date'].date()} jump={info['jump']:.3f} scale={scale_factor:.3f}")
        info['applied'] = True
    else:
        if info: info['applied']=False
        print('[INFO] no material bank_off break detected (or scaling invalid)')

    # Optional household share
    phi_house_raw = pd.Series(dtype=float)
    phi_house_adj = pd.Series(dtype=float)
    hh_b = hh_b.reindex(idx)
    if plmbs_b.empty:
        plmbs_full = pd.Series(0.0, index=idx)
    else:
        plmbs_full = plmbs_b.reindex(idx).fillna(0.0)
    num_house = pools_b.fillna(0.0) + plmbs_full
    phi_house_raw = (num_house / hh_b).rename('phi_US_household_share_raw').dropna()
    if args.auto_fix_hh and (phi_house_raw.max() > 1.05) and args.hh_units is None:
        print(f"[WARN] household share max {phi_house_raw.max():.3f} >1; check units")
    if scale_factor:
        phi_house_adj = phi_house_raw.copy()
        phi_house_adj.loc[phi_house_adj.index <= info['prev_date']] = phi_house_adj.loc[phi_house_adj.index <= info['prev_date']]*scale_factor
        phi_house_adj.name = 'phi_US_household_share_adj'


    # --- optional meta sidecar (info already prepared) ---
    if args.break_meta_out:
        meta_path = Path(args.break_meta_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w', encoding='utf-8') as mf:
            if info:
                for k,v in info.items():
                    mf.write(f"{k}: {v}\n")
            else:
                mf.write("applied: False\n")
                mf.write("reason: no_break_detected\n")
        print('[OK] wrote break meta ->', meta_path)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        base = pd.read_csv(out_path, parse_dates=[0])
        dcol = base.columns[0]
        base = base.dropna(subset=[dcol]).set_index(dcol).sort_index()
        base.index = base.index.to_period("Q").to_timestamp("Q")
    else:
        base = pd.DataFrame(index=phi_bank.index)
        base.index.name = "Date"
    # Reindex base to cover all dates we will assign to (pools/loans/HH/PLMBS and phi variants)
    # Be explicit to avoid KeyError on .loc with unseen labels
    need_idx = base.index.union(idx)
    need_idx = need_idx.union(phi_bank.index)
    if not phi_bank_adj.empty:
        need_idx = need_idx.union(phi_bank_adj.index)
    if not phi_house_raw.empty:
        need_idx = need_idx.union(phi_house_raw.index)
    if not phi_house_adj.empty:
        need_idx = need_idx.union(phi_house_adj.index)
    base = base.reindex(need_idx)
    base.loc[phi_bank.index, "Pools_bilUSD"] = pools_b.reindex(phi_bank.index).values
    base.loc[phi_bank.index, "Loans_on_bilUSD"] = loans_b.reindex(phi_bank.index).values
    base.loc[phi_bank.index, "phi_US_bank_off_raw"] = phi_bank.values
    if not phi_bank_adj.empty:
        base.loc[phi_bank_adj.index, 'phi_US_bank_off_adj'] = phi_bank_adj.values
    base.loc[phi_house_raw.index, 'HH_mortgages_bilUSD'] = hh_b.reindex(phi_house_raw.index).values
    if not plmbs_b.empty:
        base.loc[phi_house_raw.index, 'PLMBS_bilUSD'] = plmbs_b.reindex(phi_house_raw.index).values
    base.loc[phi_house_raw.index, 'phi_US_household_share_raw'] = phi_house_raw.values
    if not phi_house_adj.empty:
        base.loc[phi_house_adj.index, 'phi_US_household_share_adj'] = phi_house_adj.values
    # New primary definition: φ = (Agency Pools + PLMBS) / HH mortgages
    # Map to canonical column names phi_US_raw / phi_US_adj for downstream simplicity
    base.loc[phi_house_raw.index, 'phi_US_raw'] = phi_house_raw.values
    if not phi_house_adj.empty:
        base.loc[phi_house_adj.index, 'phi_US_adj'] = phi_house_adj.values
    base = base.sort_index()
    base.reset_index().rename(columns={"index":"Date"}).to_csv(out_path, index=False)
    print("[OK] wrote", out_path)

    last = phi_bank.index.max()
    with open(DEF_AUDIT, "w", encoding="utf-8") as f:
        f.write("US φ bank_off — unit audit\n")
        f.write("mode: bank_off+household (unconditional)\n")
        f.write(f"pools source: {pools_path}\nloans source: {loans_path}\n")
        f.write(f"pools units: {pools_flag}\nloans units: {loans_flag}\n")
        f.write(f"household source: {args.household}\n")
        f.write(f"plmbs source: {args.plmbs or 'None'}\n")
        f.write(f"household units: {hh_flag}\nplmbs units: {plmbs_flag}\n")
        f.write("\n")
        f.write(f"latest quarter: {last.date()}\n")
        f.write(f"Pools (bil): {pools_b.loc[last]:,.3f}\n")
        f.write(f"Loans (bil): {loans_b.loc[last]:,.3f}\n")
        if last in hh_b.index:
            f.write(f"HH mortgages (bil): {hh_b.loc[last]:,.3f}\n")
            if not plmbs_b.empty and last in plmbs_b.index:
                f.write(f"PLMBS (bil): {plmbs_b.loc[last]:,.3f}\n")
        f.write(f"phi_US_bank_off_raw: {phi_bank.loc[last]:.6f}\n")
        if not phi_bank_adj.empty and last in phi_bank_adj.index:
            f.write(f"phi_US_bank_off_adj: {phi_bank_adj.loc[last]:.6f}\n")
        if last in phi_house_raw.index:
            f.write(f"phi_US_household_share_raw: {phi_house_raw.loc[last]:.6f}\n")
            if not phi_house_adj.empty and last in phi_house_adj.index:
                f.write(f"phi_US_household_share_adj: {phi_house_adj.loc[last]:.6f}\n")
        if last in phi_house_raw.index:
            f.write(f"phi_US_raw: {phi_house_raw.loc[last]:.6f}\n")
            if not phi_house_adj.empty and last in phi_house_adj.index:
                f.write(f"phi_US_adj: {phi_house_adj.loc[last]:.6f}\n")
    print("[OK] audit ->", DEF_AUDIT)

if __name__ == "__main__":
    main()
