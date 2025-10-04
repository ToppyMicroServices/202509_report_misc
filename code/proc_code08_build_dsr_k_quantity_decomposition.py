#!/usr/bin/env python3
"""proc_code08_build_dsr_k_quantity_decomposition (rebuilt)

Summary:
- Build an independent US debt-to-income proxy (Z.1 household mortgages divided by FRED DPI).
- Align units: monthly `k` and `dk_*` from proc_code01 are annualised (*12).
- Require the independent DI series for the US unless explicitly allowed otherwise.
- Emit quarterly percentage-point contributions with optional clipping configured via CLI flags.

Outputs:
  data_processed/proc_code08_US_DSR_k_quantity_decomposition.csv
  data_processed/proc_code08_JP_DSR_k_quantity_decomposition.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import io, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data_processed'

US_DSR_PANEL = PROC / 'proc_code03_US_DSR_CreditGrowth_panel.csv'
JP_DSR_PANEL = PROC / 'proc_code03_JP_DSR_CreditGrowth_panel.csv'
US_K_FILE    = PROC / 'proc_code01_k_decomposition_US_quarterly.csv'
JP_K_FILE    = PROC / 'proc_code01_k_decomposition_JP_quarterly.csv'

# --- Network/series helpers ---
UA = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}
DDP_BASE = "https://www.federalreserve.gov/datadownload/Output.aspx"
DDP_PARAMS = {"rel":"Z1","lastobs":"","from":"","to":"","filetype":"csv","label":"omit","layout":"seriescolumn"}
FRED_DPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DPI"

# FRED-specific aliases used as fallbacks when the primary ID fails.
FRED_ALIASES = {
    "FL153165105.Q": "HMLBSHNO",  # Households; Home Mortgages; Liability, Level (Bil. $)
}

def _log(m: str) -> None:
    print(f"[PROC08] {m}")

def _make_session(timeout: float = 60.0) -> requests.Session:
    sess = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    sess.headers.update(UA)
    sess.request_timeout = timeout  # type: ignore[attr-defined]
    return sess

def _normalize_ddp_series_id(series_id: str) -> str:
    s = series_id.strip().upper()
    if s.startswith("BOGZ1"):
        s = s.replace("BOGZ1", "", 1)
    if not s.endswith(".Q") and s != "MORTGAGE30US":
        s = s + ".Q"
    return s

def _fred_id_primary(ddp_id: str) -> str:
    s = _normalize_ddp_series_id(ddp_id)
    return "BOGZ1" + s.replace(".", "")

def _fred_id_alias(ddp_id: str) -> str:
    s = _normalize_ddp_series_id(ddp_id)
    return FRED_ALIASES.get(s, _fred_id_primary(ddp_id))

def _fetch_fred_series_try(fred_id: str, sess: requests.Session, timeout: float) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
    r = sess.get(url, timeout=timeout)
    r.raise_for_status()
    text = r.text
    if text.lstrip().lower().startswith("<!doctype html") or "<html" in text[:200].lower():
        raise RuntimeError(f"FRED returned HTML for {fred_id}")
    df = pd.read_csv(io.StringIO(text))
    lower = {c.lower(): c for c in df.columns}
    dcol = lower.get("date") or lower.get("observation_date") or df.columns[0]
    vcol = next((c for c in df.columns if c != dcol), None)
    if vcol is None:
        raise RuntimeError(f"Unexpected FRED CSV for {fred_id}: {list(df.columns)}")
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.rename(columns={dcol:"Date", vcol:fred_id})
    q = df.set_index("Date")[fred_id].resample("QE-DEC").mean().to_frame().reset_index()
    q[fred_id] = pd.to_numeric(q[fred_id], errors="coerce")
    return q

def _fetch_fred_series(ddp_series_id: str, sess: requests.Session, timeout: float = 60.0) -> pd.DataFrame:
    if ddp_series_id == "MORTGAGE30US":
        return _fetch_fred_series_try("MORTGAGE30US", sess, timeout)
    # Try the BOGZ1-prefixed identifier first, then fall back to known aliases.
    for fred_id in (_fred_id_primary(ddp_series_id), _fred_id_alias(ddp_series_id)):
        try:
            q = _fetch_fred_series_try(fred_id, sess, timeout)
            q = q.rename(columns={fred_id: _normalize_ddp_series_id(ddp_series_id)})
            return q
        except Exception:
            continue
    raise RuntimeError(f"FRED fetch failed for {ddp_series_id}")

def _fetch_ddp_series(series_id: str, sess: requests.Session, timeout: float = 60.0) -> pd.DataFrame:
    sid = _normalize_ddp_series_id(series_id)
    params = DDP_PARAMS.copy(); params["series"] = sid
    try:
        r = sess.get(DDP_BASE, params=params, timeout=timeout)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        time_cols = [c for c in df.columns if str(c).lower().startswith("time")]
        if not time_cols:
            raise RuntimeError("no time col")
        tcol = time_cols[0]
        vcol = [c for c in df.columns if c != tcol][0]
        df = df.rename(columns={tcol:"Date", vcol:sid})
        if df["Date"].astype(str).str.contains("Q").any():
            df["Date"] = pd.PeriodIndex(df["Date"], freq="Q").to_timestamp(how="end")
        else:
            df["Date"] = pd.to_datetime(df["Date"])
        df[sid] = pd.to_numeric(df[sid], errors="coerce")
        return df
    except Exception:
        # Retry with the fully-qualified namespace before falling back to FRED.
        try:
            params2 = params.copy(); params2["series"] = f"Z1/Z1/{sid}"
            r2 = sess.get(DDP_BASE, params=params2, timeout=timeout)
            r2.raise_for_status()
            df2 = pd.read_csv(io.StringIO(r2.text))
            time_cols = [c for c in df2.columns if str(c).lower().startswith("time")]
            if not time_cols:
                raise RuntimeError("no time col 2")
            tcol = time_cols[0]
            vcol = [c for c in df2.columns if c != tcol][0]
            df2 = df2.rename(columns={tcol:"Date", vcol:sid})
            if df2["Date"].astype(str).str.contains("Q").any():
                df2["Date"] = pd.PeriodIndex(df2["Date"], freq="Q").to_timestamp(how="end")
            else:
                df2["Date"] = pd.to_datetime(df2["Date"])
            df2[sid] = pd.to_numeric(df2[sid], errors="coerce")
            return df2
        except Exception:
            return _fetch_fred_series(series_id, sess, timeout=timeout)

def _fred_quarterly_DPI() -> pd.DataFrame:
    sess = _make_session(timeout=60.0)
    r = sess.get(FRED_DPI_URL, timeout=60.0)
    r.raise_for_status()
    txt = r.text
    if txt.lstrip().lower().startswith("<!doctype html") or "<html" in txt[:200].lower():
        raise RuntimeError("FRED returned HTML for DPI")
    df = pd.read_csv(io.StringIO(txt))
    lower = {c.lower(): c for c in df.columns}
    dcol = lower.get("date") or lower.get("observation_date") or df.columns[0]
    vcol = next(c for c in df.columns if c != dcol)
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.rename(columns={dcol:"Date", vcol:"DPI_SAAR"})
    q = df.set_index("Date")["DPI_SAAR"].resample("QE-DEC").mean().to_frame().reset_index()
    q["DPI_SAAR"] = pd.to_numeric(q["DPI_SAAR"], errors="coerce")
    return q

def load_panel(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, index_col=0)
    except Exception:
        return None
    df.index = pd.to_datetime(df.index, errors="coerce").to_period("Q").to_timestamp("Q")
    if "DSR_pct" not in df.columns:
        return None
    out = df[["DSR_pct"]].copy()
    return out[~out.index.duplicated()].sort_index()

def load_k(p: Path, require_term: bool, country: str) -> pd.DataFrame | None:
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=["Date"])
    except Exception:
        return None
    need = {"Date","k","dk_rate"}
    if not need.issubset(df.columns):
        _log(f"{country} k-file missing {need} in {p.name}")
        return None
    if "dk_term" not in df.columns:
        if require_term:
            _log(f"{country} missing dk_term in {p.name} (require_term=True)")
            return None
        else:
            _log(f"{country} dk_term absent; using 0.0")
            df["dk_term"] = 0.0
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    df.index = df.index.to_period("Q").to_timestamp("Q")
    return df[["k","dk_rate","dk_term"]]

def _ann_month_to_annual(x: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(x, float) * 12.0

def _build_di_proxy_us() -> pd.DataFrame | None:
    """US debt-to-income proxy using Z.1 FL153165105.Q (home mortgage liabilities, billions USD)
    and FRED DPI (SAAR, billions USD)."""
    sess = _make_session(timeout=60.0)
    try:
        debt = _fetch_ddp_series("FL153165105.Q", sess, timeout=60.0)
    except Exception as e:
        _log(f"Z1 FL153165105.Q fetch failed: {e}")
        return None
    sid = _normalize_ddp_series_id("FL153165105.Q")
    debt = debt.rename(columns={sid:"HH_home_mort_level"})[["Date","HH_home_mort_level"]].dropna().sort_values("Date")
    try:
        dpi = _fred_quarterly_DPI()
    except Exception as e:
        _log(f"DPI fetch failed: {e}")
        return None
    df = pd.merge(debt, dpi, on="Date", how="inner")
    if df.empty:
        _log("No overlap for Debt/DPI")
        return None
    df["di_proxy"] = pd.to_numeric(df["HH_home_mort_level"], errors="coerce") / pd.to_numeric(df["DPI_SAAR"], errors="coerce")
    df = df[["Date","di_proxy"]].dropna().drop_duplicates("Date").sort_values("Date")
    df["di_proxy"] = df["di_proxy"].clip(0.0, 5.0)
    return df

def build(country: str, panel: pd.DataFrame, kdf: pd.DataFrame, out_path: Path,
          use_independent_di: bool = True, require_independent_di: bool = True,
          caps_pp: dict[str, float | None] | None = None) -> bool:
    # Join DSR with k (inner)
    df = panel.join(kdf, how="inner")
    if df.empty:
        _log(f"{country} no overlap between DSR panel and k -> skip")
        return False

    # Units alignment
    df["k_annual"]       = _ann_month_to_annual(df["k"])
    df["dk_rate_annual"] = _ann_month_to_annual(df["dk_rate"])
    df["dk_term_annual"] = _ann_month_to_annual(df["dk_term"])
    df["DSR_frac"]       = df["DSR_pct"] / 100.0

    # Decide DI source
    di_source = "DSR_over_k"
    if country == "US" and use_independent_di:
        di = _build_di_proxy_us()
        if di is not None and not di.empty:
            df = df.join(di.set_index("Date"), how="inner")
            if "di_proxy" in df.columns and not df["di_proxy"].isna().all():
                df["debt_income_ratio"] = df["di_proxy"].astype(float)
                di_source = "Z1_debt_over_DPI"
    if country == "US" and require_independent_di and di_source != "Z1_debt_over_DPI":
        _log("US requires independent (Debt/DPI) proxy but it was not available. Aborting to avoid DSR/k fallback.")
        return False

    if "debt_income_ratio" not in df.columns:
        # Fallback (JP or when explicitly permitted): infer DI from DSR and k.
        df["debt_income_ratio"] = np.where(df["k_annual"] > 0, df["DSR_frac"] / df["k_annual"], np.nan)
    df["debt_income_ratio"] = pd.to_numeric(df["debt_income_ratio"], errors="coerce").clip(0.0, 5.0)

    # Prepare outputs
    for c in ["delta_DSR_pp","quantity_contrib_pp","k_rate_contrib_pp","k_term_contrib_pp","residual_pp"]:
        df[c] = 0.0

    prev = None
    for idx, row in df.iterrows():
        if prev is None:
            prev = row; continue
        d_curr = row["DSR_frac"]; d_prev = prev["DSR_frac"]
        k_curr = row["k_annual"];  k_prev = prev["k_annual"]
        di_curr = row["debt_income_ratio"]; di_prev = prev["debt_income_ratio"]
        if not all(np.isfinite([d_curr,d_prev,k_curr,k_prev])):
            prev = row; continue

        delta_dsr = d_curr - d_prev
        k_mid = 0.5*(k_curr + k_prev)
        if np.isfinite(di_curr) and np.isfinite(di_prev):
            di_mid  = 0.5*(di_curr + di_prev)
            delta_di = di_curr - di_prev
        else:
            di_mid, delta_di = np.nan, 0.0

        # Quantity
        quantity_contrib = (k_mid * delta_di) if np.isfinite(k_mid) else 0.0
        # k contributions (dk_* already annualised)
        dk_rate = row["dk_rate_annual"]
        dk_term = row["dk_term_annual"]
        k_rate_contrib = (di_mid * dk_rate) if np.isfinite(di_mid) else 0.0
        k_term_contrib = (di_mid * dk_term) if np.isfinite(di_mid) else 0.0

        residual = delta_dsr - (quantity_contrib + k_rate_contrib + k_term_contrib)

        df.loc[idx, "delta_DSR_pp"]        = delta_dsr * 100.0
        df.loc[idx, "quantity_contrib_pp"] = quantity_contrib * 100.0
        df.loc[idx, "k_rate_contrib_pp"]   = k_rate_contrib * 100.0
        df.loc[idx, "k_term_contrib_pp"]   = k_term_contrib * 100.0
        df.loc[idx, "residual_pp"]         = residual * 100.0
        prev = row

    # Sanity check before optional caps (reject if any contribution exceeds +/-5 pp).
    pre_caps = {
        "Q": float(df["quantity_contrib_pp"].abs().max()),
        "R": float(df["k_rate_contrib_pp"].abs().max()),
        "T": float(df["k_term_contrib_pp"].abs().max()),
    }
    if any(v > 5.0 for v in pre_caps.values()):
        _log(f"[ERR] Unrealistic contributions before caps (pp): {pre_caps}. Aborting.")
        return False

    # Final hard caps (pp, optional)
    if caps_pp:
        for c, cap in caps_pp.items():
            if cap is None:
                continue
            if c in df.columns:
                df[c] = df[c].clip(-cap, cap)

    out = df[["DSR_pct","k","k_annual","debt_income_ratio",
              "delta_DSR_pp","quantity_contrib_pp","k_rate_contrib_pp","k_term_contrib_pp","residual_pp"]].reset_index().rename(columns={"index":"Date"})
    out["di_source"] = di_source
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    cap_msg = "none" if not caps_pp or all(v is None for v in caps_pp.values()) else \
        ",".join(f"{k}:{v:.3f}" for k, v in caps_pp.items() if v is not None)
    _log(f"{country} wrote {out_path.name} rows={len(out)} DI={di_source} "
         f"max_abs(pp) Q:{out['quantity_contrib_pp'].abs().max():.3f} "
         f"R:{out['k_rate_contrib_pp'].abs().max():.3f} T:{out['k_term_contrib_pp'].abs().max():.3f} "
         f"caps={cap_msg}")
    return True

def main() -> int:
    ap = argparse.ArgumentParser(description="Build DSR decomposition (quantity vs k rate/term) - robust")
    ap.add_argument("--skip-us", action="store_true")
    ap.add_argument("--skip-jp", action="store_true")
    ap.add_argument("--allow-us-rate-only", action="store_true",
                    help="Proceed even if the US dk_term series is unavailable (sets it to zero).")
    ap.add_argument("--us-di-proxy", action="store_true", default=True,
                    help="Build and use the independent US Debt-to-Income proxy from Z.1 / FRED (default).")
    ap.add_argument("--no-us-di-proxy", dest="us_di_proxy", action="store_false",
                    help="Skip construction of the independent US DI proxy and fall back to DSR/k where allowed.")
    ap.add_argument("--require-us-di", action="store_true", default=True,
                    help="Abort if the independent US DI proxy cannot be built (disallow DSR/k fallback).")
    ap.add_argument("--quantity-cap-pp", type=float, default=None,
                    help="Optional quantity contribution cap in percentage points (omit or negative to disable).")
    ap.add_argument("--rate-cap-pp", type=float, default=None,
                    help="Optional rate contribution cap in percentage points (omit or negative to disable).")
    ap.add_argument("--term-cap-pp", type=float, default=None,
                    help="Optional term contribution cap in percentage points (omit or negative to disable).")
    args = ap.parse_args()

    def cap_value(v: float | None) -> float | None:
        if v is None:
            return None
        return None if v < 0 else v

    caps_pp = {
        "quantity_contrib_pp": cap_value(args.quantity_cap_pp),
        "k_rate_contrib_pp": cap_value(args.rate_cap_pp),
        "k_term_contrib_pp": cap_value(args.term_cap_pp),
    }

    any_ok = False
    if not args.skip_us:
        panel = load_panel(US_DSR_PANEL)
        kdf   = load_k(US_K_FILE, require_term=(not args.allow_us_rate_only), country="US")
        if panel is None or kdf is None:
            _log("US missing panel or k -> skip")
        else:
            ok_us = build("US", panel, kdf, PROC/"proc_code08_US_DSR_k_quantity_decomposition.csv",
                          use_independent_di=args.us_di_proxy, require_independent_di=args.require_us_di,
                          caps_pp=caps_pp)
            any_ok |= ok_us
            if not ok_us and args.require_us_di:
                _log("US build failed because independent DI could not be constructed. Check network/IDs.")

    if not args.skip_jp:
        panel = load_panel(JP_DSR_PANEL)
        kdf   = load_k(JP_K_FILE, require_term=False, country="JP")
        if panel is None or kdf is None:
            _log("JP missing panel or k -> skip")
        else:
            any_ok |= build("JP", panel, kdf, PROC/"proc_code08_JP_DSR_k_quantity_decomposition.csv",
                            use_independent_di=False, require_independent_di=False,
                            caps_pp=caps_pp)

    if not any_ok:
        _log("No decomposition built.")
        return 1
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
