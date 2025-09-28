#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fig_code04_phi_offbalance_fixed_v2.py
#
# Fixed-style figure; also auto-detects JP files and columns.

import pandas as pd, matplotlib.pyplot as plt, re, argparse, math, sys
import matplotlib.dates as mdates
import matplotlib as mpl
from pathlib import Path
from glob import glob

def _read_csv_flex(path):
    df = pd.read_csv(path)
    # Date column candidates
    for c in ['Date','date','observation_date','DATE']:
        if c in df.columns:
            x = pd.to_datetime(df[c].astype(str), errors='coerce')
            break
    else:
        # fallback: first column is date
        first = df.columns[0]
        x = pd.to_datetime(df[first].astype(str), errors='coerce')
    return x, df

def _find_phi_col(df, country='US'):
    cols = list(df.columns)
    pats = []
    if country.upper()=='US':
        pats = [
            r'^phi[_ ]?US[_ ]?observed[_ ]?proxy$',
            r'^phi[_ ]?US$',
            r'^phi[_ ]?US',
            r'^phi.*observed.*proxy$',
            r'^phi$'
        ]
    else:
        pats = [
            r'^phi[_ ]?JP[_ ]?observed[_ ]?proxy$',
            r'^phi[_ ]?JP$',
            r'^phi[_ ]?JP',
            r'^phi_JP$',
            r'^phi.*observed.*proxy$',
            r'^phi$'
        ]
    for p in pats:
        rgx = re.compile(p, re.I)
        for c in cols:
            if rgx.match(c):
                return c
    # last resort: any column starting with 'phi'
    for c in cols:
        if str(c).lower().startswith('phi'):
            return c
    return None

def _detect_jp_files():
    # Prefer single-file official JP phi first
    preferred_single = Path('data_processed/proc_code02_JP_phi_quarterly.csv')
    if preferred_single.exists():
        return None, None, preferred_single
    # Otherwise consider pre/post observed
    pre_candidates  = sorted(glob('data_processed/*JP*pre*.csv')) + sorted(glob('data_processed/proc_code04_JP_phi_series_observed_pre.csv'))
    post_candidates = sorted(glob('data_processed/*JP*post*.csv')) + sorted(glob('data_processed/proc_code04_JP_phi_series_observed_post.csv'))
    # Generic single-file fallbacks
    single_candidates = sorted(glob('data_processed/*JP*observed*.csv')) + sorted(glob('data_processed/*JP*.csv'))
    pre = Path(pre_candidates[-1]) if pre_candidates else None
    post = Path(post_candidates[-1]) if post_candidates else None
    single = None if (pre or post) else (Path(single_candidates[-1]) if single_candidates else None)
    if not (pre or post or single):
        merged = Path('data_processed/proc_merge_US_JP_phi_household.csv')
        if merged.exists():
            single = merged
    return pre, post, single


US_PRE  = Path('data_processed/proc_code04_US_phi_series_observed_pre.csv')
US_POST = Path('data_processed/proc_code04_US_phi_series_observed_post.csv')

BOUNDARY = pd.Timestamp('2010-03-31')

def _splice_scale(series: pd.Series, boundary=BOUNDARY, window:int=4, anchor:str='post'):
    """Compute scale k around boundary so that pre/post means align.
    anchor='post' returns k=post_mean/pre_mean (scale pre side); anchor='pre' returns k=pre_mean/post_mean (scale post side).
    Returns float k; 1.0 if not computable.
    """
    if series is None or series.empty:
        return 1.0
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return 1.0
    pre = s.loc[:boundary].dropna()
    post = s.loc[boundary:].dropna()
    if pre.empty or post.empty:
        return 1.0
    pre_vals = pre.iloc[-window:] if len(pre) >= window else pre
    post_vals = post.iloc[:window] if len(post) >= window else post
    pre_mean = float(pre_vals.mean()) if len(pre_vals) else float('nan')
    post_mean = float(post_vals.mean()) if len(post_vals) else float('nan')
    if not (math.isfinite(pre_mean) and math.isfinite(post_mean)) or pre_mean==0 or post_mean==0:
        return 1.0
    return (post_mean / pre_mean) if anchor=='post' else (pre_mean / post_mean)

def _apply_splice(series: pd.Series, k: float, boundary=BOUNDARY, anchor:str='post') -> pd.Series:
    if series is None or series.empty or k == 1.0:
        return series
    s = pd.to_numeric(series, errors='coerce').dropna()
    pre = s.loc[:boundary]
    post = s.loc[boundary:]
    if pre.empty or post.empty:
        return s
    if anchor == 'post':
        pre_adj = pre * k
        return pd.concat([pre_adj, post]).sort_index()
    else:
        post_adj = post * k
        return pd.concat([pre, post_adj]).sort_index()

def _load_us_phi_pre_post_from_raw(raw_path=Path('data_processed/proc_code04_US_phi_series_raw.csv')):
    """Fallback: build two US series (pre-anchored, post-anchored) from proc_code04 raw output.
    Preference of base column: phi_US_adj > phi_US_raw > first column containing 'phi'.
    If phi_US_adj exists, use it as post-anchored; derive pre-anchored by reverse scaling.
    Otherwise compute both anchors from the base series by local mean scaling.
    Returns (x_pre, y_pre, x_post, y_post) or (None,... ) if unavailable.
    """
    if not raw_path.exists():
        return None, None, None, None
    df = pd.read_csv(raw_path)
    dcol = 'Date' if 'Date' in df.columns else ('date' if 'date' in df.columns else df.columns[0])
    x = pd.to_datetime(df[dcol], errors='coerce')
    df_i = df.set_index(pd.to_datetime(df[dcol], errors='coerce')).sort_index()
    base_col = None
    for c in ['phi_US_adj','phi_US_raw']:
        if c in df_i.columns:
            base_col = c; break
    if base_col is None:
        for c in df_i.columns:
            if 'phi' in str(c).lower():
                base_col = c; break
    if base_col is None:
        return None, None, None, None
    s_base = pd.to_numeric(df_i[base_col], errors='coerce').dropna()
    # Build post-anchored
    if 'phi_US_adj' in df_i.columns:
        s_post = pd.to_numeric(df_i['phi_US_adj'], errors='coerce').dropna()
        # Derive pre-anchored by reverse scaling
        k_pre = _splice_scale(s_base, anchor='pre')
        s_pre = _apply_splice(s_base, k_pre, anchor='pre')
    else:
        # Derive both from base
        k_post = _splice_scale(s_base, anchor='post')
        s_post = _apply_splice(s_base, k_post, anchor='post')
        k_pre = _splice_scale(s_base, anchor='pre')
        s_pre = _apply_splice(s_base, k_pre, anchor='pre')
    return s_pre.index.to_pydatetime(), s_pre.values, s_post.index.to_pydatetime(), s_post.values

def _read_fred_like_series(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    # Pick date and value columns flexibly
    dcol = None
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ('date','observation_date'):
            dcol = c; break
    if dcol is None:
        dcol = df.columns[0]
    vcol = None
    for c in df.columns:
        if c != dcol:
            vcol = c; break
    if vcol is None:
        vcol = df.columns[-1]
    t = pd.to_datetime(df[dcol], errors='coerce')
    v = pd.to_numeric(df[vcol], errors='coerce')
    s = pd.Series(v.values, index=t).sort_index()
    s = s[s.index.notna()]
    return s

def _to_quarter_bil(s: pd.Series) -> pd.Series:
    # Resample to quarter end (Q-DEC) using last observation, then unit normalize to billions.
    if s.index.inferred_type not in ('datetime64','datetime'):
        s = s.copy()
    sq = s.resample('QE-DEC').last()
    med = float(pd.Series(sq).dropna().median()) if sq.notna().any() else 0.0
    # Heuristic: >1e4 treat as millions -> /1000, <10 treat as trillions -> *1000, else billions as-is
    if med > 1e4:
        sq = sq / 1000.0
    elif med < 10 and med > 0:
        sq = sq * 1000.0
    return sq

def _build_us_phi_pre_post_from_data_raw(raw_dir=Path('data_raw')):
    pools_p = raw_dir / 'AGSEBMPTCMAHDFS.csv'
    home_p  = raw_dir / 'HHMSDODNS.csv'
    plmbs_p = raw_dir / 'BOGZ1FL673065105Q.csv'
    if not pools_p.exists() or not home_p.exists():
        return None, None, None, None
    pools = _to_quarter_bil(_read_fred_like_series(pools_p))
    home  = _to_quarter_bil(_read_fred_like_series(home_p))
    if plmbs_p.exists():
        plmbs = _to_quarter_bil(_read_fred_like_series(plmbs_p))
    else:
        plmbs = pd.Series(0.0, index=home.index)
    idx = home.index
    pools_al = pools.reindex(idx).ffill()
    plmbs_al = plmbs.reindex(idx).ffill()
    num = pools_al + plmbs_al
    # Pre/post anchored via strict splice on numerator
    k_post = _splice_scale(num, anchor='post')
    num_post = _apply_splice(num, k_post, anchor='post')
    k_pre = _splice_scale(num, anchor='pre')
    num_pre = _apply_splice(num, k_pre, anchor='pre')
    phi_post = (num_post / home.reindex(num_post.index)).dropna()
    phi_pre  = (num_pre  / home.reindex(num_pre.index)).dropna()
    return phi_pre.index.to_pydatetime(), phi_pre.values, phi_post.index.to_pydatetime(), phi_post.values

def _filter_after(x, series, start_ts):
    mask = x >= start_ts
    return x[mask], series.loc[mask]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-date', default='1980-01-01', help='Plot start date (default 1980-01-01).')
    ap.add_argument('--out', default='figures/fig_code04_US_JP_phi_offbalance_fixed.png')
    # Compatibility switches (ignored unless usable); allow orchestrator to pass --jp
    ap.add_argument('--jp', default=None, help='Optional JP φ CSV (if given, overrides auto-detect).')
    ap.add_argument('--us-pre', default=None, help='Optional US pre CSV path (observed).')
    ap.add_argument('--us-post', default=None, help='Optional US post CSV path (observed).')
    # Display control: default is save-only (no show). Use --show to open a window explicitly.
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--show', dest='show', action='store_true', help='Force show() after saving.')
    g.add_argument('--no-show', dest='show', action='store_false', help='Do not show() (useful for batch).')
    ap.set_defaults(show=False)
    ap.add_argument('--backend', default=None, help='Matplotlib backend to use for interactive show (e.g., MacOSX, TkAgg, Qt5Agg).')
    args = ap.parse_args()
    start_ts = pd.Timestamp(args.start_date)
    # --- US series: prefer explicit args, then default pre/post files, then fallback from raw ---
    us_pre_path = Path(args.us_pre) if args.us_pre else (US_PRE if US_PRE.exists() else None)
    us_post_path = Path(args.us_post) if args.us_post else (US_POST if US_POST.exists() else None)
    if us_pre_path and us_post_path and us_pre_path.exists() and us_post_path.exists():
        x_us_pre,  df_us_pre  = _read_csv_flex(us_pre_path)
        x_us_post, df_us_post = _read_csv_flex(us_post_path)
        col_us_pre  = _find_phi_col(df_us_pre,  'US')
        col_us_post = _find_phi_col(df_us_post, 'US')
        x_us_pre_f,  y_us_pre  = _filter_after(x_us_pre,  df_us_pre[col_us_pre],  start_ts) if col_us_pre  else ([], [])
        x_us_post_f, y_us_post = _filter_after(x_us_post, df_us_post[col_us_post], start_ts) if col_us_post else ([], [])
    else:
        x_pre_f, y_pre, x_post_f, y_post = _load_us_phi_pre_post_from_raw()
        if x_pre_f is None:
            # Last resort: build directly from data_raw inputs
            x_pre_f, y_pre, x_post_f, y_post = _build_us_phi_pre_post_from_data_raw()
        if x_pre_f is None:
            raise SystemExit('Missing US pre/post and no usable fallback. Provide data_raw/AGSEBMPTCMAHDFS.csv and HHMSDODNS.csv (optional BOGZ1FL673065105Q.csv).')
        # Convert numpy arrays to pandas Series aligned
        x_us_pre_f, y_us_pre = pd.to_datetime(pd.Index(x_pre_f)), pd.Series(y_pre, index=pd.to_datetime(pd.Index(x_pre_f)))
        x_us_post_f, y_us_post = pd.to_datetime(pd.Index(x_post_f)), pd.Series(y_post, index=pd.to_datetime(pd.Index(x_post_f)))
        # Apply start-date filter
        keep_pre = x_us_pre_f >= start_ts
        keep_post = x_us_post_f >= start_ts
        x_us_pre_f, y_us_pre = x_us_pre_f[keep_pre], y_us_pre.loc[keep_pre]
        x_us_post_f, y_us_post = x_us_post_f[keep_post], y_us_post.loc[keep_post]
    fig, ax = plt.subplots(figsize=(10,5))
    # Marker thinning interval for dense series
    markevery = 5
    # Colors/markers for better print distinction (avoid name collision with column names)
    color_us_pre = '#D55E00'; color_us_post = '#E69F00'; color_jp = 'tab:blue'
    m_us_pre = 'o'; m_us_post = 's'; m_jp = '^'
    # Plot US (reverted to full-series plotting with markers)
    if len(x_us_pre_f) > 0:
        ax.plot(x_us_pre_f, pd.to_numeric(y_us_pre, errors='coerce'), '-', marker=m_us_pre, markevery=markevery, ms=5, lw=2.0, color=color_us_pre, label='United States φ (pre)')
    if len(x_us_post_f) > 0:
        ax.plot(x_us_post_f, pd.to_numeric(y_us_post, errors='coerce'), '--', marker=m_us_post, markevery=markevery, ms=5, lw=2.0, color=color_us_post, alpha=0.95, label='United States φ (post)')

    # JP
    # --- JP series: allow explicit override via --jp, else auto-detect ---
    if args.jp:
        jp_csv = Path(args.jp)
        if jp_csv.exists():
            x_jp, df_jp = _read_csv_flex(jp_csv)
            jp_col = _find_phi_col(df_jp, 'JP')
            if jp_col:
                x_jp_f, y_jp = _filter_after(x_jp, df_jp[jp_col], start_ts)
                if len(x_jp_f) > 0:
                    s_obs = pd.Series(pd.to_numeric(y_jp, errors='coerce').values, index=x_jp_f).dropna()
                    q_index = pd.date_range(start=x_jp_f.min(), end=x_jp_f.max(), freq='QE-DEC')
                    s_line = s_obs.reindex(q_index).ffill()
                    ax.plot(s_line.index, s_line.values, '-', lw=2.0, marker=m_jp, markevery=markevery, ms=5, color=color_jp, label='Japan φ')
                    if not s_obs.empty:
                        ax.plot(s_obs.index, s_obs.values, m_jp, markevery=markevery, ms=5, color=color_jp, label='_nolegend_')
        else:
            print(f"[WARN] --jp path not found: {jp_csv}; falling back to auto-detect")
        jp_pre, jp_post, jp_single = (None, None, None)  # prevent auto block below from re-plotting
    else:
        jp_pre, jp_post, jp_single = _detect_jp_files()
    if jp_single is not None:
        x_jp, df_jp = _read_csv_flex(jp_single)
        jp_col = _find_phi_col(df_jp, 'JP')
        if jp_col:
            x_jp_f, y_jp = _filter_after(x_jp, df_jp[jp_col], start_ts)
            if len(x_jp_f) > 0:
                # Build continuous quarterly line by forward filling; markers only at observed points
                s_obs = pd.Series(pd.to_numeric(y_jp, errors='coerce').values, index=x_jp_f).dropna()
                q_index = pd.date_range(start=x_jp_f.min(), end=x_jp_f.max(), freq='QE-DEC')
                s_line = s_obs.reindex(q_index).ffill()
                ax.plot(s_line.index, s_line.values, '-', lw=2.0, marker=m_jp, markevery=markevery, ms=5, color=color_jp, label='Japan φ')
                if not s_obs.empty:
                    ax.plot(s_obs.index, s_obs.values, m_jp, markevery=markevery, ms=5, color=color_jp, label='_nolegend_')
    else:
        if jp_pre is not None:
            x_jp_pre, df_jp_pre = _read_csv_flex(jp_pre)
            col_jp_pre_name = _find_phi_col(df_jp_pre, 'JP')
            if col_jp_pre_name:
                x_jp_pre_f, y_jp_pre = _filter_after(x_jp_pre, df_jp_pre[col_jp_pre_name], start_ts)
                if len(x_jp_pre_f) > 0:
                    q_idx = pd.date_range(start=x_jp_pre_f.min(), end=x_jp_pre_f.max(), freq='QE-DEC')
                    s = pd.Series(y_jp_pre.values, index=x_jp_pre_f)
                    s_full = s.reindex(q_idx).interpolate(limit_direction='both')
                    ax.plot(s_full.index, s_full.values, '-', lw=2.0, marker=m_jp, markevery=markevery, ms=5, color=color_jp, label='Japan φ (pre)')
                    ax.plot(x_jp_pre_f, y_jp_pre, m_jp, markevery=markevery, ms=5, color=color_jp, label='_nolegend_')
        if jp_post is not None:
            x_jp_post, df_jp_post = _read_csv_flex(jp_post)
            col_jp_post_name = _find_phi_col(df_jp_post, 'JP')
            if col_jp_post_name:
                x_jp_post_f, y_jp_post = _filter_after(x_jp_post, df_jp_post[col_jp_post_name], start_ts)
                if len(x_jp_post_f) > 0:
                    s_obs = pd.Series(pd.to_numeric(y_jp_post, errors='coerce').values, index=x_jp_post_f).dropna()
                    q_index = pd.date_range(start=x_jp_post_f.min(), end=x_jp_post_f.max(), freq='QE-DEC')
                    s_line = s_obs.reindex(q_index).ffill()
                    ax.plot(s_line.index, s_line.values, '-', lw=2.0, marker=m_jp, markevery=markevery, ms=5, color=color_jp, label='Japan φ (post)')
                    if not s_obs.empty:
                        ax.plot(s_obs.index, s_obs.values, m_jp, markevery=markevery, ms=5, color=color_jp, label='_nolegend_')
                    q_idx = pd.date_range(start=x_jp_post_f.min(), end=x_jp_post_f.max(), freq='QE-DEC')
                    s = pd.Series(y_jp_post.values, index=x_jp_post_f)
                    s_full = s.reindex(q_idx).interpolate(limit_direction='both')
                    ax.plot(s_full.index, s_full.values, '-', lw=2.0, color=color_jp, label='_nolegend_')
                    ax.plot(x_jp_post_f, y_jp_post, m_jp, markevery=markevery, ms=5, color=color_jp, label='_nolegend_')

    # Labeled vertical line and nicer ticks
    ax.axvline(pd.Timestamp('2010-03-31'), linestyle='--', color='gray', alpha=0.7, label='Accounting rule change (2010Q1)')
    ax.set_title('Off-balance share (φ): US vs Japan')
    ax.set_xlabel('Date'); ax.set_ylabel('φ (share)'); ax.set_ylim(0,1)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        # Order: US pre → Rule change → US post → Japan*
        seen = {}
        ordered = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
                ordered.append((h, l))
        def pick(label_query):
            for i,(h,l) in enumerate(ordered):
                if l == label_query:
                    return ordered.pop(i)
            return None
        final = []
        for key in ['United States φ (pre)', 'Accounting rule change (2010Q1)', 'United States φ (post)']:
            item = pick(key)
            if item: final.append(item)
        rest = [(h,l) for (h,l) in ordered if 'Japan' in l]
        others = [(h,l) for (h,l) in ordered if 'Japan' not in l]
        final.extend(rest)
        final.extend(others)
    ax.legend([h for h,_ in final], [l for _,l in final], loc='upper left', ncol=1, frameon=True)
    Path('figures').mkdir(parents=True, exist_ok=True)
    # Enforce fig_code04_ prefix on output filename
    out_png = Path(args.out)
    if out_png.name and not out_png.name.startswith('fig_code04_'):
        out_png = out_png.with_name('fig_code04_' + out_png.name)
    # Force final figure size before saving (inches)
    try:
        fig.set_size_inches(10, 5, forward=True)
    except Exception:
        pass
    # Report final size for verification
    try:
        w_in, h_in = fig.get_size_inches()
        dpi = 220
        print(f"[INFO] final figsize: {w_in:.2f}x{h_in:.2f} inches, ~{int(w_in*dpi)}x{int(h_in*dpi)} px @ {dpi} dpi")
    except Exception:
        dpi = 220
    fig.tight_layout(); fig.savefig(out_png, dpi=dpi)
    print(f'[OK] saved {out_png}')
    # Display only when explicitly requested via --show
    do_show = bool(args.show)
    if do_show:
        # Ensure an interactive backend when showing
        try:
            current = mpl.get_backend()
        except Exception:
            current = ''
        def _try_switch(name: str) -> bool:
            try:
                if name:
                    plt.switch_backend(name)
                    print(f"[INFO] backend -> {name}")
                    return True
            except Exception:
                pass
            return False
        # If explicit backend is provided, try it first
        if args.backend and not _try_switch(args.backend):
            print(f"[WARN] failed to switch backend to {args.backend}; trying auto candidates")
        # Auto-fallback if current backend looks non-interactive (Agg/Template/etc.)
        backend_l = str(current).lower()
        if ('agg' in backend_l) or ('template' in backend_l) or (args.backend is not None and mpl.get_backend() != args.backend):
            for cand in ('MacOSX','TkAgg','Qt5Agg'):
                if _try_switch(cand):
                    break
        try:
            plt.show(block=True)
        except Exception as e:
            print(f"[WARN] show() failed: {e}")

if __name__ == '__main__':
    main()
