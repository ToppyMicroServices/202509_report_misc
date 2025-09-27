import argparse, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from util_code01_lib_io import safe_savefig, figure_name_with_code, find

def _auto_csv():
    # Progressive preference order (new -> legacy):
    candidates_priority = [
        Path('data_processed/proc_code05_JP_US_MBS_RMBS_to_NGDP_compare_2012_2021.csv'),  # new prefixed comparison slice
        Path('data_processed/JP_US_MBS_to_AnnualGDP_2012_2021.csv'),  # legacy
    ]
    for p in candidates_priority:
        if p.exists():
            return p
    # broader pattern search (legacy variants)
    found = find(['proc_code05_JP_US_MBS_RMBS_to_NGDP_compare_*.csv','JP_US_MBS_to_AnnualGDP*.csv']) or []
    return found[0] if found else None

def main():
    ap = argparse.ArgumentParser(description='Dual-axis MBS/RMBS to GDP figure (auto CSV detection)')
    ap.add_argument("--csv", default=None, help="CSV (auto-detect if omitted)")
    ap.add_argument("--out", default=None, help="output PNG (auto if omitted)")
    ap.add_argument("--title", default="MBS/RMBS to GDP (US vs JP)")
    args = ap.parse_args()
    # (2025-09) 要望により無引数時の複数バリアント出力を廃止: 常に1枚のみ保存

    csv_path = Path(args.csv) if args.csv else _auto_csv()
    if csv_path is None or not csv_path.exists():
        print('[ERROR] no CSV found (specify --csv)')
        return 1
    # Load CSV with flexible date column detection
    raw = pd.read_csv(csv_path)
    # Detect date-like column
    cand_cols = [c for c in raw.columns if c.lower() in ("date","time","quarter","period")] or [raw.columns[0]]
    date_col = cand_cols[0]
    raw[date_col] = pd.to_datetime(raw[date_col], errors='coerce')
    raw = raw.dropna(subset=[date_col])
    df = raw.rename(columns={date_col: 'Date'})
    # Harmonize expected column names
    def _match_col(candidates):
        """Return first matching actual column for any candidate token.

        Normalization rules:
        - case-insensitive
        - remove underscores and non-alphanumerics
        - treat '%' and 'pct' equivalently
        - allow substring containment as fallback
        """
        import re
        def norm(s:str):
            s = s.lower()
            s = s.replace('%','pct')  # unify % -> pct
            s = re.sub(r'[^a-z0-9]','', s)
            return s
        actual_norm_map = {norm(c): c for c in df.columns}
        # exact normalized match
        for cand in candidates:
            nc = norm(cand)
            if nc in actual_norm_map:
                return actual_norm_map[nc]
        # substring fallback (normalized)
        for cand in candidates:
            nc = norm(cand)
            for k,v in actual_norm_map.items():
                if nc in k:
                    return v
        return None

    us_col = _match_col([
        'US_mbs_gdp_pct', 'US_MBS_to_GDP_%', 'US_MBS_to_GDP_pct', 'US_MBS_to_NGDP_%', 'US_MBS_to_AnnualGDP_pct'
    ])
    jp_col = _match_col([
        'JP_rmbs_gdp_pct', 'JP_RMBS_to_GDP_%', 'JP_RMBS_to_GDP_pct', 'JP_RMBS_to_NGDP_%', 'JP_RMBS_to_AnnualGDP_pct'
    ])
    if us_col and us_col != 'US_mbs_gdp_pct':
        df = df.rename(columns={us_col:'US_mbs_gdp_pct'})
    if jp_col and jp_col != 'JP_rmbs_gdp_pct':
        df = df.rename(columns={jp_col:'JP_rmbs_gdp_pct'})
    if 'US_mbs_gdp_pct' not in df.columns and 'JP_rmbs_gdp_pct' not in df.columns:
        print('[ERROR] Expected US and/or JP MBS/RMBS ratio columns not found. Columns present:', list(df.columns))
        return 1
    fig, ax1 = plt.subplots(figsize=(11,4.6))
    ax2 = ax1.twinx()
    n_points = len(df)
    if n_points < 15:
        print(f"[INFO] few data points: {n_points} (annual) — increasing marker size for visibility")
    ms = 8 if n_points < 15 else 5
    lw = 2.2
    handles = []
    labels = []
    proc05_used = 'proc_code05' in str(csv_path)
    if "US_mbs_gdp_pct" in df.columns:
        label_us = "US MBS/GDP (%)" + (" (proc_code05)" if proc05_used else "")
        (l1,) = ax1.plot(df["Date"], df["US_mbs_gdp_pct"], marker="o", ms=ms, lw=lw,
                          label=label_us, color="#E69F00")
        ax1.set_ylabel("US MBS/GDP (%)", color='black')
        handles.append(l1); labels.append(l1.get_label())
        # annotate start/end
        ax1.annotate(f"{df['US_mbs_gdp_pct'].iloc[0]:.1f}", xy=(df['Date'].iloc[0], df['US_mbs_gdp_pct'].iloc[0]),
                     xytext=(0,8), textcoords='offset points', ha='center', fontsize=8, color='#E69F00')
        ax1.annotate(f"{df['US_mbs_gdp_pct'].iloc[-1]:.1f}", xy=(df['Date'].iloc[-1], df['US_mbs_gdp_pct'].iloc[-1]),
                     xytext=(0,8), textcoords='offset points', ha='center', fontsize=8, color='#E69F00')
    if "JP_rmbs_gdp_pct" in df.columns:
        label_jp = "Japan RMBS/GDP (%)" + (" (proc_code05)" if proc05_used else "")
        (l2,) = ax2.plot(df["Date"], df["JP_rmbs_gdp_pct"], marker="s", ms=ms, lw=lw,
                          label=label_jp, color="#56B4E9")
        ax2.set_ylabel("JP RMBS/GDP (%)", color='black')
        handles.append(l2); labels.append(l2.get_label())
        ax2.annotate(f"{df['JP_rmbs_gdp_pct'].iloc[0]:.2f}", xy=(df['Date'].iloc[0], df['JP_rmbs_gdp_pct'].iloc[0]),
                     xytext=(0,-12), textcoords='offset points', ha='center', fontsize=8, color='#56B4E9')
        ax2.annotate(f"{df['JP_rmbs_gdp_pct'].iloc[-1]:.2f}", xy=(df['Date'].iloc[-1], df['JP_rmbs_gdp_pct'].iloc[-1]),
                     xytext=(0,-12), textcoords='offset points', ha='center', fontsize=8, color='#56B4E9')
    if "US_mbs_gdp_pct" not in df.columns:
        ax1.text(0.5,0.5,'US series missing', transform=ax1.transAxes, ha='center', va='center', fontsize=9, color='gray')
    if "JP_rmbs_gdp_pct" not in df.columns:
        ax2.text(0.5,0.5,'JP series missing', transform=ax2.transAxes, ha='center', va='center', fontsize=9, color='gray')
    # legend (single box)
    if handles:
        ax1.legend(handles, labels, frameon=True, facecolor='white', edgecolor='black', fontsize=9, loc='upper left')
    ax1.set_title(args.title, color='black')
    ax1.grid(True, axis="y", alpha=0.3, linestyle='--', linewidth=0.7)
    for spine in ax1.spines.values():
        spine.set_color('black')
    for spine in ax2.spines.values():
        spine.set_color('black')
    fig.tight_layout()
    def _save_with_stats(base_name: str):
        # Build output path (respect --out override, otherwise derive from base_name)
        out_path = Path(f'figures/{base_name}.png') if not args.out else Path(args.out)
        out_path = figure_name_with_code(__file__, out_path)
        saved = safe_savefig(fig, out_path)
        try:
            from PIL import Image
            import numpy as np
            im = Image.open(saved)
            arr = np.array(im.convert('RGB'))
            nonwhite_ratio = 1 - np.mean((arr > 245).all(axis=2))
            size_kb = saved.stat().st_size / 1024.0
            print(f"saved: {saved}\n[FIG_STATS] file_kb={size_kb:.1f} nonwhite_ratio={nonwhite_ratio:.4f}")
        except Exception as e:
            print("saved:", saved, "(stats unavailable:", e, ")")
        return saved
    _save_with_stats('MBS_RMBS_to_GDP_dualaxis')
    return 0

if __name__ == "__main__":
    main()
