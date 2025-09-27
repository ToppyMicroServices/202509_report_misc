import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from util_code01_lib_io import find, safe_savefig, ensure_unique, figure_name_with_code

OUTDIR = Path("figures")

def make():
    preferred = [
        Path("data_processed/JP_RMBS_components_semiannual_JSDA_2012_2025.csv"),
        Path("../data_processed/JP_RMBS_components_semiannual_JSDA_2012_2025.csv"),
    ]
    chosen = None
    for p in preferred:
        if p.exists():
            chosen = p; break
    if chosen is None:
        files = find(["JP_RMBS_components_semiannual_JSDA_2012_2025.csv"]) or []
        if files:
            files = sorted(files, key=lambda x: (0 if "data_processed" in str(x) else 1, len(str(x))))
            chosen = files[0]
    if chosen is None:
        print("[JSDA comps] skip: CSV not found"); return
    df = pd.read_csv(chosen)
    df.columns = [c.strip().lower() for c in df.columns]
    dtc = [c for c in df.columns if "date" in c or "month" in c]
    if not dtc:
        print("[JSDA comps] skip: date col missing"); return
    dtc = dtc[0]
    jhf = [c for c in df.columns if "jhf" in c]
    prv = [c for c in df.columns if "priv" in c]
    tot = [c for c in df.columns if "total" in c]
    if (not tot) and jhf and prv:
        df["total_100m_yen"] = df[jhf[0]] + df[prv[0]]
        tot = ["total_100m_yen"]
    df[dtc] = pd.to_datetime(df[dtc], errors="coerce")
    df = df.dropna(subset=[dtc]).sort_values(dtc)
    fig, ax = plt.subplots(figsize=(7,4))
    if jhf:
        ax.plot(df[dtc], df[jhf[0]] * 0.0001, marker="o", label="JHF (trillion yen)")
    if prv:
        ax.plot(df[dtc], df[prv[0]] * 0.0001, marker="o", label="Private (trillion yen)")
    if tot:
        ax.plot(df[dtc], df[tot[0]] * 0.0001, marker="o", label="Total (trillion yen)")
    ax.set_title("Japan RMBS components (semiannual, JSDA)")
    ax.set_ylabel("Units: trillion yen")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    target = figure_name_with_code(__file__, OUTDIR/"JP_RMBS_components_semiannual_JSDA_2012_2025_FINAL.png")
    unique = ensure_unique(target)
    safe_savefig(fig, unique)
    print("[JSDA comps] read:", chosen)
    print("WROTE:", unique)

if __name__=="__main__":
    make()
