#!/usr/bin/env python3
"""scan_jhf_wac_wam_candidates

目的:
  data_raw/jhf_factors_xlsx 以下の全 .xlsx (デフォルト) を走査し、シート内テキストセルから
  WAC (加重平均クーポン) / WAM (加重平均残存期間) 推定に使えそうなラベル候補を抽出。

探索対象キーワード (大文字小文字無視):
  coupon, interest, rate, 利率, クーポン, 平均, average, avg, maturity, remaining, 残存, 期間, life, duration, term

抽出ルール:
  1) 文字列セル value に上記キーワードのいずれかを含む
  2) そのセルの右隣 or 下隣が数値 (float/int) なら value をラベル候補として記録
  3) 右/下が両方数値でない場合でも、ラベルのみ記録 (後で手作業レビュー用)

出力:
  data_processed/jhf_wac_wam_scan_candidates.csv
    columns: file,sheet,row_idx,col_idx,label,right_value,below_value,num_right,num_below
  data_processed/jhf_wac_wam_scan_summary.json
    集計: label 正規化 (小文字) ごとの出現回数と数値隣接ヒット内訳

注意:
  - 大規模ファイル対策で各シートは最大 max_rows スキャン (デフォルト 300 行) まで。
  - エラーは標準出力に [WARN] として継続。
  - .xls (バイナリ) は依存 (xlrd) 無しのため無視 (オプション指定で追加可)。

次ステップ例:
  - CSV を見て WAC: 例えば 'Coupon Rate' / '利率(%)' / 'Interest Rate' の右隣数値列番号を特定
  - WAM: 'Average Remaining Months' / 'Avg Remaining Maturity' / '平均残存期間' 等を特定
  - マッピング定義を別スクリプトで作成し、四半期集計 (期末 Date をどう与えるか: 発行日 or 公表基準日) を設計。

CLI:
  python code/scan_jhf_wac_wam_candidates.py --xlsx-glob data_raw/jhf_factors_xlsx/*.xlsx

"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data_processed'

KEYWORDS = [
    'coupon','interest','rate','利率','クーポン','平均','average','avg','maturity','remaining','残存','期間','life','duration','term'
]
KW_PATTERN = re.compile('|'.join(re.escape(k) for k in KEYWORDS), re.IGNORECASE)

def is_number(x):
    try:
        if x is None: return False
        if isinstance(x,str) and (x.strip()=='' or x.strip().lower() in {'nan','na','n/a','--'}):
            return False
        float(x)
        return True
    except Exception:
        return False

def scan_file(path: Path, max_rows: int):
    out_rows = []
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        print(f"[WARN] open fail {path.name}: {e}")
        return out_rows
    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet_name=sheet, header=None)
        except Exception as e:
            print(f"[WARN] parse fail {path.name}:{sheet}: {e}")
            continue
        nrows = min(len(df), max_rows)
        ncols = len(df.columns)
        for r in range(nrows):
            for c in range(ncols):
                val = df.iat[r,c]
                if not isinstance(val,str):
                    continue
                if not KW_PATTERN.search(val):
                    continue
                right = df.iat[r,c+1] if c+1 < ncols else None
                below = df.iat[r+1,c] if r+1 < nrows else None
                num_right = is_number(right)
                num_below = is_number(below)
                # 軽いフィルタ: ラベルに数字しか無い場合 skip
                if re.fullmatch(r"\s*[0-9.%-]+\s*", val):
                    continue
                out_rows.append({
                    'file': path.name,
                    'sheet': sheet,
                    'row_idx': r,
                    'col_idx': c,
                    'label': val.strip(),
                    'right_value': right if (isinstance(right,(int,float)) or (isinstance(right,str) and len(right)<50)) else None,
                    'below_value': below if (isinstance(below,(int,float)) or (isinstance(below,str) and len(below)<50)) else None,
                    'num_right': int(num_right),
                    'num_below': int(num_below),
                })
    return out_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx-glob', default='data_raw/jhf_factors_xlsx/*.xlsx')
    ap.add_argument('--max-rows', type=int, default=300, help='各シート最大走査行')
    args = ap.parse_args()
    files = sorted(Path().glob(args.xlsx_glob))
    if not files:
        print(f"[INFO] no files matched glob: {args.xlsx_glob}")
        return 0
    PROC.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for f in files:
        rows = scan_file(f, args.max_rows)
        all_rows.extend(rows)
    if not all_rows:
        print('[INFO] no candidate labels found')
        return 0
    df = pd.DataFrame(all_rows)
    out_csv = PROC / 'jhf_wac_wam_scan_candidates.csv'
    df.to_csv(out_csv, index=False)
    # Summary
    norm = df['label'].str.lower().str.replace('\s+',' ', regex=True)
    df['label_norm'] = norm
    grp = df.groupby('label_norm').agg(
        occurrences=('label_norm','count'),
        any_num_neighbor=('num_right', 'max'),
        any_num_neighbor2=('num_below','max')
    ).reset_index()
    grp['any_numeric_adjacent'] = grp[['any_num_neighbor','any_num_neighbor2']].max(axis=1)
    summary = grp.sort_values(['any_numeric_adjacent','occurrences'], ascending=[False,False]).to_dict(orient='records')
    out_json = PROC / 'jhf_wac_wam_scan_summary.json'
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] candidates -> {out_csv.name} rows={len(df)} unique_labels={len(grp)}")
    print(f"[OK] summary    -> {out_json.name}")
    # 上位数件プレビュー
    preview = [r for r in summary[:10]]
    print('[PREVIEW top10 label_norm]:')
    for r in preview:
        print('  -', r['label_norm'], 'occ=', r['occurrences'], 'numeric_adj=', r['any_numeric_adjacent'])
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
