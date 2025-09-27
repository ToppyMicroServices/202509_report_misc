"""fetch_code11_bis_dsr

目的:
  BIS公式の Debt Service Ratio (DSR) 系列をダウンロード / 正規化して
  プロジェクトの合成/legacy 代替として利用できる加工済みCSVを生成する。

方針:
  1) 設定された国ごとのターゲット系列(例: 総民間非金融部門)を API で取得
  2) 四半期インデックス(Date: quarter end) に変換
  3) Value を DSR_pct としてそのまま採用
  4) メタ情報 (Source='BIS', SeriesCode, FetchTimestamp) を付与

注意 / 前提:
  - BIS API の安定構造を仮定。ネットワーク遮断時はスキップし保持ファイルを利用。
  - シリーズコードは国ごとに差異あり; 必要に応じ config を更新。
  - ここでは『民間非金融部門 合計 (Private non-financial sector, Total)』想定の place holder。
    実際の BIS コードは最新ドキュメントで要確認。

出力:
  raw: data_raw/bis_dsr_{country}.csv  (取得素のCSVレスポンス)
  processed merged: data_processed/fetch_bis_DSR_JP_US_panel.csv

CLI:
  python code/fetch_code11_bis_dsr.py --countries JP US --force
  python code/fetch_code11_bis_dsr.py --offline   (既存 raw を再加工)

あとで行うべき改善 (TODO):
  - 正式な BIS DSR コード一覧自動取得 (メタデータ API 呼び出し)
  - 他セクター(HH, NFC) 分離列
  - validation (ギャップ, 範囲, NA, データ更新差分) レポート
"""
from __future__ import annotations
import argparse, json, sys, io
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests

RAW_DIR = Path('data_raw')
PROC_DIR = Path('data_processed')
OUT_PANEL = PROC_DIR / 'fetch_bis_DSR_JP_US_panel.csv'
META_JSON = PROC_DIR / 'fetch_bis_DSR_fetchmeta.json'

# 仮のシリーズコード定義: 実際の BIS API 形式は要確認 (PLACEHOLDER)
# 想定 URL: https://stats.bis.org/api/v1/bis/DSR/Q/{REF_AREA}.PSSN.PNF.T.A?format=csv 等 (例構造)
# 下記 series_code はテンプレ; 正式コードは BIS Stats Explorer で確認して書き換え。
SERIES_MAP = {
    'JP': 'Q.JP.T.PNF.T.A',  # JP = Japan, T=Total PNF=Private Non-Financial (placeholder)
    'US': 'Q.US.T.PNF.T.A',  # US = United States (placeholder)
}

BASE_URL = 'https://stats.bis.org/api/v1/bis/DSR/'  # 末尾に series_code + '?format=csv'

# Conservative headers to avoid 406 (content negotiation) and basic bot blocks
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Accept": "text/csv,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

def fetch_series(country: str, code: str, timeout=20):
    url = BASE_URL + code + '?format=csv'
    try:
        with requests.Session() as s:
            r = s.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            # Include short body snippet for diagnostics (HTML error page etc.)
            snippet = r.text[:200] if isinstance(r.text, str) else ''
            return None, f'status {r.status_code}' + (f' body={snippet!r}' if snippet else '')
        raw_path = RAW_DIR / f'bis_dsr_{country}.csv'
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(r.content)
        return raw_path, None
    except Exception as e:
        return None, str(e)

def parse_raw(country: str, raw_path: Path):
    try:
        df = pd.read_csv(raw_path)
    except Exception as e:
        raise RuntimeError(f'parse failed {country}: {e}')
    # 想定: 列に TIME_PERIOD, OBS_VALUE 等 (BIS SDMX 典型). 汎用的に柔軟対応。
    # 候補列探索
    time_col = None
    for c in df.columns:
        if c.lower() in ('time_period','time','period'):
            time_col = c; break
    if time_col is None:
        # fallback: 最初の列
        time_col = df.columns[0]
    value_col = None
    for c in df.columns:
        if c.lower() in ('obs_value','value','observed_value'):
            value_col = c; break
    if value_col is None:
        # heuristic: numeric dtype columns
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise RuntimeError('no numeric value column detected')
        value_col = num_cols[-1]
    # 時間パース (例: 2024-Q3)
    dates = pd.to_datetime(df[time_col].astype(str).str.replace('Q','-Q'), errors='coerce')
    # 失敗多数なら自前解析
    if dates.isna().mean() > 0.5:
        parsed = []
        for v in df[time_col].astype(str):
            if 'Q' in v:
                y, q = v.split('Q')
                q = q.strip(' -_')
                try:
                    pr = pd.Period(f'{int(y)}Q{int(q)}', freq='Q')
                    parsed.append(pr.to_timestamp('Q'))
                except Exception:
                    parsed.append(pd.NaT)
            else:
                try:
                    parsed.append(pd.to_datetime(v))
                except Exception:
                    parsed.append(pd.NaT)
        dates = pd.Series(parsed)
    df['Date'] = dates
    df['DSR_pct'] = pd.to_numeric(df[value_col], errors='coerce')
    out = df[['Date','DSR_pct']].dropna().drop_duplicates('Date')
    out['Country'] = country
    out = out.sort_values('Date').set_index('Date')
    return out

def build_panel(dfs):
    if not dfs:
        return pd.DataFrame()
    panel = pd.concat(dfs, axis=0).sort_index()
    return panel

def main(argv=None):
    ap = argparse.ArgumentParser(description='Fetch BIS DSR series (JP/US)')
    ap.add_argument('--countries', nargs='+', default=['JP','US'])
    ap.add_argument('--offline', action='store_true', help='Skip download; parse existing raw files')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    collected = []
    meta = {'fetched': [], 'errors': [], 'timestamp': datetime.utcnow().isoformat()+'Z'}

    for c in args.countries:
        code = SERIES_MAP.get(c)
        if not code:
            meta['errors'].append({'country': c, 'error': 'no series code mapping'})
            continue
        raw_path = RAW_DIR / f'bis_dsr_{c}.csv'
        if not args.offline:
            rp, err = fetch_series(c, code)
            if err:
                meta['errors'].append({'country': c, 'error': err})
            else:
                meta['fetched'].append({'country': c, 'path': str(rp), 'code': code})
        if raw_path.exists():
            try:
                df_country = parse_raw(c, raw_path)
                collected.append(df_country)
            except Exception as e:
                meta['errors'].append({'country': c, 'error': str(e)})
        else:
            meta['errors'].append({'country': c, 'error': 'raw file missing'})

    panel = build_panel(collected)
    if panel.empty:
        print('[ERR] no data parsed; see meta JSON')
        # Practical fallback guidance (English)
        print('[MANUAL] BIS DSR portal export fallback:')
        print('  1) Open https://data.bis.org/topics/dsr in your browser')
        print("  2) Use 'Export' to download CSV for the desired countries/sector (e.g., PNFS)")
        print("  3) Save the downloaded file into data_raw/ with its original name (e.g. bis_dp_search_export_2025Q3.csv)")
        print('  4) Downstream builders (proc_code03_*) will pick it up automatically')
        print('     Or re-run this script with --offline to skip the network step')
    else:
        # カノニカル列名調整: 国ごとDSRをpivotも同時出力
        panel_long = panel.copy()
        panel_long['ProxyFlag'] = 9  # 9=official external source
        panel_long['Source'] = 'BIS'
        # ロング形式保存
        panel_long.reset_index().to_csv(OUT_PANEL, index=False)
        # ピボット (オプション) も追加保存
        pivot = panel_long.reset_index().pivot(index='Date', columns='Country', values='DSR_pct')
        PIVOT_OUT = PROC_DIR / 'fetch_bis_DSR_wide.csv'
        pivot.to_csv(PIVOT_OUT)
        meta['panel_rows'] = int(len(panel_long))
        meta['start'] = panel_long.index.min().strftime('%Y-%m-%d') if len(panel_long)>0 else None
        meta['end'] = panel_long.index.max().strftime('%Y-%m-%d') if len(panel_long)>0 else None
        print('[OK] wrote', OUT_PANEL, 'rows', len(panel_long))
    META_JSON.write_text(json.dumps(meta, indent=2))
    print('[INFO] meta ->', META_JSON)
    if meta['errors']:
        print('[WARN] errors encountered:', meta['errors'])
    return 0 if panel is not None else 1

if __name__ == '__main__':
    raise SystemExit(main())
