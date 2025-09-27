#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runall_proc_code.py

目的:
    論文/図生成に必要な主要 processed CSV を一括構築する統合ビルダー。
    (k decomposition, φ (JP/US), annual MBS/RMBS ratios, JHF RMBS vs GDP, DSR panels, BIS DSR fetch)

実行例:
    python code/runall_proc_code.py --skip-dsr --verbose
    python code/runall_proc_code.py --skip-bis-dsr (BISオンライン取得省略)

出力(既存各 proc スクリプトが生成するもの):
  data_processed/proc_code01_k_decomposition_*.csv (trim 等)
  data_processed/proc_code02_JP_phi_quarterly.csv
  data_processed/proc_code04_US_phi_series_*.csv
  data_processed/proc_code05_*_MBS_to_NGDP_annual.csv など
  data_processed/proc_code06_JP_JHF_RMBS_vs_GDP_*.csv
  data_processed/proc_code03_*_DSR_CreditGrowth_panel.csv (DSRパネル)

特徴:
    - 冪等: 失敗しても後続を継続 (--fail-fast で即終了可)
    - 進捗ログ (--quiet で抑制)
    - 個別スキップフラグ (--skip-xxx)
    - BIS DSR フェッチ (fetch_code11_bis_dsr.py) を追加 (JP/US) / ネット障害時も続行
    - 生成ファイル存在チェック & 簡易サマリー

将来拡張: データソースダウンロード統合, キャッシュ制御.
"""
from __future__ import annotations
import argparse, subprocess, shutil, sys, time, threading, queue
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / 'code'
PROC = ROOT / 'data_processed'
RAW  = ROOT / 'data_raw'
PY = sys.executable  # 現在実行中の Python を子プロセスにも利用 (pyenv 環境で 'python' 無い問題対策)

STEPS = [
    # 1) proc_code01 all-in-one flow (JP WAC/WAM build if needed -> k decomposition US/JP -> JP trim)
    ('k_decomp_allinone', 'proc_code01_all_in_one.py', ''),
    ('phi_jp',    'proc_code02_build_phi_quarterly.py',         ''),
    ('dsr_panels','proc_code03_build_dsr_creditgrowth_panels.py',''),
    ('phi_us',    'proc_code04_build_us_phi_series.py',         ''),
    ('annual_rat','proc_code05_build_annual_mbs_rmbs_ratios.py',''),
    ('jhf_vs_gdp','proc_code06_build_jp_jhf_rmbs_vs_gdp.py',    ''),
    # RMBS/JHF -> GDP (standard builder)
    ('rmbs_total_raw','proc_code06_build_jp_jhf_rmbs_vs_gdp.py',''),
    # DSR quantity vs k (rate/term) decomposition
    ('dsr_k_qty','proc_code08_build_dsr_k_quantity_decomposition.py',''),
    # interaction panel (phi x capital thinness -> next-quarter credit growth)
    ('phi_capital_inter','proc_code09_build_phi_capital_interaction_panel.py',''),
]

SKIP_FLAGS = {
    'k_decomp_allinone':'skip_k',
    'phi_jp':'skip_phi_jp', 'dsr_panels':'skip_dsr',
    'phi_us':'skip_phi_us', 'annual_rat':'skip_annual', 'jhf_vs_gdp':'skip_jhf',
    'rmbs_total_raw':'skip_rmbs_total_raw', 'dsr_k_qty':'skip_dsr_k_qty', 'phi_capital_inter':'skip_phi_capital_inter'
}

# NOTE: 一部 (adjusted 系 / US 年次比率) は未実装のため期待リストから除外
#       実装後に追加する。
EXPECT_FILES = [
    'proc_code02_JP_phi_quarterly.csv',
    'proc_code04_US_phi_series_raw.csv',
    # 'proc_code04_US_phi_series_adjusted.csv',  # TODO (未実装)
    # 'proc_code05_US_MBS_to_NGDP_annual.csv',   # TODO (US データ未整備)
    'proc_code05_JP_RMBS_to_NGDP_annual.csv',
    'proc_code06_JP_JHF_RMBS_vs_GDP_quarterly.csv',
    'proc_code03_US_DSR_CreditGrowth_panel.csv',
    'proc_code03_JP_DSR_CreditGrowth_panel.csv',
]
# map expected file -> producing step key (for skip-aware missing logic)
EXPECT_PRODUCER = {
    'proc_code02_JP_phi_quarterly.csv':'phi_jp',
    'proc_code04_US_phi_series_raw.csv':'phi_us',
    'proc_code05_JP_RMBS_to_NGDP_annual.csv':'annual_rat',
    'proc_code06_JP_JHF_RMBS_vs_GDP_quarterly.csv':'jhf_vs_gdp',
    'proc_code03_US_DSR_CreditGrowth_panel.csv':'dsr_panels',
    'proc_code03_JP_DSR_CreditGrowth_panel.csv':'dsr_panels'
}
OPTIONAL_BIS_FILES = ['proc_code03_bis_DSR_JP_US_panel.csv','proc_code03_bis_DSR_wide.csv']

COLORS = {
    'ok':'\033[32m', 'warn':'\033[33m', 'err':'\033[31m', 'reset':'\033[0m'
}

def run_cmd(label: str, cmd: list[str], fail_fast: bool, quiet: bool, live: bool=False, timeout: float|None=None) -> bool:
    """Run a command.

    live=True で標準出力/標準エラーをリアルタイム表示 (フリーズ誤認防止)。
    timeout 秒指定で経過後に強制終了 (子プロセス kill)。
    """
    t0 = time.time()
    if not quiet:
        print(f"[RUN][{label}] {' '.join(cmd)}")
    if not live:
        try:
            cp = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"{COLORS['err']}[TIMEOUT][{label}] exceeded {timeout}s -> terminate{COLORS['reset']}")
            return False
        except Exception as e:
            print(f"{COLORS['err']}[FAIL][{label}] spawn error: {e}{COLORS['reset']}")
            return False
        dur = time.time() - t0
        if cp.returncode == 0:
            if not quiet:
                print(f"{COLORS['ok']}[OK][{label}] {dur:.1f}s{COLORS['reset']}")
                if cp.stdout.strip():
                    print(cp.stdout.strip())
        else:
            print(f"{COLORS['err']}[ERR][{label}] rc={cp.returncode} {dur:.1f}s{COLORS['reset']}")
            if cp.stdout.strip():
                print(cp.stdout.strip())
            if cp.stderr.strip():
                print(cp.stderr.strip())
            if fail_fast:
                sys.exit(1)
            return False
        return True
    # live streaming branch
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    except Exception as e:
        print(f"{COLORS['err']}[FAIL][{label}] spawn error (live): {e}{COLORS['reset']}")
        return False

    q: queue.Queue[tuple[str,str]] = queue.Queue()

    def _reader(stream, name):
        for line in iter(stream.readline, ''):
            q.put((name, line.rstrip('\n')))
        stream.close()

    threads = [threading.Thread(target=_reader, args=(proc.stdout, 'O'), daemon=True),
               threading.Thread(target=_reader, args=(proc.stderr, 'E'), daemon=True)]
    for th in threads:
        th.start()

    timed_out = False
    while True:
        try:
            item = q.get(timeout=0.1)
            if not quiet:
                prefix = '' if item[0]=='O' else '[stderr] '
                print(prefix + item[1])
        except queue.Empty:
            pass
        ret = proc.poll()
        if ret is not None and q.empty():
            break
        if timeout and (time.time() - t0) > timeout:
            proc.kill()
            timed_out = True
            break
    dur = time.time() - t0
    rc = proc.returncode if not timed_out else -9
    if timed_out:
        print(f"{COLORS['err']}[TIMEOUT][{label}] killed after {timeout}s{COLORS['reset']}")
        return False
    if rc == 0:
        if not quiet:
            print(f"{COLORS['ok']}[OK][{label}] {dur:.1f}s (live){COLORS['reset']}")
        return True
    else:
        print(f"{COLORS['err']}[ERR][{label}] rc={rc} {dur:.1f}s (live){COLORS['reset']}")
        if fail_fast:
            sys.exit(1)
        return False

def main():
    ap = argparse.ArgumentParser(description='Build all core processed datasets (integrated pipeline + auto fetch)')
    ap.add_argument('--fail-fast', action='store_true', help='最初の失敗で停止')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--verbose', action='store_true', help='成功時stdoutも全表示')
    ap.add_argument('--skip-fetch', action='store_true', help='fetch_* 自動実行を全てスキップ')
    ap.add_argument('--fetch-strict', action='store_true', help='fetch 段階で1つでも失敗したら即終了')
    ap.add_argument('--live-output', action='store_true', help='fetch 段階など長時間処理の標準出力をリアルタイム表示')
    ap.add_argument('--skip-bis-dsr', action='store_true', help='BIS DSR fetch_code11_bis_dsr.py をスキップ')
    ap.add_argument('--bis-offline', action='store_true', help='BIS DSR をオフライン再加工モードで実行 (--offline)')
    ap.add_argument('--cmd-timeout', type=float, help='各サブコマンドの最大実行秒数 (超過で kill)')
    # Avoid duplicate flags (multiple steps may map to same skip flag like --skip_k)
    added_flags = set()
    for _,flag in SKIP_FLAGS.items():
        if flag not in added_flags:
            ap.add_argument(f'--{flag}', action='store_true')
            added_flags.add(flag)
    args = ap.parse_args()

    # 1. Fetch stage (raw + some processed). バックアップコピーは行わない。
    fetch_scripts: list[tuple[str,list[str]]] = [
        ('fetch_capital_k',      [PY, str(CODE/'fetch_code01_capital_k_data.py'), '--out', 'data_raw']),
        ('fetch_jhf_convert',    [PY, str(CODE/'fetch_code02_convert_jhf_xls_to_xlsx.py')]),
        ('fetch_jsda_components',[PY, str(CODE/'fetch_code03_extract_jsda_rmbs_components.py')]),
        ('fetch_us_mbs_gdp',     [PY, str(CODE/'fetch_code05_fred_us_mbs_gdp.py')]),
        ('fetch_imf_gdp',        [PY, str(CODE/'fetch_code07_imf_jp_gdp.py'), '--start','2010','--end','2025','--out','data_processed/fetch_code07_JP_IMF_GDP_quarterly.csv']),
        # pooled & household loan stocks for φ_US builder prerequisites (if not already local)
        ('fetch_fred_pools',     [PY, str(CODE/'fetch_code08_fred_series.py'), 'AGSEBMPTCMAHDFS']),
        ('fetch_fred_loans',     [PY, str(CODE/'fetch_code08_fred_series.py'), 'HHMSDODNS']),
    ]
    # BIS DSR fetch (optional network). Place at end of fetch stage.
    if not args.skip_bis_dsr:
        bis_cmd = [PY, str(CODE/'fetch_code11_bis_dsr.py'), '--countries','JP','US']
        if args.bis_offline:
            bis_cmd.append('--offline')
        fetch_scripts.append(('fetch_bis_dsr', bis_cmd))
    elif not args.quiet:
        print('[INFO] skip-bis-dsr 指定のため BIS DSR fetch をスキップ')
    fetch_results: dict[str,str] = {}
    # Map fetch label to expected raw file(s)
    fetch_raw_map = {
        'fetch_capital_k': ['MORTGAGE30US.csv','CET1_RWA_US_WB.csv','CET1_RWA_JP_WB.csv'],
        'fetch_jhf_convert': [],
        'fetch_jsda_components': ['JP_JHF_RMBS_vs_GDP.csv'],
        'fetch_us_mbs_gdp': ['AGSEBMPTCMAHDFS.csv','GDP.csv'],
        'fetch_imf_gdp': ['JP_IMF_GDP_quarterly.csv'],
        'fetch_fred_pools': ['AGSEBMPTCMAHDFS.csv'],
        'fetch_fred_loans': ['HHMSDODNS.csv'],
        'fetch_bis_dsr': ['bis_dp_search_export_JP.csv','bis_dp_search_export_US.csv'],
    }
    if not args.skip_fetch:
        for label, cmd in fetch_scripts:
            ok = run_cmd(label, cmd, args.fetch_strict, args.quiet and not args.verbose, live=args.live_output, timeout=args.cmd_timeout)
            # Check if expected raw file(s) exist if fetch failed
            if not ok:
                raw_files = fetch_raw_map.get(label,[])
                found = False
                for rf in raw_files:
                    if (RAW/rf).exists():
                        found = True
                        break
                if found:
                    print(f"[WARN][{label}] fetch failed but data_raw/{rf} exists; continuing.")
                    fetch_results[label] = 'WARN'
                else:
                    fetch_results[label] = 'ERR'
                    if args.fetch_strict:
                        print('[FATAL] fetch strict mode: abort.')
                        sys.exit(2)
            else:
                fetch_results[label] = 'OK'
            # Retry logic for FRED pools/loans
            if not ok and label in ('fetch_fred_pools','fetch_fred_loans'):
                if not args.quiet:
                    print(f"[RETRY-EXTERNAL] {label} second attempt...")
                ok = run_cmd(label + '_retry', cmd, args.fetch_strict, args.quiet and not args.verbose, live=args.live_output, timeout=args.cmd_timeout)
                if not ok:
                    raw_files = fetch_raw_map.get(label,[])
                    found = False
                    for rf in raw_files:
                        if (RAW/rf).exists():
                            found = True
                            break
                    if found:
                        print(f"[WARN][{label}_retry] fetch failed but data_raw/{rf} exists; continuing.")
                        fetch_results[label+'_retry'] = 'WARN'
                    else:
                        fetch_results[label+'_retry'] = 'ERR'
                        if args.fetch_strict:
                            print('[FATAL] fetch strict mode: abort.')
                            sys.exit(2)
                else:
                    fetch_results[label+'_retry'] = 'OK'
    else:
        if not args.quiet:
            print('[INFO] skip-fetch 指定のため fetch_* 実行をスキップ')

    # 2. Process stage
    results: dict[str,str] = {}
    for key, script, extra in STEPS:
        skip_flag = SKIP_FLAGS[key]
        if getattr(args, skip_flag):
            results[key] = 'SKIP'
            if not args.quiet:
                print(f"[SKIP][{key}] --{skip_flag}")
            continue
        script_path = CODE / script
        if not script_path.is_file():
            results[key] = 'MISSING'
            print(f"{COLORS['warn']}[MISS][{key}] {script} not found{COLORS['reset']}")
            continue
        cmd = [PY, str(script_path)] + (extra.split() if extra else [])
        ok = run_cmd(key, cmd, args.fail_fast, args.quiet and not args.verbose, live=args.live_output, timeout=args.cmd_timeout)
        results[key] = 'OK' if ok else 'ERR'

    # Summary
    print("\n===== SUMMARY (runall) =====")
    if fetch_results:
        print('  [FETCH]')
        for k,v in fetch_results.items():
            print(f"    {k:18s}: {v}")
    print('  [PROCESS]')
    for k in [k for k,_ ,_ in STEPS]:
        print(f"  {k:14s} : {results.get(k,'?')}")

    # Existence check (required)
    print("\n[CHECK] Expected processed files (required):")
    missing=0
    for fn in EXPECT_FILES:
        p = PROC / fn
        producer = EXPECT_PRODUCER.get(fn)
        if p.exists():
            print(f"  {COLORS['ok']}OK{COLORS['reset']} {fn}")
        else:
            # If the producing step was skipped, mark as SKIP not counted as missing
            if producer and results.get(producer) == 'SKIP':
                print(f"  {COLORS['warn']}SKIP{COLORS['reset']} {fn} (producing step skipped)")
            else:
                print(f"  {COLORS['warn']}MISS{COLORS['reset']} {fn}")
                missing+=1
    if missing==0:
        print("[CHECK] All required processed outputs present.")
    else:
        print(f"[CHECK] Missing {missing} required file(s).")

    # Optional BIS fetch outputs (only if fetch attempted)
    if 'fetch_bis_dsr' in fetch_results:
        print("\n[CHECK][optional] BIS DSR fetch outputs:")
        for fn in OPTIONAL_BIS_FILES:
            p = PROC / fn
            if p.exists():
                print(f"  {COLORS['ok']}OK{COLORS['reset']} {fn}")
            else:
                print(f"  {COLORS['warn']}MISS(opt){COLORS['reset']} {fn}")

if __name__ == '__main__':
    main()
