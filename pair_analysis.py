#!/usr/bin/env python3
"""
通貨ペア別の利益率分析スクリプト
"""

import subprocess
import json
from datetime import datetime, timezone
import os
import re
import zipfile
from glob import glob
import argparse

class PairAnalyzer:
    def __init__(self, timerange: str = "20231030-20240427", top_n: int = 0, use_existing_results: bool = True):
        self.pairs = [
            "BTC/USDT", "ETH/USDT", "LINK/USDT", "XRP/USDT",
            "ADA/USDT", "DOT/USDT", "LTC/USDT", "BCH/USDT",
            "BNB/USDT", "SOL/USDT", "AVAX/USDT", "UNI/USDT",
            "ATOM/USDT", "NEAR/USDT", "ALGO/USDT", "VET/USDT",
            "ICP/USDT", "FIL/USDT"
        ]
        self.results = []
        # Ensure paths resolve correctly regardless of where this script is launched
        self.freqtrade_dir = os.path.dirname(__file__)
        self.config_path = os.path.join(self.freqtrade_dir, "user_data", "config.json")
        self.timerange = timerange
        self.top_n = max(0, int(top_n or 0))
        self.use_existing_results = bool(use_existing_results)

    def _parse_backtest_output(self, output: str):
        """Parse freqtrade backtesting stdout to extract key metrics.
        Returns dict with trades, total_profit (USDT), and total_profit_percent.
        """
        summary = {
            'trades': 0,
            'profit': 0.0,
            'profit_ratio': 0.0,
        }

        # Total trades
        m = re.search(r'Total/Daily Avg Trades\s+\│\s+(\d+)\s+/', output)
        if m:
            summary['trades'] = int(m.group(1))

        # Absolute profit in USDT
        m = re.search(r'Absolute profit\s+\│\s+([-\d.]+)\s+USDT', output)
        if m:
            summary['profit'] = float(m.group(1))

        # Total profit %
        m = re.search(r'Total profit %\s+\│\s+([-\d.]+)\s*%', output)
        if m:
            summary['profit_ratio'] = float(m.group(1))

        return summary

    def run_backtest_for_pair(self, pair):
        """個別の通貨ペアでバックテストを実行"""
        print(f"🔍 {pair} のバックテストを実行中...")

        # 一時的にconfig.jsonを更新（元の設定を保存して最後に復元）
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        original_whitelist = config.get('exchange', {}).get('pair_whitelist', [])
        config['exchange']['pair_whitelist'] = [pair]
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

        try:
            # バックテストを実行
            cmd = [
                "freqtrade", "backtesting",
                "--config", self.config_path,
                "--strategy", "ichiV1",
                "--timerange", self.timerange,
                "--export", "trades"
            ]

            # Run from the freqtrade project dir so default user_data paths resolve
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.freqtrade_dir)

            if result.returncode == 0:
                summary = self._parse_backtest_output(result.stdout)

                self.results.append({
                    'pair': pair,
                    'trades': summary['trades'],
                    'profit': summary['profit'],
                    'profit_ratio': summary['profit_ratio']
                })

                print(f"✅ {pair}: {summary['profit_ratio']:.2f}% ({summary['profit']:.2f} USDT, {summary['trades']}取引)")

            else:
                print(f"❌ {pair}: エラー")
                if result.stderr:
                    print(result.stderr)
                if result.stdout:
                    print(result.stdout)
                self.results.append({
                    'pair': pair,
                    'trades': 0,
                    'profit': 0,
                    'profit_ratio': 0
                })

        except Exception as e:
            print(f"❌ {pair}: エラー - {e}")
            self.results.append({
                'pair': pair,
                'trades': 0,
                'profit': 0,
                'profit_ratio': 0
            })
        finally:
            # 設定を復元
            try:
                with open(self.config_path, 'r') as f:
                    cfg = json.load(f)
                cfg['exchange']['pair_whitelist'] = original_whitelist
                with open(self.config_path, 'w') as f:
                    json.dump(cfg, f, indent=2)
            except Exception:
                pass

    def analyze_all_pairs(self):
        """全ての通貨ペアを分析"""
        print("🚀 通貨ペア別利益率分析を開始します...")
        print("=" * 60)

        # まず既存のバックテスト結果からの解析を試みる（高速）
        if self.use_existing_results and self._analyze_from_existing_results():
            self.results.sort(key=lambda x: x['profit_ratio'], reverse=True)
            self.display_results()
            self.save_results()
            return

        # 単一回のバックテストで全ペアを集計（推奨）
        if self._run_single_backtest_all_pairs(timerange=self.timerange):
            self.results.sort(key=lambda x: x['profit_ratio'], reverse=True)
            self.display_results()
            self.save_results()
            return

        # フォールバック: ペア毎に個別実行
        for pair in self.pairs:
            self.run_backtest_for_pair(pair)
            print("-" * 40)

        # 結果を利益率でソート
        self.results.sort(key=lambda x: x['profit_ratio'], reverse=True)

        # 結果を表示
        self.display_results()

        # 結果を保存
        self.save_results()

    def _list_active_usdt_pairs(self, exchange: str = 'binance') -> set:
        """freqtrade CLIを使って取引可能なUSDT建てアクティブ通貨ペア一覧を取得"""
        try:
            cmd = [
                "freqtrade", "list-pairs",
                "--exchange", exchange,
                "--quote", "USDT",
                "--print-json",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=self.freqtrade_dir)
            if res.returncode != 0:
                return set()
            lines = res.stdout.strip().splitlines()
            json_line = next((ln for ln in reversed(lines) if ln.strip().startswith('[') and ln.strip().endswith(']')), '[]')
            pairs = json.loads(json_line)
            return set(pairs)
        except Exception:
            return set()

    def update_config_with_top_pairs(self, exchange: str = 'binance', verify_exchange: bool = True) -> list:
        """self.results を元にトップNペアで config の pair_whitelist を更新し、反映したリストを返す。"""
        if not self.results:
            print("⚠️ 先に分析結果がありません。バックテスト解析を先に実行してください。")
            return []

    def optimize_topn(self, timerange: str = None, max_n: int = 0) -> list:
        """指定期間において、トップK(1..N)でバックテストを実行し、
        Total profit %（全体の利益率）が最大となるKを探索する。
        戻り値: [{'n':K,'profit_pct':..,'profit_abs':..,'trades':..}]（Kごとの一覧）。
        推奨Nはこの配列の中でprofit_pct最大のもの。
        """
        timerange = timerange or self.timerange
        # まずランキングの元となる全ペア一括の結果を用意
        if not self.results:
            if self.use_existing_results and self._analyze_from_existing_results():
                pass
            else:
                ok = self._run_single_backtest_all_pairs(timerange=timerange)
                if not ok:
                    print("❌ 最適N探索の前提となる一括バックテストに失敗")
                    return []

        # ランキング上位の順序に従ってKを増やして評価
        ordered_pairs = [r['pair'] for r in sorted(self.results, key=lambda x: x['profit_ratio'], reverse=True)]
        if max_n <= 0:
            max_n = len(ordered_pairs)

        results_dir = os.path.join(self.freqtrade_dir, 'user_data', 'backtest_results')
        evals = []

        # バックアップ
        try:
            with open(self.config_path, 'r') as f:
                cfg_backup = json.load(f)
        except Exception as e:
            print(f"⚠️ config 読み込み失敗: {e}")
            return []

        try:
            for k in range(1, max_n + 1):
                subset = ordered_pairs[:k]
                # config反映
                cfg = dict(cfg_backup)
                cfg.setdefault('exchange', {})['pair_whitelist'] = subset
                with open(self.config_path, 'w') as f:
                    json.dump(cfg, f, indent=2)

                pre = set(glob(os.path.join(results_dir, 'backtest-result-*.zip')))
                cmd = [
                    "freqtrade", "backtesting",
                    "--config", self.config_path,
                    "--strategy", "ichiV1",
                    "--timerange", timerange,
                    "--export", "trades"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, cwd=self.freqtrade_dir)
                if res.returncode != 0:
                    print(f"⚠️ K={k} のバックテスト失敗。STDERR:\n{res.stderr}")
                    continue
                post = set(glob(os.path.join(results_dir, 'backtest-result-*.zip')))
                newz = sorted(list(post - pre))
                target = newz[-1] if newz else (sorted(list(post))[-1] if post else None)
                if not target:
                    print(f"⚠️ K={k} で結果zipが見つかりません")
                    continue
                # zipから合算利益率を抽出
                with zipfile.ZipFile(target) as z:
                    jnames = [n for n in z.namelist() if n.endswith('.json') and 'config' not in n]
                    with z.open(jnames[0]) as f:
                        data = json.load(f)
                sdict = data.get('strategy', {})
                sname = next(iter(sdict)) if isinstance(sdict, dict) else 'ichiV1'
                details = sdict[sname] if isinstance(sdict, dict) else sdict
                profit_pct = float(details.get('profit_total_pct', 0.0))
                profit_abs = float(details.get('profit_total_abs', 0.0))
                trades = int(details.get('total_trades', 0))
                evals.append({'n': k, 'profit_pct': profit_pct, 'profit_abs': profit_abs, 'trades': trades})

            # 結果概要表示
            if evals:
                best = max(evals, key=lambda x: x['profit_pct'])
                print("\n=== N最適化結果 (基準: Total profit %) ===")
                for e in evals:
                    print(f"N={e['n']:>2}  利益率={e['profit_pct']:>8.2f}%  利益={e['profit_abs']:>10.2f} USDT  取引={e['trades']}")
                print(f"\n推奨N: {best['n']}  (利益率 {best['profit_pct']:.2f}%, 利益 {best['profit_abs']:.2f} USDT, 取引 {best['trades']})")
            return evals
        finally:
            # configを復元
            try:
                with open(self.config_path, 'w') as f:
                    json.dump(cfg_backup, f, indent=2)
            except Exception:
                pass

        top_pairs = [r['pair'] for r in self.results[: (self.top_n or len(self.results))]]
        if verify_exchange:
            active = self._list_active_usdt_pairs(exchange)
            if active:
                top_pairs = [p for p in top_pairs if p in active]

        # config を更新
        try:
            with open(self.config_path, 'r') as f:
                cfg = json.load(f)
            cfg.setdefault('exchange', {})['pair_whitelist'] = top_pairs
            with open(self.config_path, 'w') as f:
                json.dump(cfg, f, indent=2)
            print("🛠  config を更新しました (pair_whitelist):")
            for p in top_pairs:
                print(f"   - {p}")
            return top_pairs
        except Exception as e:
            print(f"❌ config 更新に失敗: {e}")
            return []

    def display_results(self):
        """結果を表示"""
        print("\n" + "=" * 60)
        print("🏆 通貨ペア別利益率ランキング")
        print("=" * 60)

        print(f"{'順位':<4} {'通貨ペア':<12} {'利益率':<8} {'利益(USDT)':<12} {'取引数':<6}")
        print("-" * 60)

        to_show = self.results
        if self.top_n:
            to_show = to_show[: self.top_n]

        for i, result in enumerate(to_show, 1):
            print(f"{i:<4} {result['pair']:<12} {result['profit_ratio']:<8.2f}% {result['profit']:<12.2f} {result['trades']:<6}")

        print("\n" + "=" * 60)
        print("📊 統計情報")
        print("=" * 60)

        profitable_pairs = [r for r in self.results if r['profit_ratio'] > 0]
        if profitable_pairs:
            avg_profit_ratio = sum(r['profit_ratio'] for r in profitable_pairs) / len(profitable_pairs)
            total_profit = sum(r['profit'] for r in profitable_pairs)
            total_trades = sum(r['trades'] for r in profitable_pairs)

            print(f"利益を上げた通貨ペア数: {len(profitable_pairs)}/{len(self.results)}")
            print(f"平均利益率: {avg_profit_ratio:.2f}%")
            print(f"総利益: {total_profit:.2f} USDT")
            print(f"総取引数: {total_trades}")
            print(f"最良通貨ペア: {profitable_pairs[0]['pair']} ({profitable_pairs[0]['profit_ratio']:.2f}%)")
            print(f"最悪通貨ペア: {profitable_pairs[-1]['pair']} ({profitable_pairs[-1]['profit_ratio']:.2f}%)")

    def save_results(self):
        """結果をファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pair_analysis_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'results': self.results,
                'summary': {
                    'total_pairs': len(self.results),
                    'profitable_pairs': len([r for r in self.results if r['profit_ratio'] > 0]),
                    'total_profit': sum(r['profit'] for r in self.results),
                    'total_trades': sum(r['trades'] for r in self.results)
                }
            }, f, indent=2, ensure_ascii=False)

        print(f"\n💾 結果を {filename} に保存しました")

    # 既存のバックテスト結果(zip)からランキングを作成
    def _analyze_from_existing_results(self) -> bool:
        """user_data/backtest_results にある既存結果から 5m ichiV1 の最新/対象期間を読み取りランキングを作成。
        成功した場合 True を返す。
        """
        try:
            results_dir = os.path.join(self.freqtrade_dir, 'user_data', 'backtest_results')
            zips = sorted(glob(os.path.join(results_dir, 'backtest-result-*.zip')))
            if not zips:
                return False

            # ターゲット期間を UNIX 秒に変換
            try:
                start_str, end_str = self.timerange.split('-')
                start_dt = datetime.strptime(start_str, '%Y%m%d').replace(tzinfo=timezone.utc)
                end_dt = datetime.strptime(end_str, '%Y%m%d').replace(tzinfo=timezone.utc)
                TARGET_START = int(start_dt.timestamp())
                TARGET_END = int(end_dt.timestamp())
            except Exception:
                TARGET_START = TARGET_END = None

            candidates = []
            for zf in zips:
                meta_path = zf.replace('.zip', '.meta.json')
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    strat = next(iter(meta))
                    m = meta[strat]
                except Exception:
                    continue
                if m.get('timeframe') != '5m':
                    continue
                # まずは対象期間に一致するものを優先、無ければ5mの最新を使う
                score = 0
                if TARGET_START and TARGET_END and m.get('backtest_start_ts') == TARGET_START and m.get('backtest_end_ts') == TARGET_END:
                    score = 2
                else:
                    score = 1

                # zip 内の JSON を調べて、通貨ペア件数を測る
                try:
                    with zipfile.ZipFile(zf) as z:
                        jnames = [n for n in z.namelist() if n.endswith('.json') and 'config' not in n]
                        if not jnames:
                            continue
                        with z.open(jnames[0]) as f:
                            data = json.load(f)
                    sdict = data.get('strategy', {})
                    sname = next(iter(sdict)) if isinstance(sdict, dict) else 'ichiV1'
                    details = sdict[sname] if isinstance(sdict, dict) else sdict
                    rpp = details.get('results_per_pair') or []
                    # list 形式を想定（TOTALを除く）
                    pair_count = sum(1 for r in rpp if isinstance(r, dict) and r.get('key') not in (None, 'TOTAL'))
                    candidates.append((score, pair_count, zf))
                except Exception:
                    continue

            if not candidates:
                return False

            # スコア(期間一致) → ペア数 → ファイル名 で最大のものを採用
            candidates.sort(key=lambda x: (x[0], x[1], x[2]))
            score, _, best_zip = candidates[-1]

            # 選んだ zip から結果を抽出
            with zipfile.ZipFile(best_zip) as z:
                jnames = [n for n in z.namelist() if n.endswith('.json') and 'config' not in n]
                with z.open(jnames[0]) as f:
                    data = json.load(f)
            sdict = data.get('strategy', {})
            sname = next(iter(sdict)) if isinstance(sdict, dict) else 'ichiV1'
            details = sdict[sname] if isinstance(sdict, dict) else sdict
            rpp = details.get('results_per_pair') or []

            tmp_results = []
            for r in rpp:
                if not isinstance(r, dict):
                    continue
                pair = r.get('key')
                if not pair or pair == 'TOTAL':
                    continue
                profit_ratio = r.get('profit_total_pct') or 0.0
                profit = r.get('profit_total_abs') or 0.0
                trades = r.get('trades') or 0
                tmp_results.append({
                    'pair': pair.replace('_', '/'),
                    'trades': int(trades),
                    'profit': float(profit),
                    'profit_ratio': float(profit_ratio)
                })

            if not tmp_results:
                return False

            self.results = tmp_results
            print("📦 既存のバックテスト結果からランキングを作成しました:")
            print(f"    -> {os.path.basename(best_zip)} を使用")
            return True

        except Exception:
            return False

    def _run_single_backtest_all_pairs(self, timerange: str) -> bool:
        """1回のバックテストでペア一覧をまとめて評価し、zip結果を解析してself.resultsへ格納。
        成功時 True を返す。
        """
        print("🧪 全ペア一括バックテストを実行します…")
        # 設定をバックアップしてホワイトリストを差し替え
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            original_whitelist = config.get('exchange', {}).get('pair_whitelist', [])
            config['exchange']['pair_whitelist'] = self.pairs
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"⚠️ 設定の読み書きに失敗: {e}")
            return False

        # 実行前の既存zip一覧（新規作成を識別するため）
        results_dir = os.path.join(self.freqtrade_dir, 'user_data', 'backtest_results')
        pre_existing = set(glob(os.path.join(results_dir, 'backtest-result-*.zip')))

        try:
            cmd = [
                "freqtrade", "backtesting",
                "--config", self.config_path,
                "--strategy", "ichiV1",
                "--timerange", timerange,
                "--export", "trades"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.freqtrade_dir)
            if result.returncode != 0:
                print("❌ 一括バックテストに失敗しました")
                if result.stderr:
                    print(result.stderr)
                if result.stdout:
                    print(result.stdout)
                return False

            # 新規に作成されたzipを特定
            post_existing = set(glob(os.path.join(results_dir, 'backtest-result-*.zip')))
            new_zips = sorted(list(post_existing - pre_existing))
            target_zip = new_zips[-1] if new_zips else None
            if not target_zip:
                # 最後の手段: 最終更新時刻で最新を取得
                all_zips = glob(os.path.join(results_dir, 'backtest-result-*.zip'))
                if not all_zips:
                    print("⚠️ バックテスト結果(zip)が見つかりません")
                    return False
                target_zip = max(all_zips, key=os.path.getmtime)

            # zip を解析して results_per_pair からランキングを構築
            with zipfile.ZipFile(target_zip) as z:
                jnames = [n for n in z.namelist() if n.endswith('.json') and 'config' not in n]
                if not jnames:
                    print("⚠️ zip内に結果JSONが見当たりません")
                    return False
                with z.open(jnames[0]) as f:
                    data = json.load(f)
            sdict = data.get('strategy', {})
            sname = next(iter(sdict)) if isinstance(sdict, dict) else 'ichiV1'
            details = sdict[sname] if isinstance(sdict, dict) else sdict
            rpp = details.get('results_per_pair') or []

            tmp_results = []
            for r in rpp:
                if not isinstance(r, dict):
                    continue
                pair = r.get('key')
                if not pair or pair == 'TOTAL':
                    continue
                profit_ratio = r.get('profit_total_pct') or 0.0
                profit = r.get('profit_total_abs') or 0.0
                trades = r.get('trades') or 0
                tmp_results.append({
                    'pair': pair.replace('_', '/'),
                    'trades': int(trades),
                    'profit': float(profit),
                    'profit_ratio': float(profit_ratio)
                })

            if not tmp_results:
                print("⚠️ 結果が空でした")
                return False

            self.results = tmp_results
            print("✅ 一括バックテストの結果を解析しました")
            return True

        finally:
            # 設定を復元
            try:
                with open(self.config_path, 'r') as f:
                    cfg = json.load(f)
                cfg['exchange']['pair_whitelist'] = original_whitelist
                with open(self.config_path, 'w') as f:
                    json.dump(cfg, f, indent=2)
            except Exception:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ichiV1 5m バックテスト・ペア別ランキング")
    parser.add_argument("--timerange", default=os.getenv("TIMERANGE", "20231030-20240427"), help="期間 (例: 20250101-20250831)")
    parser.add_argument("--top", type=int, default=int(os.getenv("TOP", "0")), help="上位N件のみ表示 (0で全件)")
    parser.add_argument("--use-existing-results", dest="use_existing_results", action="store_true", help="既存結果を優先利用")
    parser.add_argument("--no-use-existing-results", dest="use_existing_results", action="store_false", help="既存結果を使わず再計算")
    parser.add_argument("--update-config", action="store_true", help="上位Nペアでconfigのpair_whitelistを更新")
    parser.add_argument("--no-verify-exchange", dest="verify_exchange", action="store_false", help="取引可否の確認をスキップ")
    parser.add_argument("--verify-exchange", dest="verify_exchange", action="store_true", help="Binanceの取引可否を確認して反映")
    parser.set_defaults(verify_exchange=True)
    parser.add_argument("--optimize-topn", action="store_true", help="1..Nで一括バックテストし、利益率最大のNを探索")
    parser.set_defaults(use_existing_results=os.getenv("USE_EXISTING_RESULTS", "1") == "1")

    args = parser.parse_args()
    analyzer = PairAnalyzer(timerange=args.timerange, top_n=args.top, use_existing_results=args.use_existing_results)
    analyzer.analyze_all_pairs()
    if args.update_config:
        analyzer.update_config_with_top_pairs(verify_exchange=args.verify_exchange)
    if args.optimize_topn:
        analyzer.optimize_topn(timerange=args.timerange, max_n=args.top or 0)
