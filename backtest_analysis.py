#!/usr/bin/env python3
"""
バックテスト分析スクリプト
過去6ヶ月間を様々なパターンでバックテストを実行し、結果を集計分析する
"""

import subprocess
import json
import pandas as pd
from datetime import datetime, timedelta
import os
import re

class BacktestAnalyzer:
    def __init__(self, config_path="user_data/config.json", strategy="ichiV1"):
        self.config_path = config_path
        self.strategy = strategy
        self.results = []

    def run_backtest(self, timerange, description):
        """バックテストを実行"""
        print(f"\n🔍 実行中: {description}")
        print(f"📅 期間: {timerange}")

        cmd = [
            "freqtrade", "backtesting",
            "--config", self.config_path,
            "--strategy", self.strategy,
            "--timerange", timerange,
            "--export", "trades"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/Users/takaakikamio/Workspace/_FreqtradeProjects/freqtrade")

            if result.returncode == 0:
                # 結果を解析
                summary = self.parse_backtest_output(result.stdout)
                summary['description'] = description
                summary['timerange'] = timerange
                self.results.append(summary)

                print(f"✅ 完了: {description}")
                print(f"📊 取引数: {summary.get('total_trades', 0)}")
                print(f"💰 利益: {summary.get('total_profit', 0):.2f} USDT")
                print(f"📈 利益率: {summary.get('total_profit_percent', 0):.2f}%")

            else:
                print(f"❌ エラー: {description}")
                print(f"エラー内容: {result.stderr}")

        except Exception as e:
            print(f"❌ 例外: {description} - {str(e)}")

    def parse_backtest_output(self, output):
        """バックテスト出力を解析"""
        summary = {}

        # 総取引数
        trades_match = re.search(r'Total/Daily Avg Trades\s+\│\s+(\d+) /', output)
        if trades_match:
            summary['total_trades'] = int(trades_match.group(1))

        # 総利益
        profit_match = re.search(r'Absolute profit\s+\│\s+([-\d.]+) USDT', output)
        if profit_match:
            summary['total_profit'] = float(profit_match.group(1))

        # 利益率
        profit_percent_match = re.search(r'Total profit %\s+\│\s+([-\d.]+)%', output)
        if profit_percent_match:
            summary['total_profit_percent'] = float(profit_percent_match.group(1))

        # 勝率
        winrate_match = re.search(r'Win  Draw  Loss  Win%\s+\│\s+\d+\s+\d+\s+\d+\s+(\d+\.?\d*)', output)
        if winrate_match:
            summary['winrate'] = float(winrate_match.group(1))

        # 平均取引時間
        duration_match = re.search(r'Avg Duration Winners\s+\│\s+(\d+:\d+:\d+)', output)
        if duration_match:
            summary['avg_duration'] = duration_match.group(1)

        # 最大ドローダウン
        drawdown_match = re.search(r'Absolute Drawdown \(Account\)\s+\│\s+([\d.]+)%', output)
        if drawdown_match:
            summary['max_drawdown'] = float(drawdown_match.group(1))

        # Sharpe ratio
        sharpe_match = re.search(r'Sharpe\s+\│\s+([-\d.]+)', output)
        if sharpe_match:
            summary['sharpe'] = float(sharpe_match.group(1))

        return summary

    def run_all_patterns(self):
        """全てのパターンでバックテストを実行"""
        print("🚀 バックテスト分析を開始します...")

        # 直近6ヶ月間の期間を定義（利用可能なデータに合わせて調整）
        end_date = datetime(2024, 4, 27)  # 利用可能なデータの最終日
        start_date = end_date - timedelta(days=180)  # 6ヶ月前

        # パターン1: 1ヶ月刻み（6回）
        print("\n📊 パターン1: 1ヶ月刻みでバックテスト")
        for i in range(6):
            period_start = start_date + timedelta(days=i*30)
            period_end = period_start + timedelta(days=30)

            timerange = f"{period_start.strftime('%Y%m%d')}-{period_end.strftime('%Y%m%d')}"
            description = f"1ヶ月刻み {i+1}/6 ({period_start.strftime('%Y-%m')})"

            self.run_backtest(timerange, description)

        # パターン2: 2ヶ月刻み（3回）
        print("\n📊 パターン2: 2ヶ月刻みでバックテスト")
        for i in range(3):
            period_start = start_date + timedelta(days=i*60)
            period_end = period_start + timedelta(days=60)

            timerange = f"{period_start.strftime('%Y%m%d')}-{period_end.strftime('%Y%m%d')}"
            description = f"2ヶ月刻み {i+1}/3 ({period_start.strftime('%Y-%m')} to {(period_start + timedelta(days=59)).strftime('%Y-%m')})"

            self.run_backtest(timerange, description)

        # パターン3: 3ヶ月刻み（2回）
        print("\n📊 パターン3: 3ヶ月刻みでバックテスト")
        for i in range(2):
            period_start = start_date + timedelta(days=i*90)
            period_end = period_start + timedelta(days=90)

            timerange = f"{period_start.strftime('%Y%m%d')}-{period_end.strftime('%Y%m%d')}"
            description = f"3ヶ月刻み {i+1}/2 ({period_start.strftime('%Y-%m')} to {(period_start + timedelta(days=89)).strftime('%Y-%m')})"

            self.run_backtest(timerange, description)

        # パターン4: 全期間（1回）
        print("\n📊 パターン4: 全期間でバックテスト")
        timerange = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
        description = f"全期間 ({start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')})"

        self.run_backtest(timerange, description)

    def generate_report(self):
        """分析レポートを生成"""
        if not self.results:
            print("❌ 結果がありません")
            return

        print("\n" + "="*80)
        print("📊 バックテスト分析レポート")
        print("="*80)

        # DataFrameに変換
        df = pd.DataFrame(self.results)

        # 基本統計
        print("\n📈 基本統計")
        print("-" * 50)
        print(f"総テスト数: {len(df)}")
        if 'total_trades' in df.columns:
            print(f"平均取引数: {df['total_trades'].mean():.1f}")
        if 'total_profit' in df.columns:
            print(f"平均利益: {df['total_profit'].mean():.2f} USDT")
        if 'total_profit_percent' in df.columns:
            print(f"平均利益率: {df['total_profit_percent'].mean():.2f}%")
        if 'winrate' in df.columns:
            print(f"平均勝率: {df['winrate'].mean():.1f}%")
        if 'max_drawdown' in df.columns:
            print(f"平均ドローダウン: {df['max_drawdown'].mean():.2f}%")

        # パターン別分析
        print("\n📊 パターン別分析")
        print("-" * 50)

        patterns = {
            '1ヶ月刻み': df[df['description'].str.contains('1ヶ月刻み')],
            '2ヶ月刻み': df[df['description'].str.contains('2ヶ月刻み')],
            '3ヶ月刻み': df[df['description'].str.contains('3ヶ月刻み')],
            '全期間': df[df['description'].str.contains('全期間')]
        }

        for pattern_name, pattern_df in patterns.items():
            if len(pattern_df) > 0:
                print(f"\n{pattern_name}:")
                print(f"  テスト数: {len(pattern_df)}")
                if 'total_profit_percent' in pattern_df.columns:
                    print(f"  平均利益率: {pattern_df['total_profit_percent'].mean():.2f}%")
                if 'total_trades' in pattern_df.columns:
                    print(f"  平均取引数: {pattern_df['total_trades'].mean():.1f}")
                if 'winrate' in pattern_df.columns:
                    print(f"  平均勝率: {pattern_df['winrate'].mean():.1f}%")

        # 最良・最悪の結果
        print("\n🏆 最良・最悪の結果")
        print("-" * 50)

        if 'total_profit' in df.columns and len(df[df['total_profit'] != 0]) > 0:
            best_profit = df.loc[df['total_profit'].idxmax()]
            worst_profit = df.loc[df['total_profit'].idxmin()]

            print(f"最良利益: {best_profit['description']}")
            if 'total_profit' in best_profit:
                print(f"  利益: {best_profit['total_profit']:.2f} USDT ({best_profit['total_profit_percent']:.2f}%)")
            if 'total_trades' in best_profit:
                print(f"  取引数: {best_profit['total_trades']}")
            if 'winrate' in best_profit:
                print(f"  勝率: {best_profit['winrate']:.1f}%")

            print(f"\n最悪利益: {worst_profit['description']}")
            if 'total_profit' in worst_profit:
                print(f"  利益: {worst_profit['total_profit']:.2f} USDT ({worst_profit['total_profit_percent']:.2f}%)")
            if 'total_trades' in worst_profit:
                print(f"  取引数: {worst_profit['total_trades']}")
            if 'winrate' in worst_profit:
                print(f"  勝率: {worst_profit['winrate']:.1f}%")
        else:
            print("⚠️  全ての期間で取引が0件でした")
            print("   これは以下の理由が考えられます:")
            print("   - ストラテジの条件が非常に厳格")
            print("   - 取引ペアの制限が影響")
            print("   - 市場条件が条件を満たさない")

        # 詳細結果テーブル
        print("\n📋 詳細結果")
        print("-" * 80)
        print(f"{'説明':<30} {'期間':<20} {'取引数':<8} {'利益(USDT)':<12} {'利益率(%)':<10} {'勝率(%)':<8}")
        print("-" * 80)

        for result in self.results:
            print(f"{result['description']:<30} {result['timerange']:<20} {result.get('total_trades', 0):<8} "
                  f"{result.get('total_profit', 0):<12.2f} {result.get('total_profit_percent', 0):<10.2f} "
                  f"{result.get('winrate', 0):<8.1f}")

        # 結果をJSONファイルに保存
        output_file = f"backtest_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"\n💾 結果を {output_file} に保存しました")

        return df

def main():
    analyzer = BacktestAnalyzer()
    analyzer.run_all_patterns()
    analyzer.generate_report()

if __name__ == "__main__":
    main()
