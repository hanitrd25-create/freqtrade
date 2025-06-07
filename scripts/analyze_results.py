#!/usr/bin/env python3
"""
Analyze Freqtrade backtest results and generate meaningful metrics
"""
import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime


class BacktestAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.results = []
        
    def load_results(self):
        """Load all backtest result files"""
        for json_file in self.results_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                self.results.append({
                    'filename': json_file.name,
                    'data': data
                })
    
    def calculate_metrics(self, result):
        """Calculate key performance metrics"""
        strategy_name = result['filename'].split('_')[0]
        backtest_result = result['data']['backtest_result']
        
        if not backtest_result:
            return None
            
        stats = backtest_result[0]  # First result
        
        # Calculate additional metrics
        trades = pd.DataFrame(stats.get('trades', []))
        
        if len(trades) == 0:
            return None
            
        # Win rate
        winning_trades = trades[trades['profit_abs'] > 0]
        win_rate = len(winning_trades) / len(trades) * 100
        
        # Average trade duration
        trades['duration'] = pd.to_datetime(trades['close_date']) - pd.to_datetime(trades['open_date'])
        avg_duration = trades['duration'].mean()
        
        # Risk metrics
        returns = trades['profit_ratio'].values
        sharpe = self.calculate_sharpe(returns) if len(returns) > 1 else 0
        
        return {
            'strategy': strategy_name,
            'timerange': result['filename'].split('_')[1].replace('.json', ''),
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(trades) - len(winning_trades),
            'win_rate': round(win_rate, 2),
            'profit_mean': round(stats['profit_mean'] * 100, 2),
            'profit_total': round(stats['profit_total'] * 100, 2),
            'profit_total_abs': round(stats['profit_total_abs'], 2),
            'max_drawdown': round(stats['max_drawdown'] * 100, 2),
            'max_drawdown_abs': round(stats['max_drawdown_abs'], 2),
            'sharpe_ratio': round(sharpe, 2),
            'avg_duration': str(avg_duration).split('.')[0],
            'best_trade': round(trades['profit_ratio'].max() * 100, 2),
            'worst_trade': round(trades['profit_ratio'].min() * 100, 2),
            'calmar_ratio': round(abs(stats['profit_total'] / stats['max_drawdown']), 2) if stats['max_drawdown'] != 0 else 0,
            'expectancy': round(stats['expectancy'], 2) if 'expectancy' in stats else 0,
            'profit_factor': round(stats['profit_factor'], 2) if 'profit_factor' in stats else 0,
        }
    
    def calculate_sharpe(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        import numpy as np
        excess_returns = returns - risk_free_rate / 365  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365) if np.std(excess_returns) > 0 else 0
    
    def generate_markdown_report(self, metrics_list):
        """Generate markdown report"""
        report = []
        report.append("# Backtest Results Summary\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Check if meets investor requirements
        for metrics in metrics_list:
            if metrics:
                meets_requirements = (
                    metrics['profit_total'] >= 30 and 
                    metrics['max_drawdown'] <= 15
                )
                
                report.append(f"## Strategy: {metrics['strategy']} ({metrics['timerange']})\n")
                
                if meets_requirements:
                    report.append("✅ **MEETS INVESTOR REQUIREMENTS** (30% APY, 10-15% drawdown)\n")
                else:
                    report.append("❌ **DOES NOT MEET REQUIREMENTS**\n")
                
                report.append("### Performance Metrics\n")
                report.append(f"- **Total Return**: {metrics['profit_total']}%")
                report.append(f"- **Max Drawdown**: {metrics['max_drawdown']}%")
                report.append(f"- **Sharpe Ratio**: {metrics['sharpe_ratio']}")
                report.append(f"- **Calmar Ratio**: {metrics['calmar_ratio']}")
                report.append(f"- **Profit Factor**: {metrics['profit_factor']}\n")
                
                report.append("### Trade Statistics\n")
                report.append(f"- **Total Trades**: {metrics['total_trades']}")
                report.append(f"- **Win Rate**: {metrics['win_rate']}%")
                report.append(f"- **Average Trade**: {metrics['profit_mean']}%")
                report.append(f"- **Best Trade**: {metrics['best_trade']}%")
                report.append(f"- **Worst Trade**: {metrics['worst_trade']}%")
                report.append(f"- **Avg Duration**: {metrics['avg_duration']}\n")
                
                report.append("---\n")
        
        return '\n'.join(report)
    
    def run(self, output_format='markdown'):
        """Run the analysis"""
        self.load_results()
        
        metrics_list = []
        for result in self.results:
            metrics = self.calculate_metrics(result)
            if metrics:
                metrics_list.append(metrics)
        
        if output_format == 'markdown':
            return self.generate_markdown_report(metrics_list)
        else:
            return json.dumps(metrics_list, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Analyze Freqtrade backtest results')
    parser.add_argument('--results-dir', required=True, help='Directory containing backtest results')
    parser.add_argument('--output-format', choices=['markdown', 'json'], default='markdown')
    
    args = parser.parse_args()
    
    analyzer = BacktestAnalyzer(args.results_dir)
    report = analyzer.run(args.output_format)
    print(report)


if __name__ == "__main__":
    main()
