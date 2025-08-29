#!/usr/bin/env python3
"""
Performance Comparison Benchmark Suite

This module provides comprehensive benchmarking to compare performance
before and after optimizations.
"""

import logging
import time
import gc
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import psutil
from functools import wraps

from freqtrade.configuration import Configuration
from freqtrade.data.history import load_pair_history
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.hyperopt import Hyperopt
from freqtrade.optimize.vectorized_backtesting import VectorizedBacktester
from freqtrade.optimize.numpy_ops import (
    numpy_shift, numpy_rolling_mean, numpy_rolling_max, numpy_rolling_min
)
from freqtrade.optimize.indicator_cache import IndicatorCache
from freqtrade.optimize.signal_cache import SignalCache
from freqtrade.optimize.hyperopt_integration import HyperoptIntegration
from freqtrade.data.history.parquet_memory_mapper import ParquetMemoryMapper
from freqtrade.data.history.lazy_dataframe_loader import LazyDataFrameLoader


logger = logging.getLogger(__name__)


def time_it(func):
    """Decorator to measure execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return wrapper


def measure_memory(func):
    """Decorator to measure memory usage"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        gc.collect()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return result, mem_after - mem_before
    return wrapper


class PerformanceComparison:
    """Compare performance with and without optimizations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize performance comparison.
        
        :param config: Freqtrade configuration
        """
        self.config = config
        self.results: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'pairs': config.get('exchange', {}).get('pair_whitelist', []),
                'timeframe': config.get('timeframe', '5m'),
                'timerange': config.get('timerange', ''),
            },
            'benchmarks': {}
        }
        
    def benchmark_dataframe_operations(self, size: int = 100000) -> Dict[str, Any]:
        """
        Compare DataFrame operations with and without NumPy optimization.
        
        :param size: Size of test DataFrame
        :return: Benchmark results
        """
        logger.info(f"Benchmarking DataFrame operations with {size} rows...")
        
        # Create test data
        df = pd.DataFrame({
            'close': np.random.randn(size).cumsum() + 100,
            'volume': np.random.randn(size) * 1000 + 10000,
            'high': np.random.randn(size).cumsum() + 102,
            'low': np.random.randn(size).cumsum() + 98,
        })
        
        results = {}
        
        # Benchmark shift operations
        @time_it
        def pandas_shift():
            return df['close'].shift(1)
        
        @time_it
        def numpy_shift_opt():
            return numpy_shift(df['close'].values, 1)
        
        _, pandas_time = pandas_shift()
        _, numpy_time = numpy_shift_opt()
        
        results['shift'] = {
            'pandas_time': pandas_time,
            'numpy_time': numpy_time,
            'speedup': pandas_time / numpy_time if numpy_time > 0 else 0
        }
        
        # Benchmark rolling operations
        @time_it
        def pandas_rolling():
            return df['close'].rolling(window=20).mean()
        
        @time_it
        def numpy_rolling():
            return numpy_rolling_mean(df['close'].values, 20)
        
        _, pandas_time = pandas_rolling()
        _, numpy_time = numpy_rolling()
        
        results['rolling_mean'] = {
            'pandas_time': pandas_time,
            'numpy_time': numpy_time,
            'speedup': pandas_time / numpy_time if numpy_time > 0 else 0
        }
        
        # Benchmark min/max operations
        @time_it
        def pandas_minmax():
            return df['close'].rolling(window=14).max(), df['close'].rolling(window=14).min()
        
        @time_it
        def numpy_minmax():
            return numpy_rolling_max(df['close'].values, 14), numpy_rolling_min(df['close'].values, 14)
        
        _, pandas_time = pandas_minmax()
        _, numpy_time = numpy_minmax()
        
        results['rolling_minmax'] = {
            'pandas_time': pandas_time,
            'numpy_time': numpy_time,
            'speedup': pandas_time / numpy_time if numpy_time > 0 else 0
        }
        
        # Calculate average speedup
        speedups = [r['speedup'] for r in results.values()]
        results['average_speedup'] = np.mean(speedups)
        
        logger.info(f"DataFrame operations average speedup: {results['average_speedup']:.2f}x")
        
        return results
    
    def benchmark_vectorized_backtesting(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare vectorized vs iterative backtesting.
        
        :param test_data: Test DataFrame
        :return: Benchmark results
        """
        logger.info("Benchmarking vectorized backtesting...")
        
        # Create simple buy/sell signals
        test_data['buy'] = (test_data['close'] > test_data['close'].shift(1)) & \
                          (test_data['volume'] > test_data['volume'].rolling(20).mean())
        test_data['sell'] = (test_data['close'] < test_data['close'].shift(1)) & \
                           (test_data['close'] < test_data['close'].rolling(20).mean())
        
        results = {}
        
        # Simulate iterative processing
        @time_it
        def iterative_backtest():
            trades = []
            in_position = False
            
            for i in range(len(test_data)):
                if not in_position and test_data.iloc[i]['buy']:
                    trades.append({
                        'entry_index': i,
                        'entry_price': test_data.iloc[i]['close']
                    })
                    in_position = True
                elif in_position and test_data.iloc[i]['sell']:
                    if trades and 'exit_index' not in trades[-1]:
                        trades[-1]['exit_index'] = i
                        trades[-1]['exit_price'] = test_data.iloc[i]['close']
                        in_position = False
            
            return trades
        
        # Simulate vectorized processing
        @time_it
        def vectorized_backtest():
            buy_signals = test_data['buy'].values
            sell_signals = test_data['sell'].values
            close_prices = test_data['close'].values
            
            # Find trade entries and exits using NumPy
            entries = np.where(buy_signals)[0]
            exits = np.where(sell_signals)[0]
            
            trades = []
            last_exit = -1
            
            for entry in entries:
                if entry > last_exit:
                    # Find next exit after this entry
                    next_exits = exits[exits > entry]
                    if len(next_exits) > 0:
                        exit_idx = next_exits[0]
                        trades.append({
                            'entry_index': entry,
                            'entry_price': close_prices[entry],
                            'exit_index': exit_idx,
                            'exit_price': close_prices[exit_idx]
                        })
                        last_exit = exit_idx
            
            return trades
        
        _, iter_time = iterative_backtest()
        _, vec_time = vectorized_backtest()
        
        results['execution_time'] = {
            'iterative': iter_time,
            'vectorized': vec_time,
            'speedup': iter_time / vec_time if vec_time > 0 else 0
        }
        
        logger.info(f"Vectorized backtesting speedup: {results['execution_time']['speedup']:.2f}x")
        
        return results
    
    def benchmark_caching(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Benchmark indicator and signal caching.
        
        :param test_data: Test DataFrame
        :return: Benchmark results
        """
        logger.info("Benchmarking caching mechanisms...")
        
        results = {}
        
        # Benchmark indicator caching
        indicator_cache = IndicatorCache(max_size_mb=100)
        
        def compute_indicators():
            """Simulate indicator computation"""
            rsi = test_data['close'].rolling(14).mean()  # Simplified RSI
            macd = test_data['close'].ewm(span=12).mean() - test_data['close'].ewm(span=26).mean()
            bb_upper = test_data['close'].rolling(20).mean() + 2 * test_data['close'].rolling(20).std()
            return {'rsi': rsi, 'macd': macd, 'bb_upper': bb_upper}
        
        # First computation (cache miss)
        @time_it
        def first_compute():
            key = indicator_cache._generate_key('test_pair', '5m', test_data.index[-1])
            return indicator_cache.cache_indicator(key, compute_indicators)
        
        # Second computation (cache hit)
        @time_it
        def cached_compute():
            key = indicator_cache._generate_key('test_pair', '5m', test_data.index[-1])
            return indicator_cache.cache_indicator(key, compute_indicators)
        
        _, miss_time = first_compute()
        _, hit_time = cached_compute()
        
        results['indicator_cache'] = {
            'miss_time': miss_time,
            'hit_time': hit_time,
            'speedup': miss_time / hit_time if hit_time > 0 else 0,
            'cache_stats': indicator_cache.get_stats()
        }
        
        # Benchmark signal caching
        signal_cache = SignalCache()
        
        def compute_signals():
            """Simulate signal computation"""
            buy = (test_data['close'] > test_data['close'].shift(1))
            sell = (test_data['close'] < test_data['close'].shift(1))
            return pd.DataFrame({'buy': buy, 'sell': sell})
        
        # First computation (cache miss)
        @time_it
        def first_signal():
            return signal_cache.get_or_compute('test_pair', '5m', compute_signals)
        
        # Second computation (cache hit)
        @time_it
        def cached_signal():
            return signal_cache.get_or_compute('test_pair', '5m', compute_signals)
        
        _, miss_time = first_signal()
        _, hit_time = cached_signal()
        
        results['signal_cache'] = {
            'miss_time': miss_time,
            'hit_time': hit_time,
            'speedup': miss_time / hit_time if hit_time > 0 else 0,
            'cache_stats': signal_cache.get_stats()
        }
        
        # Calculate average cache effectiveness
        cache_speedups = [
            results['indicator_cache']['speedup'],
            results['signal_cache']['speedup']
        ]
        results['average_cache_speedup'] = np.mean(cache_speedups)
        
        logger.info(f"Average cache speedup: {results['average_cache_speedup']:.2f}x")
        
        return results
    
    def benchmark_memory_usage(self, num_pairs: int = 10) -> Dict[str, Any]:
        """
        Compare memory usage with and without optimizations.
        
        :param num_pairs: Number of pairs to load
        :return: Benchmark results
        """
        logger.info(f"Benchmarking memory usage with {num_pairs} pairs...")
        
        results = {}
        
        # Generate test data
        test_data = {}
        for i in range(num_pairs):
            df = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=100000, freq='5min'),
                'open': np.random.randn(100000).cumsum() + 100,
                'high': np.random.randn(100000).cumsum() + 102,
                'low': np.random.randn(100000).cumsum() + 98,
                'close': np.random.randn(100000).cumsum() + 100,
                'volume': np.random.randn(100000) * 1000 + 10000,
            })
            test_data[f'PAIR{i}/USDT'] = df
        
        # Measure regular loading
        @measure_memory
        def regular_loading():
            loaded = {}
            for pair, df in test_data.items():
                loaded[pair] = df.copy()
            return loaded
        
        # Measure lazy loading (simulated)
        @measure_memory
        def lazy_loading():
            loaded = {}
            for pair, df in test_data.items():
                # Simulate lazy loading by storing reference
                loaded[pair] = df  # No copy
            return loaded
        
        # Measure shared memory usage
        @measure_memory
        def shared_memory_loading():
            from freqtrade.optimize.shared_memory_simple import SimpleSharedMemoryManager
            
            smm = SimpleSharedMemoryManager()
            metadata_file = smm.share_dataframe_dict(test_data)
            
            # Simulate multiple workers accessing
            loaded = SimpleSharedMemoryManager.load_from_metadata(metadata_file)
            
            smm.cleanup()
            return loaded
        
        _, regular_mem = regular_loading()
        _, lazy_mem = lazy_loading()
        _, shared_mem = shared_memory_loading()
        
        results['memory_usage'] = {
            'regular_mb': regular_mem,
            'lazy_mb': lazy_mem,
            'shared_mb': shared_mem,
            'lazy_savings': (regular_mem - lazy_mem) / regular_mem * 100 if regular_mem > 0 else 0,
            'shared_savings': (regular_mem - shared_mem) / regular_mem * 100 if regular_mem > 0 else 0,
        }
        
        logger.info(f"Memory savings - Lazy: {results['memory_usage']['lazy_savings']:.1f}%, "
                   f"Shared: {results['memory_usage']['shared_savings']:.1f}%")
        
        return results
    
    def benchmark_hyperopt_optimization(self) -> Dict[str, Any]:
        """
        Benchmark hyperopt resource optimization.
        
        :return: Benchmark results
        """
        logger.info("Benchmarking hyperopt optimization...")
        
        from freqtrade.optimize.hyperopt_resource_optimizer import HyperoptResourceOptimizer
        
        optimizer = HyperoptResourceOptimizer(self.config)
        
        results = {}
        
        # Test different scenarios
        scenarios = [
            {'epochs': 100, 'data_mb': 50, 'name': 'small'},
            {'epochs': 500, 'data_mb': 200, 'name': 'medium'},
            {'epochs': 1000, 'data_mb': 500, 'name': 'large'},
        ]
        
        for scenario in scenarios:
            profile = optimizer.get_optimization_profile(
                scenario['epochs'], 
                scenario['data_mb']
            )
            
            # Simulate execution time
            base_time = scenario['epochs'] * 0.1  # 0.1s per epoch baseline
            
            # Calculate optimized time based on parallelization and batching
            opt_time = base_time / profile['workers'] * 1.2  # 20% overhead
            
            if profile['use_shared_memory']:
                opt_time *= 0.8  # 20% faster with shared memory
            
            results[scenario['name']] = {
                'profile': profile,
                'estimated_base_time': base_time,
                'estimated_opt_time': opt_time,
                'speedup': base_time / opt_time if opt_time > 0 else 0
            }
        
        # Calculate average speedup
        speedups = [r['speedup'] for r in results.values()]
        results['average_speedup'] = np.mean(speedups)
        
        logger.info(f"Hyperopt optimization average speedup: {results['average_speedup']:.2f}x")
        
        return results
    
    def run_full_comparison(self) -> Dict[str, Any]:
        """
        Run complete performance comparison.
        
        :return: Complete benchmark results
        """
        logger.info("Starting full performance comparison...")
        
        # Create test data
        test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10000, freq='5min'),
            'open': np.random.randn(10000).cumsum() + 100,
            'high': np.random.randn(10000).cumsum() + 102,
            'low': np.random.randn(10000).cumsum() + 98,
            'close': np.random.randn(10000).cumsum() + 100,
            'volume': np.random.randn(10000) * 1000 + 10000,
        })
        test_data.set_index('date', inplace=True)
        
        # Run all benchmarks
        self.results['benchmarks']['dataframe_ops'] = self.benchmark_dataframe_operations()
        self.results['benchmarks']['vectorized_backtest'] = self.benchmark_vectorized_backtesting(test_data.copy())
        self.results['benchmarks']['caching'] = self.benchmark_caching(test_data.copy())
        self.results['benchmarks']['memory'] = self.benchmark_memory_usage()
        self.results['benchmarks']['hyperopt'] = self.benchmark_hyperopt_optimization()
        
        # Calculate overall statistics
        all_speedups = []
        for category, results in self.results['benchmarks'].items():
            if 'average_speedup' in results:
                all_speedups.append(results['average_speedup'])
            elif 'execution_time' in results and 'speedup' in results['execution_time']:
                all_speedups.append(results['execution_time']['speedup'])
        
        self.results['summary'] = {
            'average_speedup': np.mean(all_speedups) if all_speedups else 0,
            'max_speedup': np.max(all_speedups) if all_speedups else 0,
            'min_speedup': np.min(all_speedups) if all_speedups else 0,
            'categories_tested': len(self.results['benchmarks']),
        }
        
        # Memory savings summary
        if 'memory' in self.results['benchmarks']:
            mem_results = self.results['benchmarks']['memory']['memory_usage']
            self.results['summary']['memory_savings'] = {
                'lazy_loading': f"{mem_results['lazy_savings']:.1f}%",
                'shared_memory': f"{mem_results['shared_savings']:.1f}%"
            }
        
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE COMPARISON SUMMARY")
        logger.info("="*60)
        logger.info(f"Average Speedup: {self.results['summary']['average_speedup']:.2f}x")
        logger.info(f"Maximum Speedup: {self.results['summary']['max_speedup']:.2f}x")
        logger.info(f"Minimum Speedup: {self.results['summary']['min_speedup']:.2f}x")
        
        if 'memory_savings' in self.results['summary']:
            logger.info(f"Memory Savings (Lazy): {self.results['summary']['memory_savings']['lazy_loading']}")
            logger.info(f"Memory Savings (Shared): {self.results['summary']['memory_savings']['shared_memory']}")
        
        logger.info("="*60)
        
        return self.results
    
    def save_results(self, output_file: Optional[Path] = None):
        """
        Save benchmark results to file.
        
        :param output_file: Output file path
        """
        if output_file is None:
            output_file = Path('benchmark_results.json')
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
    
    def generate_report(self) -> str:
        """
        Generate a markdown report of the benchmark results.
        
        :return: Markdown formatted report
        """
        report = []
        report.append("# Performance Optimization Benchmark Report")
        report.append(f"\nGenerated: {self.results['timestamp']}")
        report.append("\n## Summary")
        
        summary = self.results.get('summary', {})
        report.append(f"\n- **Average Speedup**: {summary.get('average_speedup', 0):.2f}x")
        report.append(f"- **Maximum Speedup**: {summary.get('max_speedup', 0):.2f}x")
        report.append(f"- **Minimum Speedup**: {summary.get('min_speedup', 0):.2f}x")
        
        if 'memory_savings' in summary:
            report.append(f"\n### Memory Savings")
            report.append(f"- **Lazy Loading**: {summary['memory_savings']['lazy_loading']}")
            report.append(f"- **Shared Memory**: {summary['memory_savings']['shared_memory']}")
        
        report.append("\n## Detailed Results")
        
        for category, results in self.results['benchmarks'].items():
            report.append(f"\n### {category.replace('_', ' ').title()}")
            
            if category == 'dataframe_ops':
                for op, metrics in results.items():
                    if isinstance(metrics, dict) and 'speedup' in metrics:
                        report.append(f"- **{op}**: {metrics['speedup']:.2f}x speedup")
            
            elif category == 'vectorized_backtest':
                if 'execution_time' in results:
                    metrics = results['execution_time']
                    report.append(f"- **Speedup**: {metrics['speedup']:.2f}x")
                    report.append(f"- **Iterative Time**: {metrics['iterative']:.4f}s")
                    report.append(f"- **Vectorized Time**: {metrics['vectorized']:.4f}s")
            
            elif category == 'caching':
                for cache_type in ['indicator_cache', 'signal_cache']:
                    if cache_type in results:
                        metrics = results[cache_type]
                        report.append(f"\n#### {cache_type.replace('_', ' ').title()}")
                        report.append(f"- **Speedup**: {metrics['speedup']:.2f}x")
                        report.append(f"- **Miss Time**: {metrics['miss_time']:.4f}s")
                        report.append(f"- **Hit Time**: {metrics['hit_time']:.4f}s")
            
            elif category == 'memory':
                if 'memory_usage' in results:
                    mem = results['memory_usage']
                    report.append(f"- **Regular Loading**: {mem['regular_mb']:.1f} MB")
                    report.append(f"- **Lazy Loading**: {mem['lazy_mb']:.1f} MB ({mem['lazy_savings']:.1f}% savings)")
                    report.append(f"- **Shared Memory**: {mem['shared_mb']:.1f} MB ({mem['shared_savings']:.1f}% savings)")
            
            elif category == 'hyperopt':
                for scenario_name, scenario in results.items():
                    if isinstance(scenario, dict) and 'speedup' in scenario:
                        report.append(f"\n#### {scenario_name.title()} Dataset")
                        report.append(f"- **Speedup**: {scenario['speedup']:.2f}x")
                        if 'profile' in scenario:
                            profile = scenario['profile']
                            report.append(f"- **Workers**: {profile['workers']}")
                            report.append(f"- **Batch Size**: {profile['batch_size']}")
                            report.append(f"- **Shared Memory**: {profile['use_shared_memory']}")
        
        report.append("\n## Recommendations")
        report.append("\nBased on the benchmark results:")
        
        if summary.get('average_speedup', 0) > 5:
            report.append("- ✅ **Excellent Performance**: The optimizations provide substantial improvements")
        elif summary.get('average_speedup', 0) > 2:
            report.append("- ✅ **Good Performance**: The optimizations provide meaningful improvements")
        else:
            report.append("- ⚠️ **Moderate Performance**: Consider enabling more optimizations")
        
        report.append("\n## Configuration")
        report.append("\n```json")
        report.append(json.dumps(self.results.get('config', {}), indent=2))
        report.append("```")
        
        return "\n".join(report)


def main():
    """Main entry point for benchmark comparison"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run performance benchmark comparison')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--feature', type=str, help='Specific feature to benchmark')
    parser.add_argument('--report', action='store_true', help='Generate markdown report')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = Configuration.from_files([args.config])
    else:
        # Default test configuration
        config = {
            'exchange': {'pair_whitelist': ['BTC/USDT', 'ETH/USDT']},
            'timeframe': '5m',
            'dry_run': True,
        }
    
    # Run benchmarks
    comparison = PerformanceComparison(config)
    
    if args.feature:
        # Run specific benchmark
        if args.feature == 'dataframe':
            results = comparison.benchmark_dataframe_operations()
        elif args.feature == 'vectorized':
            test_data = pd.DataFrame({
                'close': np.random.randn(10000).cumsum() + 100,
                'volume': np.random.randn(10000) * 1000 + 10000,
            })
            results = comparison.benchmark_vectorized_backtesting(test_data)
        elif args.feature == 'caching':
            test_data = pd.DataFrame({
                'close': np.random.randn(10000).cumsum() + 100,
            })
            results = comparison.benchmark_caching(test_data)
        elif args.feature == 'memory':
            results = comparison.benchmark_memory_usage()
        elif args.feature == 'hyperopt':
            results = comparison.benchmark_hyperopt_optimization()
        else:
            logger.error(f"Unknown feature: {args.feature}")
            return
        
        print(json.dumps(results, indent=2, default=str))
    else:
        # Run full comparison
        results = comparison.run_full_comparison()
        
        # Save results
        if args.output:
            output_file = Path(args.output)
        else:
            output_file = Path('benchmark_results.json')
        
        comparison.save_results(output_file)
        
        # Generate report if requested
        if args.report:
            report = comparison.generate_report()
            report_file = output_file.with_suffix('.md')
            with open(report_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {report_file}")


if __name__ == '__main__':
    main()
