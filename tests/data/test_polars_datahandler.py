"""
Tests for Polars data handler
"""
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import pytest
import pandas as pd
import polars as pl
import numpy as np

from freqtrade.data.history.polars_datahandler import PolarsDataHandler
from freqtrade.data.history.polars_compat import polars_to_pandas, pandas_to_polars
from freqtrade.enums import CandleType
from freqtrade.configuration import TimeRange


class TestPolarsDataHandler:
    """Test suite for PolarsDataHandler"""
    
    @pytest.fixture
    def temp_datadir(self):
        """Create a temporary directory for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_ohlcv_df(self):
        """Create a sample OHLCV pandas DataFrame"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        data = {
            'date': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 115, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        return df
    
    def test_initialization(self, temp_datadir):
        """Test PolarsDataHandler initialization"""
        handler = PolarsDataHandler(datadir=temp_datadir)
        assert handler._datadir == temp_datadir
        assert handler.default_suffix == 'arrow'
    
    def test_ohlcv_filename_generation(self, temp_datadir):
        """Test OHLCV filename generation for different formats"""
        handler = PolarsDataHandler(datadir=temp_datadir)
        
        # Test Arrow IPC format (default)
        filename = handler._pair_data_filename(
            pair='BTC/USDT',
            timeframe='1h',
            candle_type=CandleType.SPOT,
            suffix='arrow'
        )
        assert filename == temp_datadir / 'BTC_USDT-1h-spot.arrow'
        
        # Test Feather format
        filename = handler._pair_data_filename(
            pair='BTC/USDT',
            timeframe='1h',
            candle_type=CandleType.SPOT,
            suffix='feather'
        )
        assert filename == temp_datadir / 'BTC_USDT-1h-spot.feather'
        
        # Test Parquet format
        filename = handler._pair_data_filename(
            pair='BTC/USDT',
            timeframe='1h',
            candle_type=CandleType.SPOT,
            suffix='parquet'
        )
        assert filename == temp_datadir / 'BTC_USDT-1h-spot.parquet'
    
    def test_ohlcv_store_and_load_arrow(self, temp_datadir, sample_ohlcv_df):
        """Test storing and loading OHLCV data in Arrow IPC format"""
        handler = PolarsDataHandler(datadir=temp_datadir)
        
        # Store data
        handler.ohlcv_store(
            pair='BTC/USDT',
            timeframe='1h',
            data=sample_ohlcv_df,
            candle_type=CandleType.SPOT
        )
        
        # Verify file exists
        expected_file = temp_datadir / 'BTC_USDT-1h-spot.arrow'
        assert expected_file.exists()
        
        # Load data back
        loaded_df = handler._ohlcv_load(
            pair='BTC/USDT',
            timeframe='1h',
            timerange=None,
            candle_type=CandleType.SPOT
        )
        
        # Verify data integrity
        pd.testing.assert_frame_equal(sample_ohlcv_df, loaded_df)
    
    def test_ohlcv_store_and_load_feather(self, temp_datadir, sample_ohlcv_df):
        """Test storing and loading OHLCV data in Feather format"""
        handler = PolarsDataHandler(datadir=temp_datadir)
        handler.default_suffix = 'feather'
        
        # Store data
        handler.ohlcv_store(
            pair='ETH/USDT',
            timeframe='5m',
            data=sample_ohlcv_df,
            candle_type=CandleType.SPOT
        )
        
        # Verify file exists
        expected_file = temp_datadir / 'ETH_USDT-5m-spot.feather'
        assert expected_file.exists()
        
        # Load data back
        loaded_df = handler._ohlcv_load(
            pair='ETH/USDT',
            timeframe='5m',
            timerange=None,
            candle_type=CandleType.SPOT
        )
        
        # Verify data integrity
        pd.testing.assert_frame_equal(sample_ohlcv_df, loaded_df)
    
    def test_ohlcv_store_and_load_parquet(self, temp_datadir, sample_ohlcv_df):
        """Test storing and loading OHLCV data in Parquet format"""
        handler = PolarsDataHandler(datadir=temp_datadir)
        handler.default_suffix = 'parquet'
        
        # Store data
        handler.ohlcv_store(
            pair='SOL/USDT',
            timeframe='15m',
            data=sample_ohlcv_df,
            candle_type=CandleType.SPOT
        )
        
        # Verify file exists
        expected_file = temp_datadir / 'SOL_USDT-15m-spot.parquet'
        assert expected_file.exists()
        
        # Load data back
        loaded_df = handler._ohlcv_load(
            pair='SOL/USDT',
            timeframe='15m',
            timerange=None,
            candle_type=CandleType.SPOT
        )
        
        # Verify data integrity
        pd.testing.assert_frame_equal(sample_ohlcv_df, loaded_df)
    
    def test_format_fallback(self, temp_datadir, sample_ohlcv_df):
        """Test automatic format fallback when loading data"""
        handler = PolarsDataHandler(datadir=temp_datadir)
        
        # Store data in parquet format
        handler.default_suffix = 'parquet'
        handler.ohlcv_store(
            pair='ADA/USDT',
            timeframe='30m',
            data=sample_ohlcv_df,
            candle_type=CandleType.SPOT
        )
        
        # Reset to arrow format (which doesn't exist)
        handler.default_suffix = 'arrow'
        
        # Should fall back to parquet and load successfully
        loaded_df = handler._ohlcv_load(
            pair='ADA/USDT',
            timeframe='30m',
            timerange=None,
            candle_type=CandleType.SPOT
        )
        
        # Verify data loaded correctly
        pd.testing.assert_frame_equal(sample_ohlcv_df, loaded_df)
    
    def test_timerange_filtering(self, temp_datadir, sample_ohlcv_df):
        """Test loading data with timerange filtering"""
        handler = PolarsDataHandler(datadir=temp_datadir)
        
        # Store data
        handler.ohlcv_store(
            pair='DOT/USDT',
            timeframe='1h',
            data=sample_ohlcv_df,
            candle_type=CandleType.SPOT
        )
        
        # Create timerange for middle 50 candles
        start_time = sample_ohlcv_df.index[25]
        end_time = sample_ohlcv_df.index[74]
        timerange = TimeRange()
        timerange.starttype = 'date'
        timerange.stoptype = 'date'
        timerange.startdt = start_time
        timerange.stopdt = end_time
        
        # Load with timerange
        loaded_df = handler._ohlcv_load(
            pair='DOT/USDT',
            timeframe='1h',
            timerange=timerange,
            candle_type=CandleType.SPOT
        )
        
        # Verify correct subset loaded
        assert len(loaded_df) == 50
        assert loaded_df.index[0] == start_time
        assert loaded_df.index[-1] == end_time
    
    def test_ohlcv_append(self, temp_datadir, sample_ohlcv_df):
        """Test appending data to existing OHLCV file"""
        handler = PolarsDataHandler(datadir=temp_datadir)
        
        # Store initial data (first 50 rows)
        initial_data = sample_ohlcv_df.iloc[:50]
        handler.ohlcv_store(
            pair='MATIC/USDT',
            timeframe='1h',
            data=initial_data,
            candle_type=CandleType.SPOT
        )
        
        # Append new data (last 50 rows)
        new_data = sample_ohlcv_df.iloc[50:]
        handler.ohlcv_append(
            pair='MATIC/USDT',
            timeframe='1h',
            data=new_data,
            candle_type=CandleType.SPOT
        )
        
        # Load all data
        loaded_df = handler._ohlcv_load(
            pair='MATIC/USDT',
            timeframe='1h',
            timerange=None,
            candle_type=CandleType.SPOT
        )
        
        # Verify complete data
        pd.testing.assert_frame_equal(sample_ohlcv_df, loaded_df)
    
    def test_load_multiple_pairs(self, temp_datadir, sample_ohlcv_df):
        """Test loading multiple pairs in parallel"""
        handler = PolarsDataHandler(datadir=temp_datadir)
        
        pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        # Store data for multiple pairs
        for pair in pairs:
            handler.ohlcv_store(
                pair=pair,
                timeframe='1h',
                data=sample_ohlcv_df,
                candle_type=CandleType.SPOT
            )
        
        # Load multiple pairs
        result = handler.load_multiple_pairs(
            pairs=pairs,
            timeframe='1h',
            candle_type=CandleType.SPOT
        )
        
        # Verify all pairs loaded
        assert len(result) == 3
        for pair in pairs:
            assert pair in result
            pd.testing.assert_frame_equal(sample_ohlcv_df, result[pair])
    
    def test_ohlcv_get_available_data(self, temp_datadir, sample_ohlcv_df):
        """Test getting list of available data"""
        handler = PolarsDataHandler(datadir=temp_datadir)
        
        # Store data in different formats
        handler.default_suffix = 'arrow'
        handler.ohlcv_store('BTC/USDT', '1h', sample_ohlcv_df, CandleType.SPOT)
        
        handler.default_suffix = 'feather'
        handler.ohlcv_store('ETH/USDT', '5m', sample_ohlcv_df, CandleType.SPOT)
        
        handler.default_suffix = 'parquet'
        handler.ohlcv_store('SOL/USDT', '15m', sample_ohlcv_df, CandleType.FUTURES)
        
        # Get available data
        available = handler.ohlcv_get_available_data()
        
        # Verify results
        assert len(available) == 3
        assert ('BTC/USDT', '1h', CandleType.SPOT) in available
        assert ('ETH/USDT', '5m', CandleType.SPOT) in available
        assert ('SOL/USDT', '15m', CandleType.FUTURES) in available
    
    def test_optimize_storage(self, temp_datadir, sample_ohlcv_df):
        """Test storage optimization by converting formats"""
        handler = PolarsDataHandler(datadir=temp_datadir)
        
        # Store data in parquet format
        handler.default_suffix = 'parquet'
        handler.ohlcv_store('AVAX/USDT', '1d', sample_ohlcv_df, CandleType.SPOT)
        
        parquet_file = temp_datadir / 'AVAX_USDT-1d-spot.parquet'
        assert parquet_file.exists()
        parquet_size = parquet_file.stat().st_size
        
        # Optimize storage to Arrow IPC format
        result = handler.optimize_storage(target_format='arrow')
        
        # Verify conversion
        arrow_file = temp_datadir / 'AVAX_USDT-1d-spot.arrow'
        assert arrow_file.exists()
        assert result['converted'] == 1
        assert result['errors'] == 0
        
        # Load and verify data integrity
        handler.default_suffix = 'arrow'
        loaded_df = handler._ohlcv_load(
            pair='AVAX/USDT',
            timeframe='1d',
            timerange=None,
            candle_type=CandleType.SPOT
        )
        pd.testing.assert_frame_equal(sample_ohlcv_df, loaded_df)
    
    def test_pandas_polars_conversion(self, sample_ohlcv_df):
        """Test conversion between pandas and polars DataFrames"""
        # Convert pandas to polars
        polars_df = pandas_to_polars(sample_ohlcv_df)
        assert isinstance(polars_df, pl.DataFrame)
        assert len(polars_df) == len(sample_ohlcv_df)
        
        # Convert back to pandas
        pandas_df = polars_to_pandas(polars_df)
        assert isinstance(pandas_df, pd.DataFrame)
        
        # Verify data integrity
        pd.testing.assert_frame_equal(sample_ohlcv_df, pandas_df)
    
    def test_empty_dataframe_handling(self, temp_datadir):
        """Test handling of empty DataFrames"""
        handler = PolarsDataHandler(datadir=temp_datadir)
        
        # Create empty DataFrame with correct columns
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        empty_df.index.name = 'date'
        
        # Store empty DataFrame
        handler.ohlcv_store(
            pair='EMPTY/USDT',
            timeframe='1h',
            data=empty_df,
            candle_type=CandleType.SPOT
        )
        
        # Load empty DataFrame
        loaded_df = handler._ohlcv_load(
            pair='EMPTY/USDT',
            timeframe='1h',
            timerange=None,
            candle_type=CandleType.SPOT
        )
        
        # Verify empty DataFrame
        assert len(loaded_df) == 0
        assert list(loaded_df.columns) == ['open', 'high', 'low', 'close', 'volume']
    
    def test_performance_arrow_vs_parquet(self, temp_datadir):
        """Benchmark Arrow IPC vs Parquet loading performance"""
        import time
        
        handler = PolarsDataHandler(datadir=temp_datadir)
        
        # Create larger dataset for meaningful benchmark
        dates = pd.date_range(start='2020-01-01', periods=100000, freq='1min')
        large_df = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(100, 110, 100000),
            'high': np.random.uniform(110, 120, 100000),
            'low': np.random.uniform(90, 100, 100000),
            'close': np.random.uniform(95, 115, 100000),
            'volume': np.random.uniform(1000, 10000, 100000)
        })
        large_df.set_index('date', inplace=True)
        
        # Store in Arrow IPC format
        handler.default_suffix = 'arrow'
        handler.ohlcv_store('PERF/USDT', '1m', large_df, CandleType.SPOT)
        
        # Store in Parquet format
        handler.default_suffix = 'parquet'
        handler.ohlcv_store('PERF/USDT', '5m', large_df, CandleType.SPOT)
        
        # Benchmark Arrow IPC loading
        handler.default_suffix = 'arrow'
        start = time.time()
        for _ in range(10):
            _ = handler._ohlcv_load('PERF/USDT', '1m', None, CandleType.SPOT)
        arrow_time = time.time() - start
        
        # Benchmark Parquet loading
        handler.default_suffix = 'parquet'
        start = time.time()
        for _ in range(10):
            _ = handler._ohlcv_load('PERF/USDT', '5m', None, CandleType.SPOT)
        parquet_time = time.time() - start
        
        # Arrow IPC should be faster
        print(f"Arrow IPC loading time: {arrow_time:.3f}s")
        print(f"Parquet loading time: {parquet_time:.3f}s")
        print(f"Arrow IPC is {parquet_time/arrow_time:.2f}x faster")
        
        # Assert Arrow is at least as fast as Parquet
        assert arrow_time <= parquet_time * 1.1  # Allow 10% margin
