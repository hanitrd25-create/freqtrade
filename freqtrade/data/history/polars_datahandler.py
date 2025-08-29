"""
Polars-based data handler for ultra-fast data loading.

This module provides high-performance data loading using Polars
while maintaining backward compatibility with pandas-based strategies.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone

import polars as pl
import pandas as pd
import numpy as np

from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS, Config
from freqtrade.configuration import TimeRange
from freqtrade.data.history.datahandlers.idatahandler import IDataHandler
from freqtrade.enums import CandleType, TradingMode
from freqtrade.util import format_ms_time


logger = logging.getLogger(__name__)


class PolarsDataHandler(IDataHandler):
    """
    High-performance data handler using Polars for I/O operations.
    
    This handler uses Polars for all file I/O operations to maximize
    loading speed, while providing pandas DataFrames for backward
    compatibility with existing strategies.
    """
    
    def __init__(self, datadir: Path):
        """
        Initialize the Polars data handler.
        
        :param datadir: Directory to use for data storage
        """
        super().__init__(datadir)
        self.default_suffix = 'arrow'  # Use Arrow IPC File format by default for speed
        
    def _get_file_path(
        self,
        pair: str,
        timeframe: str,
        candle_type: CandleType,
        suffix: Optional[str] = None
    ) -> Path:
        """
        Generate file path for given parameters.
        
        :param pair: Trading pair
        :param timeframe: Timeframe (e.g., '5m', '1h')
        :param candle_type: Type of candles (spot, futures, etc.)
        :param suffix: File extension (arrow, feather, or parquet). 
                      'arrow' and 'feather' both use Arrow IPC File format (Feather v2)
        :return: Path to data file
        """
        if suffix is None:
            suffix = self._default_format
        pair_s = pair.replace("/", "_").replace(":", "_")
        candle_type_s = candle_type.value if candle_type else ""
        
        if candle_type_s:
            filename = f"{pair_s}-{timeframe}-{candle_type_s}.{suffix}"
        else:
            filename = f"{pair_s}-{timeframe}.{suffix}"
            
        return self._datadir / filename
    
    def load_pair_history_polars(
        self,
        pair: str,
        timeframe: str,
        candle_type: CandleType = CandleType.SPOT,
        timerange: Optional[Any] = None,
        fill_missing: bool = True,
        drop_incomplete: bool = False,
        startup_candles: int = 0,
        suffix: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Load pair history as Polars DataFrame (internal use).
        
        :param pair: Trading pair
        :param timeframe: Timeframe
        :param candle_type: Candle type
        :param timerange: Time range to filter
        :param fill_missing: Fill missing candles
        :param drop_incomplete: Drop incomplete candles
        :param startup_candles: Number of startup candles
        :param suffix: File extension (arrow, feather, or parquet)
        :return: Polars DataFrame with OHLCV data
        """
        if suffix is None:
            suffix = self._default_format
            
        # Try to find the file in order of speed: arrow -> feather -> parquet
        filename = None
        actual_suffix = suffix
        
        # First try the requested format
        test_filename = self._get_file_path(pair, timeframe, candle_type, suffix)
        if test_filename.exists():
            filename = test_filename
        else:
            # Try other formats in order of loading speed
            for try_suffix in ['arrow', 'feather', 'parquet']:
                if try_suffix != suffix:
                    test_filename = self._get_file_path(pair, timeframe, candle_type, try_suffix)
                    if test_filename.exists():
                        filename = test_filename
                        actual_suffix = try_suffix
                        logger.debug(f"Using {try_suffix} format for {pair}/{timeframe}")
                        break
        
        if not filename or not filename.exists():
            logger.warning(f"No data file found for {pair}/{timeframe}")
            return pl.DataFrame()
        
        try:
            # Load data using Polars optimized for each format
            if actual_suffix in ['arrow', 'feather']:
                # Arrow IPC File format (Feather v2) - fastest loading
                # Both 'arrow' and 'feather' extensions use the same format
                if self._use_lazy:
                    df = pl.scan_ipc(filename, memory_map=True)
                else:
                    df = pl.read_ipc(filename, memory_map=True)
            elif actual_suffix == 'parquet':
                # Parquet format - good compression but slower than Arrow IPC
                if self._use_lazy:
                    df = pl.scan_parquet(filename)
                else:
                    df = pl.read_parquet(filename)
            else:
                raise ValueError(f"Unsupported file format: {actual_suffix}")
            
            # Apply time range filter if specified (using Polars lazy evaluation)
            if timerange:
                if hasattr(timerange, 'startts') and timerange.startts:
                    start_date = datetime.fromtimestamp(timerange.startts, tz=timezone.utc)
                    df = df.filter(pl.col('date') >= start_date)
                    
                if hasattr(timerange, 'stopts') and timerange.stopts:
                    stop_date = datetime.fromtimestamp(timerange.stopts, tz=timezone.utc)
                    df = df.filter(pl.col('date') <= stop_date)
            
            # Collect if using lazy evaluation
            if isinstance(df, pl.LazyFrame):
                df = df.collect()
            
            # Sort by date (Polars is very fast at sorting)
            df = df.sort('date')
            
            # Drop duplicates
            df = df.unique(subset=['date'])
            
            # Fill missing values if requested
            if fill_missing and len(df) > 0:
                df = self._fill_missing_polars(df, timeframe)
            
            # Drop incomplete candles if requested
            if drop_incomplete and len(df) > 0:
                df = self._drop_incomplete_polars(df, timeframe)
            
            # Apply startup candles offset
            if startup_candles and len(df) > startup_candles:
                df = df.slice(startup_candles)
            
            logger.info(f"Loaded {len(df)} candles for {pair}/{timeframe} using Polars")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return pl.DataFrame()
    
    def load_pair_history(
        self,
        pair: str,
        timeframe: str,
        candle_type: CandleType = CandleType.SPOT,
        timerange: Optional[Any] = None,
        fill_missing: bool = True,
        drop_incomplete: bool = False,
        startup_candles: int = 0,
        convert_to_pandas: bool = True,
        suffix: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load pair history data (pandas-compatible interface).
        
        :param pair: Trading pair
        :param timeframe: Timeframe
        :param candle_type: Candle type
        :param timerange: Time range to filter
        :param fill_missing: Fill missing candles
        :param drop_incomplete: Drop incomplete candles
        :param startup_candles: Number of startup candles
        :param convert_to_pandas: Convert to pandas DataFrame
        :param suffix: File extension (arrow, feather, or parquet)
        :return: DataFrame with OHLCV data
        """
        # Load data using Polars for speed
        pl_df = self.load_pair_history_polars(
            pair=pair,
            timeframe=timeframe,
            candle_type=candle_type,
            timerange=timerange,
            fill_missing=fill_missing,
            drop_incomplete=drop_incomplete,
            startup_candles=startup_candles,
            suffix=suffix
        )
        
        if len(pl_df) == 0:
            return pd.DataFrame()
        
        if convert_to_pandas:
            # Convert to pandas for backward compatibility
            return self.polars_to_pandas(pl_df)
        else:
            return pl_df
    
    def save_pair_history(
        self,
        pair: str,
        timeframe: str,
        data: pd.DataFrame,
        candle_type: CandleType = CandleType.SPOT,
        suffix: Optional[str] = None
    ) -> None:
        """
        Save pair history data using Polars for speed.
        
        :param pair: Trading pair
        :param timeframe: Timeframe
        :param data: DataFrame to save
        :param candle_type: Candle type
        :param suffix: File extension (arrow, feather, or parquet)
        """
        if suffix is None:
            suffix = self._default_format
            
        filename = self._get_file_path(pair, timeframe, candle_type, suffix)
        
        # Ensure directory exists
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert pandas to Polars if needed
        if isinstance(data, pd.DataFrame):
            pl_df = self.pandas_to_polars(data)
        else:
            pl_df = data
        
        # Save using Polars optimized for each format
        if suffix in ['arrow', 'feather']:
            # Arrow IPC File format (Feather v2) - fastest for reading
            # Use lz4 compression for balance of speed and size
            pl_df.write_ipc(filename, compression='lz4')
        elif suffix == 'parquet':
            # Parquet format - better compression but slower reads
            pl_df.write_parquet(
                filename,
                compression='snappy',  # Fast compression
                statistics=True,  # Enable statistics for faster queries
                row_group_size=50000  # Optimize for our typical data size
            )
        else:
            # Default to Arrow IPC for unknown formats
            logger.warning(f"Unknown format {suffix}, defaulting to Arrow IPC File format")
            pl_df.write_ipc(filename, compression='lz4')
        
        logger.info(f"Saved {len(pl_df)} candles to {filename} using Polars")
    
    @staticmethod
    def polars_to_pandas(pl_df: pl.DataFrame) -> pd.DataFrame:
        """
        Convert Polars DataFrame to pandas DataFrame efficiently.
        
        :param pl_df: Polars DataFrame
        :return: pandas DataFrame
        """
        # Use arrow for zero-copy conversion when possible
        pd_df = pl_df.to_pandas(use_pyarrow_extension_array=False)
        
        # Ensure date column is datetime with UTC timezone
        if 'date' in pd_df.columns:
            pd_df['date'] = pd.to_datetime(pd_df['date'], utc=True)
            pd_df.set_index('date', inplace=True)
        
        # Ensure correct dtypes for OHLCV columns
        float_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in float_columns:
            if col in pd_df.columns:
                pd_df[col] = pd_df[col].astype('float64')
        
        return pd_df
    
    @staticmethod
    def pandas_to_polars(pd_df: pd.DataFrame) -> pl.DataFrame:
        """
        Convert pandas DataFrame to Polars DataFrame efficiently.
        
        :param pd_df: pandas DataFrame
        :return: Polars DataFrame
        """
        # Reset index if date is the index
        if pd_df.index.name == 'date':
            pd_df = pd_df.reset_index()
        
        # Convert to Polars
        pl_df = pl.from_pandas(pd_df)
        
        # Ensure date column is datetime
        if 'date' in pl_df.columns:
            pl_df = pl_df.with_columns([
                pl.col('date').cast(pl.Datetime('us', 'UTC'))
            ])
        
        return pl_df
    
    def _fill_missing_polars(self, df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
        """
        Fill missing candles in Polars DataFrame.
        
        :param df: Polars DataFrame
        :param timeframe: Timeframe string
        :return: DataFrame with filled missing candles
        """
        if len(df) < 2:
            return df
        
        # Convert timeframe to timedelta
        from freqtrade.exchange import timeframe_to_minutes
        minutes = timeframe_to_minutes(timeframe)
        interval = pl.duration(minutes=minutes)
        
        # Get date range
        start_date = df['date'].min()
        end_date = df['date'].max()
        
        # Create complete date range
        date_range = pl.datetime_range(
            start_date,
            end_date,
            interval,
            time_unit='us',
            time_zone='UTC'
        ).alias('date')
        
        # Create DataFrame with complete date range
        complete_df = pl.DataFrame({'date': date_range})
        
        # Left join to fill missing dates
        filled_df = complete_df.join(df, on='date', how='left')
        
        # Forward fill for missing values
        filled_df = filled_df.with_columns([
            pl.col(col).forward_fill()
            for col in ['open', 'high', 'low', 'close', 'volume']
            if col in filled_df.columns
        ])
        
        return filled_df
    
    def _drop_incomplete_polars(self, df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
        """
        Drop incomplete candles from Polars DataFrame.
        
        :param df: Polars DataFrame
        :param timeframe: Timeframe string
        :return: DataFrame without incomplete candles
        """
        if len(df) == 0:
            return df
        
        # Get current time
        now = datetime.now(timezone.utc)
        
        # Convert timeframe to minutes
        from freqtrade.exchange import timeframe_to_minutes
        minutes = timeframe_to_minutes(timeframe)
        
        # Calculate the start of current candle
        current_candle_start = now.replace(
            minute=(now.minute // minutes) * minutes,
            second=0,
            microsecond=0
        )
        
        # Filter out current incomplete candle
        return df.filter(pl.col('date') < current_candle_start)
    
    def load_multiple_pairs(
        self,
        pairs: List[str],
        timeframe: str,
        candle_type: CandleType = CandleType.SPOT,
        timerange: Optional[Any] = None,
        fill_missing: bool = True,
        drop_incomplete: bool = False,
        startup_candles: int = 0,
        convert_to_pandas: bool = True,
        suffix: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple pairs in parallel using Polars for maximum speed.
        
        :param pairs: List of trading pairs
        :param timeframe: Timeframe
        :param candle_type: Candle type
        :param timerange: Time range to filter
        :param fill_missing: Fill missing candles
        :param drop_incomplete: Drop incomplete candles
        :param startup_candles: Number of startup candles
        :param convert_to_pandas: Convert to pandas DataFrames
        :param suffix: File extension
        :return: Dictionary of DataFrames keyed by pair
        """
        result = {}
        
        # Load all pairs (Polars can handle parallel I/O efficiently)
        for pair in pairs:
            df = self.load_pair_history(
                pair=pair,
                timeframe=timeframe,
                candle_type=candle_type,
                timerange=timerange,
                fill_missing=fill_missing,
                drop_incomplete=drop_incomplete,
                startup_candles=startup_candles,
                convert_to_pandas=convert_to_pandas,
                suffix=suffix
            )
            
            if len(df) > 0:
                result[pair] = df
            else:
                logger.warning(f"No data loaded for {pair}")
        
        logger.info(f"Loaded data for {len(result)}/{len(pairs)} pairs using Polars")
        
        return result
    
    def get_available_data(self, suffix: Optional[str] = None) -> List[Tuple[str, str, CandleType]]:
        """
        Get list of available data files.
        
        :param suffix: File extension to look for (None = all supported formats)
        :return: List of (pair, timeframe, candle_type) tuples
        """
        available = []
        
        # Check all supported formats if suffix not specified
        suffixes = [suffix] if suffix else ['arrow', 'feather', 'parquet']
        
        for check_suffix in suffixes:
            for file in self._datadir.glob(f"*.{check_suffix}"):
                parts = file.stem.split('-')
                
                if len(parts) >= 2:
                    pair = parts[0].replace('_', '/')
                    timeframe = parts[1]
                    
                    candle_type = CandleType.SPOT
                    if len(parts) >= 3:
                        try:
                            candle_type = CandleType(parts[2])
                        except ValueError:
                            pass
                    
                    # Avoid duplicates if both arrow and feather files exist
                    if (pair, timeframe, candle_type) not in available:
                        available.append((pair, timeframe, candle_type))
        
        return available
    
    @classmethod
    def _get_file_extension(cls) -> str:
        """
        Get file extension for this particular datahandler.
        Returns 'arrow' as the default extension.
        """
        return 'arrow'
    
    def _trades_store(self, pair: str, data: pd.DataFrame, trading_mode: TradingMode) -> None:
        """
        Store trades data to file.
        
        :param pair: Pair - used for filename
        :param data: DataFrame containing trades
        :param trading_mode: Trading mode to use
        """
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to Polars and save
        df_pl = pl.from_pandas(data)
        
        # Use Arrow IPC for trades as well
        df_pl.write_ipc(filename, compression='lz4')
        logger.info(f"Trades for {pair} saved to {filename}")
    
    def trades_append(self, pair: str, data: pd.DataFrame) -> None:
        """
        Append trades data to existing file.
        
        :param pair: Pair - used for filename  
        :param data: DataFrame containing trades to append
        """
        # For now, we'll use SPOT mode. In production, this should be configurable
        trading_mode = TradingMode.SPOT
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        
        if filename.exists():
            # Load existing data
            existing_df = pl.read_ipc(filename, memory_map=True)
            new_df = pl.from_pandas(data)
            
            # Concatenate and remove duplicates
            combined_df = pl.concat([existing_df, new_df], how='vertical')
            combined_df = combined_df.unique(maintain_order=True)
            
            # Save back
            combined_df.write_ipc(filename, compression='lz4')
        else:
            # No existing file, just store
            self._trades_store(pair, data, trading_mode)
    
    def _trades_load(
        self, pair: str, trading_mode: TradingMode, timerange: Optional[TimeRange] = None
    ) -> pd.DataFrame:
        """
        Load trades data from file.
        
        :param pair: Load trades for this pair
        :param trading_mode: Trading mode to use
        :param timerange: Timerange to load trades for
        :return: DataFrame containing trades
        """
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        
        if not filename.exists():
            logger.warning(f"No trades data found for {pair} at {filename}")
            return pd.DataFrame(columns=DEFAULT_TRADES_COLUMNS)
        
        try:
            # Load with Polars for speed
            df_pl = pl.read_ipc(filename, memory_map=True)
            
            # Apply timerange filter if specified
            if timerange:
                if timerange.startts:
                    df_pl = df_pl.filter(pl.col('timestamp') >= timerange.startts * 1000)
                if timerange.stopts:
                    df_pl = df_pl.filter(pl.col('timestamp') <= timerange.stopts * 1000)
            
            # Convert to pandas for compatibility
            return df_pl.to_pandas()
            
        except Exception as e:
            logger.error(f"Error loading trades for {pair}: {e}")
            return pd.DataFrame(columns=DEFAULT_TRADES_COLUMNS)
    
    def optimize_storage(self, target_format: str = 'arrow') -> Dict[str, Any]:
        """
        Optimize storage by converting to a more efficient format.
        
        :param target_format: Target format ('arrow' for speed, 'parquet' for size)
        :return: Optimization statistics
        """
        stats = {
            'files_optimized': 0,
            'size_before_mb': 0,
            'size_after_mb': 0,
            'time_taken_s': 0
        }
        
        import time
        start_time = time.time()
        
        available = self.get_available_data()  # Get all available data
        
        for pair, timeframe, candle_type in available:
            # Find existing file
            existing_filename = None
            for check_suffix in ['arrow', 'feather', 'parquet']:
                test_file = self._get_file_path(pair, timeframe, candle_type, check_suffix)
                if test_file.exists():
                    existing_filename = test_file
                    break
            
            if not existing_filename:
                continue
                
            target_filename = self._get_file_path(pair, timeframe, candle_type, target_format)
            
            # Skip if already in target format
            if existing_filename == target_filename:
                continue
                
            size_before = existing_filename.stat().st_size / 1024 / 1024
            stats['size_before_mb'] += size_before
            
            # Load data (auto-detects format)
            df = self.load_pair_history_polars(
                pair, timeframe, candle_type
            )
            
            if len(df) > 0:
                # Save in target format
                self.save_pair_history(
                    pair, timeframe, df, candle_type, target_format
                )
                
                size_after = target_filename.stat().st_size / 1024 / 1024
                stats['size_after_mb'] += size_after
                stats['files_optimized'] += 1
                
                # Optionally remove old file if conversion successful
                if existing_filename != target_filename and target_filename.exists():
                    existing_filename.unlink()
                    logger.info(f"Converted {pair}/{timeframe} to {target_format}: "
                              f"{size_before:.2f}MB -> {size_after:.2f}MB "
                              f"({(1 - size_after/size_before) * 100:.1f}% reduction)")
        
        stats['time_taken_s'] = time.time() - start_time
        stats['compression_ratio'] = 1 - (stats['size_after_mb'] / stats['size_before_mb']) \
                                    if stats['size_before_mb'] > 0 else 0
        
        logger.info(f"Storage optimization complete: {stats['files_optimized']} files, "
                   f"{stats['compression_ratio']*100:.1f}% total reduction, "
                   f"{stats['time_taken_s']:.1f}s")
        
        return stats
