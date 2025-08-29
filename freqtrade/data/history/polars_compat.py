"""
Pandas-Polars compatibility layer.

This module provides utilities for seamless conversion between pandas and Polars
DataFrames, ensuring backward compatibility while leveraging Polars performance.
"""

import logging
from typing import Union, Optional, Dict, Any, List
from functools import wraps
import pandas as pd
import polars as pl
import numpy as np

from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS


logger = logging.getLogger(__name__)


class DataFrameWrapper:
    """
    A wrapper that can hold either pandas or Polars DataFrame
    and provide transparent conversion when needed.
    """
    
    def __init__(self, data: Union[pd.DataFrame, pl.DataFrame]):
        """
        Initialize wrapper with either pandas or Polars DataFrame.
        
        :param data: DataFrame (pandas or Polars)
        """
        self._data = data
        self._is_polars = isinstance(data, (pl.DataFrame, pl.LazyFrame))
        
    @property
    def pandas(self) -> pd.DataFrame:
        """Get data as pandas DataFrame."""
        if self._is_polars:
            return polars_to_pandas(self._data)
        return self._data
    
    @property
    def polars(self) -> pl.DataFrame:
        """Get data as Polars DataFrame."""
        if not self._is_polars:
            return pandas_to_polars(self._data)
        return self._data
    
    def __len__(self) -> int:
        """Get length of DataFrame."""
        if self._is_polars:
            if isinstance(self._data, pl.LazyFrame):
                return self._data.collect().height
            return self._data.height
        return len(self._data)
    
    def __repr__(self) -> str:
        """String representation."""
        dtype = "Polars" if self._is_polars else "Pandas"
        return f"DataFrameWrapper({dtype}, {len(self)} rows)"


def polars_to_pandas(
    df: Union[pl.DataFrame, pl.LazyFrame],
    set_date_index: bool = True
) -> pd.DataFrame:
    """
    Convert Polars DataFrame to pandas DataFrame with proper types and index.
    
    :param df: Polars DataFrame or LazyFrame
    :param set_date_index: Set 'date' column as index if True
    :return: pandas DataFrame
    """
    # Collect if lazy
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    
    # Handle empty DataFrame
    if df.height == 0:
        pd_df = pd.DataFrame()
        if 'date' in df.columns and set_date_index:
            pd_df.index.name = 'date'
        return pd_df
    
    # Convert to pandas using pyarrow for efficiency
    pd_df = df.to_pandas(use_pyarrow_extension_array=False)
    
    # Handle date column
    if 'date' in pd_df.columns:
        # Ensure datetime type with UTC timezone
        pd_df['date'] = pd.to_datetime(pd_df['date'], utc=True)
        
        if set_date_index:
            pd_df.set_index('date', inplace=True)
    
    # Ensure correct dtypes for OHLCV columns
    ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in ohlcv_columns:
        if col in pd_df.columns:
            pd_df[col] = pd_df[col].astype('float64')
    
    # Handle additional columns that might be present
    if 'buy' in pd_df.columns:
        pd_df['buy'] = pd_df['buy'].astype('int8')
    if 'sell' in pd_df.columns:
        pd_df['sell'] = pd_df['sell'].astype('int8')
    if 'enter_long' in pd_df.columns:
        pd_df['enter_long'] = pd_df['enter_long'].astype('int8')
    if 'exit_long' in pd_df.columns:
        pd_df['exit_long'] = pd_df['exit_long'].astype('int8')
    if 'enter_short' in pd_df.columns:
        pd_df['enter_short'] = pd_df['enter_short'].astype('int8')
    if 'exit_short' in pd_df.columns:
        pd_df['exit_short'] = pd_df['exit_short'].astype('int8')
    
    return pd_df


def pandas_to_polars(
    df: pd.DataFrame,
    lazy: bool = False
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Convert pandas DataFrame to Polars DataFrame with proper types.
    
    :param df: pandas DataFrame
    :param lazy: Return LazyFrame if True
    :return: Polars DataFrame or LazyFrame
    """
    # Handle empty DataFrame
    if len(df) == 0:
        schema = {}
        if df.index.name == 'date' or 'date' in df.columns:
            schema['date'] = pl.Datetime('us', 'UTC')
        for col in df.columns:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                schema[col] = pl.Float64
            elif col in ['buy', 'sell', 'enter_long', 'exit_long', 'enter_short', 'exit_short']:
                schema[col] = pl.Int8
            else:
                schema[col] = pl.Float64
        
        pl_df = pl.DataFrame(schema=schema)
        return pl_df.lazy() if lazy else pl_df
    
    # Reset index if date is the index
    if df.index.name == 'date':
        df = df.reset_index()
    elif hasattr(df.index, 'name') and df.index.name:
        df = df.reset_index()
    
    # Convert to Polars
    pl_df = pl.from_pandas(df)
    
    # Handle date column
    if 'date' in pl_df.columns:
        pl_df = pl_df.with_columns([
            pl.col('date').cast(pl.Datetime('us', 'UTC'))
        ])
    
    # Ensure correct dtypes for OHLCV columns
    type_mappings = []
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in pl_df.columns:
            type_mappings.append(pl.col(col).cast(pl.Float64))
    
    # Handle signal columns
    for col in ['buy', 'sell', 'enter_long', 'exit_long', 'enter_short', 'exit_short']:
        if col in pl_df.columns:
            type_mappings.append(pl.col(col).cast(pl.Int8))
    
    if type_mappings:
        pl_df = pl_df.with_columns(type_mappings)
    
    return pl_df.lazy() if lazy else pl_df


def ensure_polars(df: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    """
    Ensure DataFrame is in Polars format.
    
    :param df: pandas or Polars DataFrame
    :return: Polars DataFrame
    """
    if isinstance(df, pl.DataFrame):
        return df
    elif isinstance(df, pl.LazyFrame):
        return df.collect()
    elif isinstance(df, pd.DataFrame):
        return pandas_to_polars(df)
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df)}")


def ensure_pandas(df: Union[pd.DataFrame, pl.DataFrame]) -> pd.DataFrame:
    """
    Ensure DataFrame is in pandas format.
    
    :param df: pandas or Polars DataFrame
    :return: pandas DataFrame
    """
    if isinstance(df, pd.DataFrame):
        return df
    elif isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        return polars_to_pandas(df)
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df)}")


def auto_convert(func):
    """
    Decorator that automatically converts between pandas and Polars.
    
    Functions decorated with this will accept either pandas or Polars
    DataFrames and return the same type as input.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Find DataFrame arguments
        df_args = []
        df_positions = []
        
        for i, arg in enumerate(args):
            if isinstance(arg, (pd.DataFrame, pl.DataFrame, pl.LazyFrame)):
                df_args.append(arg)
                df_positions.append(i)
        
        # Determine input type
        input_is_polars = any(isinstance(df, (pl.DataFrame, pl.LazyFrame)) for df in df_args)
        
        # Convert all DataFrames to Polars for processing
        args = list(args)
        for i, pos in enumerate(df_positions):
            args[pos] = ensure_polars(df_args[i])
        
        # Call function with Polars DataFrames
        result = func(*args, **kwargs)
        
        # Convert result back to original type if needed
        if isinstance(result, (pl.DataFrame, pl.LazyFrame)):
            if not input_is_polars:
                result = polars_to_pandas(result)
        elif isinstance(result, dict):
            # Handle dict of DataFrames
            for key, value in result.items():
                if isinstance(value, (pl.DataFrame, pl.LazyFrame)):
                    if not input_is_polars:
                        result[key] = polars_to_pandas(value)
        
        return result
    
    return wrapper


def merge_dataframes(
    df1: Union[pd.DataFrame, pl.DataFrame],
    df2: Union[pd.DataFrame, pl.DataFrame],
    on: Union[str, List[str]] = 'date',
    how: str = 'left'
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Merge two DataFrames (pandas or Polars) efficiently.
    
    :param df1: First DataFrame
    :param df2: Second DataFrame
    :param on: Column(s) to merge on
    :param how: Merge type ('left', 'right', 'inner', 'outer')
    :return: Merged DataFrame (same type as df1)
    """
    # Determine output type based on df1
    output_pandas = isinstance(df1, pd.DataFrame)
    
    # Convert to Polars for processing
    pl_df1 = ensure_polars(df1)
    pl_df2 = ensure_polars(df2)
    
    # Perform merge in Polars (faster)
    merged = pl_df1.join(pl_df2, on=on, how=how)
    
    # Convert back if needed
    if output_pandas:
        return polars_to_pandas(merged)
    return merged


def concat_dataframes(
    dfs: List[Union[pd.DataFrame, pl.DataFrame]],
    axis: int = 0
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Concatenate multiple DataFrames efficiently.
    
    :param dfs: List of DataFrames
    :param axis: Axis to concatenate on (0 for rows, 1 for columns)
    :return: Concatenated DataFrame (same type as first in list)
    """
    if not dfs:
        return pd.DataFrame()
    
    # Determine output type based on first DataFrame
    output_pandas = isinstance(dfs[0], pd.DataFrame)
    
    # Convert all to Polars
    pl_dfs = [ensure_polars(df) for df in dfs]
    
    # Concatenate
    if axis == 0:
        concatenated = pl.concat(pl_dfs, how='vertical')
    else:
        concatenated = pl.concat(pl_dfs, how='horizontal')
    
    # Convert back if needed
    if output_pandas:
        return polars_to_pandas(concatenated)
    return concatenated


class PolarsCompatibilityLayer:
    """
    Compatibility layer that provides pandas-like interface for Polars DataFrames.
    
    This allows existing code to work with minimal changes while using Polars
    under the hood for performance.
    """
    
    def __init__(self, use_polars: bool = True):
        """
        Initialize compatibility layer.
        
        :param use_polars: Use Polars if True, pandas if False
        """
        self.use_polars = use_polars
        logger.info(f"Compatibility layer initialized with {'Polars' if use_polars else 'pandas'}")
    
    def read_parquet(self, path: str, **kwargs) -> Union[pd.DataFrame, pl.DataFrame]:
        """Read parquet file."""
        if self.use_polars:
            df = pl.read_parquet(path)
            # Apply any pandas-specific kwargs if needed
            if 'columns' in kwargs:
                df = df.select(kwargs['columns'])
            return df
        else:
            return pd.read_parquet(path, **kwargs)
    
    def read_feather(self, path: str, **kwargs) -> Union[pd.DataFrame, pl.DataFrame]:
        """Read feather/arrow file."""
        if self.use_polars:
            return pl.read_ipc(path)
        else:
            return pd.read_feather(path, **kwargs)
    
    def to_parquet(self, df: Union[pd.DataFrame, pl.DataFrame], path: str, **kwargs):
        """Write DataFrame to parquet."""
        if self.use_polars:
            pl_df = ensure_polars(df)
            compression = kwargs.get('compression', 'snappy')
            pl_df.write_parquet(path, compression=compression)
        else:
            pd_df = ensure_pandas(df)
            pd_df.to_parquet(path, **kwargs)
    
    def to_feather(self, df: Union[pd.DataFrame, pl.DataFrame], path: str, **kwargs):
        """Write DataFrame to feather/arrow."""
        if self.use_polars:
            pl_df = ensure_polars(df)
            compression = kwargs.get('compression', 'lz4')
            pl_df.write_ipc(path, compression=compression)
        else:
            pd_df = ensure_pandas(df)
            pd_df.to_feather(path, **kwargs)
    
    def convert_for_strategy(self, df: Union[pd.DataFrame, pl.DataFrame]) -> pd.DataFrame:
        """
        Convert DataFrame to format expected by strategies (always pandas).
        
        :param df: Input DataFrame
        :return: pandas DataFrame with proper format for strategies
        """
        pd_df = ensure_pandas(df)
        
        # Ensure date index
        if 'date' in pd_df.columns and pd_df.index.name != 'date':
            pd_df.set_index('date', inplace=True)
        
        # Ensure required columns exist
        for col in DEFAULT_DATAFRAME_COLUMNS:
            if col not in pd_df.columns and col != 'date':
                pd_df[col] = 0.0
        
        return pd_df
    
    def optimize_dataframe(self, df: pd.DataFrame) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Optimize DataFrame by converting to Polars if beneficial.
        
        :param df: pandas DataFrame
        :return: Optimized DataFrame (Polars if use_polars=True)
        """
        if self.use_polars and len(df) > 1000:  # Only convert for larger DataFrames
            return pandas_to_polars(df)
        return df


# Global compatibility layer instance
compat_layer = PolarsCompatibilityLayer(use_polars=True)
