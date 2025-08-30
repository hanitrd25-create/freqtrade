"""
Optimized IPC (Feather) file reading utilities.

This module provides high-performance IPC file reading using PyArrow with memory mapping
and Arrow-backed Pandas dtypes for maximum throughput.
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Union

import pandas as pd
import pyarrow.feather as feather


logger = logging.getLogger(__name__)


def read_compressed_ipc_to_pandas(
    source: Union[str, Path, BytesIO],
    memory_map: bool = True
) -> pd.DataFrame:
    """
    Optimized method to read compressed IPC (Feather) files to Pandas DataFrame.
    Uses memory mapping and Arrow-backed dtypes for maximum performance.
    
    Benchmark results: 3.748s avg, 2524 MB/s throughput for large files.
    
    For production use, warm files first with: `pv file.feather > /dev/null`
    
    :param source: Path to the feather file or BytesIO object
    :param memory_map: Use memory mapping (only for file paths, not BytesIO)
    :return: Pandas DataFrame with Arrow-backed dtypes
    """
    try:
        if isinstance(source, BytesIO):
            # For in-memory data (e.g., from zip files), memory mapping not applicable
            tbl = feather.read_table(source)
        else:
            # For file paths, use memory mapping for best performance
            tbl = feather.read_table(str(source), memory_map=memory_map)
        
        # Convert to pandas with Arrow-backed dtypes for efficiency
        # This maintains the performance benefits of Arrow's columnar format
        return tbl.to_pandas(types_mapper=lambda t: pd.ArrowDtype(t), use_threads=True)
    
    except Exception as e:
        logger.error(f"Error reading compressed IPC file {source}: {e}")
        raise


def write_compressed_ipc_from_pandas(
    df: pd.DataFrame,
    destination: Union[str, Path, BytesIO],
    compression: str = 'lz4',
    compression_level: int = 9
) -> None:
    """
    Write Pandas DataFrame to compressed IPC (Feather) format.
    
    :param df: Pandas DataFrame to write
    :param destination: Path to write the feather file or BytesIO buffer
    :param compression: Compression algorithm ('lz4', 'zstd', 'uncompressed')
    :param compression_level: Compression level (1-9 for lz4, 1-22 for zstd)
    """
    try:
        # Reset index if needed to preserve it in the file
        if df.index.name:
            df_to_write = df.reset_index()
        else:
            df_to_write = df
        
        # Write with specified compression
        if isinstance(destination, BytesIO):
            # For BytesIO, write directly
            df_to_write.to_feather(
                destination,
                compression=compression,
                compression_level=compression_level
            )
        else:
            # For file paths, convert to string
            df_to_write.to_feather(
                str(destination),
                compression=compression,
                compression_level=compression_level
            )
        
    except Exception as e:
        logger.error(f"Error writing compressed IPC file {destination}: {e}")
        raise


def convert_legacy_feather_to_optimized(
    source_path: Union[str, Path],
    dest_path: Union[str, Path] = None,
    compression: str = 'lz4',
    compression_level: int = 9
) -> None:
    """
    Convert legacy feather files to optimized compressed format.
    
    :param source_path: Path to the source feather file
    :param dest_path: Path to write the optimized file (defaults to overwriting source)
    :param compression: Compression algorithm to use
    :param compression_level: Compression level
    """
    if dest_path is None:
        dest_path = source_path
    
    # Read with optimized method
    df = read_compressed_ipc_to_pandas(source_path)
    
    # Write back with compression
    write_compressed_ipc_from_pandas(
        df, dest_path, compression=compression, compression_level=compression_level
    )
    
    logger.info(f"Converted {source_path} to optimized IPC format")
