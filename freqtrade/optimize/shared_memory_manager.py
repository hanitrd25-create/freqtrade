"""
Shared Memory Manager for Hyperopt - Enables efficient data sharing across parallel workers
"""
import logging
import pickle
import hashlib
from multiprocessing import shared_memory
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SharedMemoryManager:
    """
    Manages shared memory for large datasets in hyperopt parallelization.
    This reduces memory usage by avoiding data duplication across workers.
    """
    
    def __init__(self):
        """Initialize the shared memory manager."""
        self.shared_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self.metadata: Dict[str, Dict] = {}
        
    def _generate_key(self, data: Any, prefix: str = "data") -> str:
        """
        Generate a unique key for the data based on its content.
        
        :param data: Data to generate key for
        :param prefix: Prefix for the key
        :return: Unique key string
        """
        # Create a hash of the data for uniqueness
        if isinstance(data, pd.DataFrame):
            data_bytes = pickle.dumps(data.values)
        elif isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        else:
            data_bytes = pickle.dumps(data)
        
        data_hash = hashlib.md5(data_bytes).hexdigest()[:8]
        return f"{prefix}_{data_hash}"
    
    def share_dataframe(self, df: pd.DataFrame, key: Optional[str] = None) -> str:
        """
        Share a DataFrame in shared memory.
        
        :param df: DataFrame to share
        :param key: Optional key for the shared memory block
        :return: Key to access the shared DataFrame
        """
        if key is None:
            key = self._generate_key(df, "df")
        
        # Check if already shared
        if key in self.shared_blocks:
            logger.debug(f"DataFrame already shared with key: {key}")
            return key
        
        try:
            # Convert DataFrame to numpy array for efficient sharing
            values = df.values
            
            # Create shared memory block
            shm = shared_memory.SharedMemory(create=True, size=values.nbytes)
            
            # Copy data to shared memory
            shared_array = np.ndarray(
                values.shape, 
                dtype=values.dtype, 
                buffer=shm.buf
            )
            shared_array[:] = values[:]
            
            # Store metadata including column dtypes for proper reconstruction
            self.metadata[key] = {
                'shape': values.shape,
                'dtype': str(values.dtype),  # Store as string for serialization
                'columns': df.columns.tolist(),
                'column_dtypes': {col: str(df[col].dtype) for col in df.columns},
                'index': df.index.tolist(),
                'shm_name': shm.name,
                'type': 'dataframe'
            }
            
            self.shared_blocks[key] = shm
            logger.info(f"Shared DataFrame with key: {key}, size: {values.nbytes / (1024*1024):.2f}MB")
            
            return key
            
        except Exception as e:
            logger.error(f"Failed to share DataFrame: {e}")
            raise
    
    def get_dataframe_by_name(self, shm_name: str, shape: Tuple, dtype_str: str, 
                               columns: list, index: list, column_dtypes: Dict) -> pd.DataFrame:
        """
        Retrieve a DataFrame from shared memory using explicit parameters.
        
        :param shm_name: Name of the shared memory block
        :param shape: Shape of the array
        :param dtype_str: String representation of dtype
        :param columns: Column names
        :param index: Index values
        :param column_dtypes: Original column dtypes
        :return: Retrieved DataFrame
        """
        try:
            # Access existing shared memory
            shm = shared_memory.SharedMemory(name=shm_name)
            
            # Reconstruct numpy array
            shared_array = np.ndarray(
                shape,
                dtype=np.dtype(dtype_str),
                buffer=shm.buf
            )
            
            # Create DataFrame (copy to avoid issues when shared memory is released)
            df = pd.DataFrame(
                shared_array.copy(),
                columns=columns,
                index=index
            )
            
            # Restore original column dtypes
            for col, dtype_str in column_dtypes.items():
                try:
                    df[col] = df[col].astype(dtype_str)
                except Exception:
                    pass  # Skip if dtype conversion fails
            
            # Close the shared memory reference
            shm.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to retrieve DataFrame from shared memory: {e}")
            raise
    
    def share_dict(self, data: Dict, key: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Share a dictionary of DataFrames in shared memory.
        
        :param data: Dictionary to share (typically pair: DataFrame mapping)
        :param key: Optional key for the shared memory block
        :return: Key to access the shared dictionary
        """
        if key is None:
            key = self._generate_key(data, "dict")
        
        # Check if already shared
        if key in self.metadata and self.metadata[key]['type'] == 'dict':
            logger.debug(f"Dictionary already shared with key: {key}")
            return key
        
        try:
            # Share each DataFrame in the dictionary
            shared_keys = {}
            for k, v in data.items():
                if isinstance(v, pd.DataFrame):
                    df_key = f"{key}_{k}"
                    self.share_dataframe(v, df_key)
                    shared_keys[k] = df_key
                else:
                    # For non-DataFrame values, pickle them
                    shared_keys[k] = pickle.dumps(v)
            
            # Store metadata
            self.metadata[key] = {
                'shared_keys': shared_keys,
                'type': 'dict'
            }
            
            total_size = sum(
                self.metadata[v]['shape'][0] * self.metadata[v]['shape'][1] * 8 / (1024*1024)
                for v in shared_keys.values() 
                if isinstance(v, str) and v in self.metadata
            )
            logger.info(f"Shared dictionary with key: {key}, total size: {total_size:.2f}MB")
            
            return key, self.metadata[key]
            
        except Exception as e:
            logger.error(f"Failed to share dictionary: {e}")
            raise
    
    def get_dict(self, key: str, metadata: Optional[Dict[str, Dict]] = None) -> Dict:
        """
        Retrieve a dictionary from shared memory.
        
        :param key: Key of the shared dictionary
        :param metadata: Optional metadata dict for all shared memory blocks
        :return: Retrieved dictionary
        """
        # Use provided metadata or local metadata
        all_metadata = metadata if metadata else self.metadata
        
        if key not in all_metadata:
            raise KeyError(f"No shared data found with key: {key}")
        
        meta = all_metadata[key]
        if meta['type'] != 'dict':
            raise TypeError(f"Data with key {key} is not a dictionary")
        
        try:
            # Reconstruct dictionary by retrieving each DataFrame
            result = {}
            for k, v in meta['shared_keys'].items():
                if isinstance(v, str) and v in all_metadata:
                    # It's a shared DataFrame
                    result[k] = self.get_dataframe(v, all_metadata[v])
                elif isinstance(v, bytes):
                    # It's a pickled value
                    result[k] = pickle.loads(v)
                else:
                    result[k] = v
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve dictionary: {e}")
            raise
    
    def cleanup(self, key: Optional[str] = None):
        """
        Clean up shared memory blocks.
        
        :param key: Optional key to clean up specific block, or None to clean all
        """
        if key:
            keys_to_clean = [key]
            # Also clean sub-keys for dictionaries
            if key in self.metadata and self.metadata[key]['type'] == 'dict':
                keys_to_clean.extend([
                    v for v in self.metadata[key]['shared_keys'].values()
                    if isinstance(v, str) and v in self.metadata
                ])
        else:
            # Clean all blocks and metadata
            keys_to_clean = list(self.shared_blocks.keys())
            # Also include any metadata entries without shared blocks (e.g., dict metadata)
            keys_to_clean.extend([k for k in self.metadata.keys() if k not in self.shared_blocks])
        
        for k in keys_to_clean:
            if k in self.shared_blocks:
                try:
                    self.shared_blocks[k].close()
                    self.shared_blocks[k].unlink()
                    del self.shared_blocks[k]
                    logger.debug(f"Cleaned up shared memory block: {k}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup shared memory block {k}: {e}")
            
            if k in self.metadata:
                del self.metadata[k]
    
    def __del__(self):
        """Cleanup all shared memory on deletion."""
        self.cleanup()


class SharedDataWrapper:
    """
    Wrapper class to handle shared data access in worker processes.
    """
    
    def __init__(self, shared_key: str, manager_metadata: Dict):
        """
        Initialize the wrapper with shared data key and metadata.
        
        :param shared_key: Key to access shared data
        :param manager_metadata: Metadata from the manager
        """
        self.shared_key = shared_key
        self.metadata = manager_metadata
        self._cached_data = None
    
    def get_data(self) -> Any:
        """
        Get the shared data. Caches the result for repeated access.
        
        :return: The shared data
        """
        if self._cached_data is None:
            if self.metadata['type'] == 'dataframe':
                self._cached_data = self._get_dataframe()
            elif self.metadata['type'] == 'dict':
                self._cached_data = self._get_dict()
            else:
                raise ValueError(f"Unknown data type: {self.metadata['type']}")
        
        return self._cached_data
    
    def _get_dataframe(self) -> pd.DataFrame:
        """Retrieve DataFrame from shared memory."""
        meta = self.metadata
        
        # Access existing shared memory
        shm = shared_memory.SharedMemory(name=meta['shm_name'])
        
        # Reconstruct numpy array
        shared_array = np.ndarray(
            meta['shape'],
            dtype=np.dtype(meta['dtype']),  # Convert string back to dtype
            buffer=shm.buf
        )
        
        # Create DataFrame
        df = pd.DataFrame(
            shared_array.copy(),
            columns=meta['columns'],
            index=meta['index']
        )
        
        # Restore original column dtypes
        if 'column_dtypes' in meta:
            for col, dtype_str in meta['column_dtypes'].items():
                df[col] = df[col].astype(dtype_str)
        
        shm.close()
        return df
    
    def _get_dict(self) -> Dict:
        """Retrieve dictionary from shared memory."""
        result = {}
        for k, v in self.metadata['shared_keys'].items():
            if isinstance(v, str) and v in self.metadata:
                # Recursively get shared DataFrames
                wrapper = SharedDataWrapper(v, self.metadata[v])
                result[k] = wrapper.get_data()
            else:
                # Unpickle non-DataFrame data
                result[k] = pickle.loads(v)
        
        return result


def estimate_data_size(data: Any) -> float:
    """
    Estimate the size of data in MB.
    
    :param data: Data to estimate size for
    :return: Estimated size in MB
    """
    if isinstance(data, pd.DataFrame):
        return data.memory_usage(deep=True).sum() / (1024 * 1024)
    elif isinstance(data, dict):
        total_size = 0
        for v in data.values():
            if isinstance(v, pd.DataFrame):
                total_size += v.memory_usage(deep=True).sum()
        return total_size / (1024 * 1024)
    elif isinstance(data, np.ndarray):
        return data.nbytes / (1024 * 1024)
    else:
        # Rough estimate using pickle
        return len(pickle.dumps(data)) / (1024 * 1024)
