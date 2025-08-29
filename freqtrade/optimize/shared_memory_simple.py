"""
Simplified Shared Memory Manager for Hyperopt - Avoids complex metadata passing
"""
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from multiprocessing import shared_memory

logger = logging.getLogger(__name__)


class SimpleSharedMemoryManager:
    """
    Simplified shared memory manager that uses file-based metadata exchange.
    This avoids Python crashes from complex object serialization in multiprocessing.
    """
    
    def __init__(self):
        """Initialize the manager."""
        self.shared_blocks = {}
        self.metadata_file = None
        
    def share_dataframe_dict(self, data: Dict[str, pd.DataFrame], 
                            prefix: str = "hyperopt") -> Optional[str]:
        """
        Share a dictionary of DataFrames and save metadata to a file.
        
        :param data: Dictionary mapping pair names to DataFrames
        :param prefix: Prefix for the shared memory names
        :return: Path to metadata file or None on error
        """
        try:
            metadata = {}
            
            for pair, df in data.items():
                if not isinstance(df, pd.DataFrame):
                    continue
                    
                # Convert to numpy array
                values = df.values.astype(np.float64)  # Ensure consistent dtype
                
                # Create shared memory with simple name
                shm_name = f"{prefix}_{pair.replace('/', '_')}"
                
                # Try to clean up any existing shared memory with this name
                try:
                    old_shm = shared_memory.SharedMemory(name=shm_name)
                    old_shm.close()
                    old_shm.unlink()
                except:
                    pass
                
                # Create new shared memory
                shm = shared_memory.SharedMemory(create=True, size=values.nbytes)
                
                # Copy data
                shared_array = np.ndarray(values.shape, dtype=np.float64, buffer=shm.buf)
                shared_array[:] = values[:]
                
                # Store simple metadata
                metadata[pair] = {
                    'shm_name': shm.name,
                    'shape': list(values.shape),
                    'columns': df.columns.tolist(),
                    'index': df.index.tolist()
                }
                
                self.shared_blocks[pair] = shm
                logger.info(f"Shared {pair}: {values.nbytes / (1024*1024):.2f}MB")
            
            # Save metadata to temporary file
            fd, metadata_file = tempfile.mkstemp(suffix='.json', prefix='ft_shm_')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            os.close(fd)
            
            self.metadata_file = metadata_file
            logger.info(f"Metadata saved to: {metadata_file}")
            
            return metadata_file
            
        except Exception as e:
            logger.error(f"Failed to share data: {e}")
            return None
    
    @staticmethod
    def load_from_metadata(metadata_file: str) -> Dict[str, pd.DataFrame]:
        """
        Load shared DataFrames using metadata file.
        
        :param metadata_file: Path to metadata JSON file
        :return: Dictionary of DataFrames
        """
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            result = {}
            for pair, info in metadata.items():
                # Access shared memory
                shm = shared_memory.SharedMemory(name=info['shm_name'])
                
                # Reconstruct array
                shared_array = np.ndarray(
                    tuple(info['shape']), 
                    dtype=np.float64, 
                    buffer=shm.buf
                )
                
                # Create DataFrame
                df = pd.DataFrame(
                    shared_array.copy(),
                    columns=info['columns'],
                    index=info['index']
                )
                
                # Close shared memory reference
                shm.close()
                
                result[pair] = df
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to load from metadata: {e}")
            return {}
    
    def cleanup(self):
        """Clean up shared memory and metadata file."""
        # Clean up shared memory blocks
        for shm in self.shared_blocks.values():
            try:
                shm.close()
                shm.unlink()
            except:
                pass
        
        # Clean up metadata file
        if self.metadata_file and os.path.exists(self.metadata_file):
            try:
                os.unlink(self.metadata_file)
            except:
                pass
        
        self.shared_blocks.clear()
        logger.info("Cleaned up shared memory resources")
