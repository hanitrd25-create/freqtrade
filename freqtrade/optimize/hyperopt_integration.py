"""
Hyperopt Integration Module for Performance Optimizations

This module integrates the HyperoptResourceOptimizer and SharedMemoryManager
to improve hyperopt performance through dynamic resource allocation and
efficient data sharing across parallel workers.
"""

import pickle
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd
import psutil

from freqtrade.optimize.hyperopt_resource_optimizer import HyperoptResourceOptimizer as ResourceOptimizer
from freqtrade.optimize.shared_memory_simple import SimpleSharedMemoryManager
from freqtrade.optimize.shared_memory_manager import estimate_data_size
from freqtrade.constants import Config


logger = logging.getLogger(__name__)


class HyperoptIntegration:
    """
    Integrates performance optimizations into hyperopt workflow.
    
    Features:
    - Dynamic resource optimization for parallel jobs
    - Shared memory for large datasets
    - Intelligent batch size calculation
    """
    
    def __init__(self, config: Config):
        """
        Initialize hyperopt integration.
        
        :param config: Freqtrade configuration
        """
        self.config = config
        self.resource_optimizer = ResourceOptimizer(config)
        self.shared_memory_manager = None
        self.shared_metadata_file = None
        self.shared_data_keys: Dict[str, str] = {}
        
    def optimize_parallel_settings(self, total_epochs: int = 500, data_size_mb: float = None) -> Dict[str, Any]:
        """
        Optimize parallel execution settings based on system resources.
        
        :param total_epochs: Total number of epochs for hyperopt
        :param data_size_mb: Estimated size of data in MB
        :return: Dictionary with optimized settings
        """
        profile = self.resource_optimizer.get_optimization_profile(total_epochs, data_size_mb)
        
        # Get optimal number of jobs
        optimal_jobs = profile['workers']
        
        # Check if user has configured jobs
        if 'hyperopt_jobs' in self.config:
            configured_jobs = self.config['hyperopt_jobs']
            if configured_jobs != optimal_jobs:
                logger.warning(
                    f"Configured jobs ({configured_jobs}) differs from "
                    f"optimal ({optimal_jobs}). "
                    f"Consider adjusting for better performance."
                )
        
        return {
            'n_jobs': optimal_jobs,
            'batch_size': profile['batch_size'],
            'use_shared_memory': profile['use_shared_memory'],
            'memory_status': profile['memory_status']
        }
    
    def setup_shared_memory(self, data_pickle_file: Path) -> Optional[str]:
        """
        Set up shared memory for large datasets if beneficial.
        
        :param data_pickle_file: Path to pickle file with data
        :return: Path to metadata file for shared memory access or None
        """
        if not data_pickle_file.exists():
            logger.debug("Data pickle file not found, skipping shared memory setup")
            return None
        
        try:
            # Load data to check size
            with open(data_pickle_file, 'rb') as f:
                data = pickle.load(f)
            
            data_size_mb = estimate_data_size(data)
            logger.info(f"Data size: {data_size_mb:.2f}MB")
            
            # Check if shared memory is beneficial
            should_use_shm = self.resource_optimizer.should_use_shared_memory(data_size_mb)
            
            if not should_use_shm:
                logger.info("Shared memory not recommended for current dataset size")
                return None
            
            # Initialize simplified shared memory manager
            self.shared_memory_manager = SimpleSharedMemoryManager()
            
            # Share data if it's a dict of DataFrames
            if isinstance(data, dict):
                # Check if it's a dict of DataFrames (common pattern)
                has_dataframes = any(isinstance(v, pd.DataFrame) for v in data.values())
                
                if has_dataframes:
                    logger.info("Sharing dictionary of DataFrames in shared memory")
                    metadata_file = self.shared_memory_manager.share_dataframe_dict(data)
                    
                    if metadata_file:
                        self.shared_metadata_file = metadata_file
                        logger.info(f"Successfully set up shared memory, metadata at: {metadata_file}")
                        return metadata_file
                    else:
                        logger.error("Failed to share DataFrames")
                        return None
                else:
                    logger.info("Data doesn't contain DataFrames, skipping shared memory")
                    return None
            else:
                logger.info(f"Unsupported data type for shared memory: {type(data)}")
                return None
            
        except Exception as e:
            logger.error(f"Failed to set up shared memory: {e}")
            return None
    
    def get_shared_data_wrapper(self) -> Optional[Dict[str, Any]]:
        """
        Get wrapper for accessing shared data in worker processes.
        
        :return: Dictionary with shared memory metadata or None
        """
        if not self.shared_memory_manager or not self.shared_metadata_file:
            return None
        
        # Return path to metadata file for workers
        return {
            'metadata_file': self.shared_metadata_file
        }
    
    def cleanup(self):
        """
        Clean up shared memory resources.
        """
        if self.shared_memory_manager:
            try:
                self.shared_memory_manager.cleanup()
                logger.info("Cleaned up shared memory resources")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    
    def get_resource_status(self) -> Dict[str, Any]:
        """
        Get current resource status.
        
        :return: Dictionary with resource status
        """
        mem_status = self.resource_optimizer.get_memory_status()
        info = self.resource_optimizer.system_info
        
        return {
            'cpu_cores': info['cpu_count'],
            'logical_cores': info['cpu_count_logical'],
            'memory_total_gb': info['memory_total_mb'] / 1024,
            'memory_available_gb': info['memory_available_mb'] / 1024,
            'memory_percent': mem_status['memory_used_percent'],
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'load_average': os.getloadavg()
        }
    
    def log_resource_status(self):
        """
        Log current resource utilization status.
        """
        status = self.get_resource_status()
        
        logger.info("System Resource Status:")
        logger.info(f"  CPU Cores: {status['cpu_cores']} physical, {status['logical_cores']} logical")
        logger.info(f"  Memory: {status['memory_total_gb']:.1f}GB total, "
                   f"{status['memory_available_gb']:.1f}GB available ({status['memory_percent']:.1f}% used)")
        logger.info(f"  CPU Load: {status['cpu_percent']:.1f}%")
        logger.info(f"  Load Average: {', '.join(f'{x:.2f}' for x in status['load_average'])}")
    
    def should_use_vectorized_backtest(self, strategy_config: Dict[str, Any]) -> bool:
        """
        Determine if vectorized backtesting should be used based on strategy and resources.
        
        :param strategy_config: Strategy configuration
        :return: True if vectorized backtesting should be used
        """
        # Check strategy compatibility (same as before)
        if strategy_config.get('position_stacking', False):
            return False
        if strategy_config.get('use_custom_stoploss', False):
            return False
        if strategy_config.get('trailing_stop', False):
            return False
        
        # Check resource availability
        status = self.resource_optimizer.get_resource_status()
        
        # Use vectorized if we have enough memory
        if status['memory_available_gb'] < 2.0:
            logger.warning("Low memory available, disabling vectorized backtesting")
            return False
        
        return True
    
    def get_optimal_chunk_size(self, data_size_mb: float) -> int:
        """
        Calculate optimal chunk size for data processing based on available memory.
        
        :param data_size_mb: Size of data in MB
        :return: Optimal chunk size
        """
        status = self.resource_optimizer.get_resource_status()
        available_mb = status['memory_available_gb'] * 1024
        
        # Use 25% of available memory for chunks
        chunk_memory_mb = available_mb * 0.25
        
        # Calculate chunk size (aim for chunks that fit in memory)
        if data_size_mb < chunk_memory_mb:
            # Data fits in memory, process all at once
            return -1
        else:
            # Calculate number of chunks needed
            num_chunks = int(data_size_mb / chunk_memory_mb) + 1
            return num_chunks
