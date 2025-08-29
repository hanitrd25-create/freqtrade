"""
Hyperopt Resource Optimizer - Dynamic batch size optimization based on system resources
"""
import logging
import psutil
from typing import Dict, Tuple
import os
from functools import lru_cache

logger = logging.getLogger(__name__)


class HyperoptResourceOptimizer:
    """
    Optimizes hyperopt batch sizes and parallelization based on available system resources.
    """
    
    # Constants for resource calculations
    MIN_MEMORY_PER_WORKER_MB = 512  # Minimum memory per worker in MB
    OPTIMAL_MEMORY_PER_WORKER_MB = 1024  # Optimal memory per worker in MB
    MAX_MEMORY_USAGE_PERCENT = 80  # Maximum % of system memory to use
    MIN_WORKERS = 1
    MAX_WORKERS_PER_CORE = 2  # Maximum workers per CPU core
    
    # Batch size constants
    MIN_BATCH_SIZE = 1
    DEFAULT_BATCH_SIZE = 10
    MAX_BATCH_SIZE = 100
    
    def __init__(self, config: Dict = None):
        """
        Initialize the resource optimizer.
        
        :param config: Configuration dictionary
        """
        self.config = config or {}
        self.system_info = self._get_system_info()
        self._log_system_info()
    
    @lru_cache(maxsize=1)
    def _get_system_info(self) -> Dict:
        """
        Get system resource information.
        
        :return: Dictionary with system information
        """
        return {
            'cpu_count': psutil.cpu_count(logical=False) or 1,
            'cpu_count_logical': psutil.cpu_count(logical=True) or 1,
            'memory_total_mb': psutil.virtual_memory().total / (1024 * 1024),
            'memory_available_mb': psutil.virtual_memory().available / (1024 * 1024),
            'memory_percent_used': psutil.virtual_memory().percent,
        }
    
    def _log_system_info(self):
        """Log system information."""
        info = self.system_info
        logger.info(f"System Resources: {info['cpu_count']} physical CPUs "
                   f"({info['cpu_count_logical']} logical), "
                   f"{info['memory_total_mb']:.0f}MB total memory, "
                   f"{info['memory_available_mb']:.0f}MB available")
    
    def get_optimal_workers(self, data_size_mb: float = None) -> int:
        """
        Calculate optimal number of parallel workers based on system resources.
        
        :param data_size_mb: Estimated size of data in MB (optional)
        :return: Optimal number of workers
        """
        # Start with CPU-based calculation
        cpu_count = self.system_info['cpu_count']
        logical_count = self.system_info['cpu_count_logical']
        
        # Use logical CPUs if hyperthreading is available, but limit to 1.5x physical
        cpu_based_workers = min(logical_count, int(cpu_count * 1.5))
        
        # Memory-based calculation
        available_memory = self.system_info['memory_available_mb']
        max_usable_memory = self.system_info['memory_total_mb'] * (self.MAX_MEMORY_USAGE_PERCENT / 100)
        available_for_workers = min(available_memory, max_usable_memory)
        
        # Calculate memory per worker based on data size
        if data_size_mb:
            # Each worker needs memory for data + overhead
            memory_per_worker = data_size_mb + self.MIN_MEMORY_PER_WORKER_MB
        else:
            memory_per_worker = self.OPTIMAL_MEMORY_PER_WORKER_MB
        
        memory_based_workers = int(available_for_workers / memory_per_worker)
        
        # Take the minimum of CPU and memory based calculations
        optimal_workers = max(
            self.MIN_WORKERS,
            min(cpu_based_workers, memory_based_workers, cpu_count * self.MAX_WORKERS_PER_CORE)
        )
        
        # Override with config if specified
        if 'hyperopt_jobs' in self.config and self.config['hyperopt_jobs'] > 0:
            config_workers = self.config['hyperopt_jobs']
            if config_workers != optimal_workers:
                logger.info(f"Using configured workers: {config_workers} "
                           f"(optimal would be {optimal_workers})")
            return config_workers
        
        logger.info(f"Optimal number of workers: {optimal_workers} "
                   f"(CPU suggests {cpu_based_workers}, "
                   f"memory suggests {memory_based_workers})")
        
        return optimal_workers
    
    def get_optimal_batch_size(self, total_epochs: int, num_workers: int = None) -> int:
        """
        Calculate optimal batch size for hyperopt based on epochs and workers.
        
        :param total_epochs: Total number of epochs to process
        :param num_workers: Number of parallel workers (auto-calculated if None)
        :return: Optimal batch size
        """
        if num_workers is None:
            num_workers = self.get_optimal_workers()
        
        # Basic calculation: aim for reasonable number of batches
        # Too many small batches = overhead, too few large batches = poor parallelization
        target_batches = max(10, total_epochs // 20)  # Aim for at least 10 batches
        
        # Calculate batch size based on workers and epochs
        batch_size = max(
            self.MIN_BATCH_SIZE,
            min(
                total_epochs // target_batches,
                num_workers * 2,  # Process at least 2 epochs per worker per batch
                self.MAX_BATCH_SIZE
            )
        )
        
        # Adjust based on available memory
        available_memory = self.system_info['memory_available_mb']
        if available_memory < 2000:  # Less than 2GB available
            batch_size = min(batch_size, num_workers)
            logger.info(f"Limited memory available, reducing batch size to {batch_size}")
        
        logger.info(f"Optimal batch size: {batch_size} for {total_epochs} epochs "
                   f"with {num_workers} workers")
        
        return batch_size
    
    def should_use_shared_memory(self, data_size_mb: float) -> bool:
        """
        Determine if shared memory should be used for large datasets.
        
        :param data_size_mb: Size of dataset in MB
        :return: True if shared memory should be used
        """
        # Use shared memory if dataset is large relative to available memory
        available_memory = self.system_info['memory_available_mb']
        
        # If dataset would be duplicated across workers and exceeds 20% of available memory
        threshold = available_memory * 0.2
        
        return data_size_mb > threshold
    
    def get_memory_status(self) -> Dict:
        """
        Get current memory status and recommendations.
        
        :return: Dictionary with memory status and recommendations
        """
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        status = {
            'memory_used_percent': mem.percent,
            'memory_available_mb': mem.available / (1024 * 1024),
            'swap_used_percent': swap.percent,
            'status': 'healthy',
            'recommendations': []
        }
        
        # Determine status and recommendations
        if mem.percent > 90:
            status['status'] = 'critical'
            status['recommendations'].append("Memory usage critical - reduce batch size or workers")
        elif mem.percent > 80:
            status['status'] = 'warning'
            status['recommendations'].append("Memory usage high - consider reducing batch size")
        
        if swap.percent > 50:
            status['recommendations'].append("High swap usage detected - performance may be degraded")
        
        return status
    
    def adjust_for_current_load(self, initial_workers: int) -> int:
        """
        Adjust number of workers based on current system load.
        
        :param initial_workers: Initially calculated number of workers
        :return: Adjusted number of workers
        """
        # Get current CPU load
        cpu_percent = psutil.cpu_percent(interval=0.1)
        load_avg = os.getloadavg()[0]  # 1-minute load average
        
        adjusted_workers = initial_workers
        
        # Reduce workers if system is already under load
        if cpu_percent > 80:
            adjusted_workers = max(self.MIN_WORKERS, initial_workers // 2)
            logger.info(f"High CPU usage detected ({cpu_percent:.1f}%), "
                       f"reducing workers from {initial_workers} to {adjusted_workers}")
        elif load_avg > self.system_info['cpu_count'] * 1.5:
            adjusted_workers = max(self.MIN_WORKERS, int(initial_workers * 0.75))
            logger.info(f"High system load detected ({load_avg:.2f}), "
                       f"reducing workers from {initial_workers} to {adjusted_workers}")
        
        return adjusted_workers
    
    def get_optimization_profile(self, total_epochs: int, data_size_mb: float = None) -> Dict:
        """
        Get complete optimization profile with all recommendations.
        
        :param total_epochs: Total number of epochs
        :param data_size_mb: Estimated data size in MB
        :return: Dictionary with optimization settings
        """
        # Calculate optimal settings
        workers = self.get_optimal_workers(data_size_mb)
        workers_adjusted = self.adjust_for_current_load(workers)
        batch_size = self.get_optimal_batch_size(total_epochs, workers_adjusted)
        use_shared_memory = self.should_use_shared_memory(data_size_mb) if data_size_mb else False
        memory_status = self.get_memory_status()
        
        profile = {
            'workers': workers_adjusted,
            'batch_size': batch_size,
            'use_shared_memory': use_shared_memory,
            'memory_status': memory_status['status'],
            'recommendations': memory_status['recommendations'],
            'estimated_memory_per_worker_mb': (data_size_mb or 0) + self.MIN_MEMORY_PER_WORKER_MB,
            'total_memory_required_mb': workers_adjusted * ((data_size_mb or 0) + self.MIN_MEMORY_PER_WORKER_MB),
        }
        
        # Add warnings if needed
        if profile['total_memory_required_mb'] > self.system_info['memory_available_mb']:
            profile['recommendations'].append(
                f"Total memory required ({profile['total_memory_required_mb']:.0f}MB) "
                f"exceeds available memory ({self.system_info['memory_available_mb']:.0f}MB)"
            )
        
        return profile


def optimize_hyperopt_resources(config: Dict, total_epochs: int, data_size_mb: float = None) -> Dict:
    """
    Convenience function to get optimized hyperopt settings.
    
    :param config: Configuration dictionary
    :param total_epochs: Total number of epochs
    :param data_size_mb: Estimated data size in MB
    :return: Optimized settings dictionary
    """
    optimizer = HyperoptResourceOptimizer(config)
    return optimizer.get_optimization_profile(total_epochs, data_size_mb)
