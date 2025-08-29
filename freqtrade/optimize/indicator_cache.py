"""
Indicator caching module for performance optimization.

This module provides memoization for technical indicators to avoid
recalculating the same indicators multiple times during backtesting
and hyperopt runs.
"""

import hashlib
import logging
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from functools import lru_cache, wraps
from threading import Lock

logger = logging.getLogger(__name__)


class IndicatorCache:
    """
    Cache for technical indicator calculations.
    
    This class provides thread-safe caching of indicator results
    to avoid redundant calculations during backtesting and hyperopt.
    """
    
    def __init__(self, max_cache_size: int = 1000):
        """
        Initialize the indicator cache.
        
        :param max_cache_size: Maximum number of cached results to store
        """
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = max_cache_size
        self._lock = Lock()
        logger.info(f"Indicator cache initialized with max size: {max_cache_size}")
    
    def _generate_cache_key(self, 
                           dataframe: pd.DataFrame,
                           indicator_name: str,
                           params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for an indicator calculation.
        
        :param dataframe: Input dataframe
        :param indicator_name: Name of the indicator
        :param params: Indicator parameters
        :return: Unique cache key
        """
        # Create a hash of the dataframe shape, first/last values, and indicator params
        df_hash_components = [
            str(dataframe.shape),
            str(dataframe.index[0]) if len(dataframe) > 0 else "",
            str(dataframe.index[-1]) if len(dataframe) > 0 else "",
            str(dataframe.columns.tolist()),
        ]
        
        # Add sample values from key columns if they exist
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in dataframe.columns:
                # Use first, middle, and last values for better uniqueness
                first_val = dataframe[col].iloc[0] if len(dataframe) > 0 else 0
                mid_val = dataframe[col].iloc[len(dataframe)//2] if len(dataframe) > 0 else 0
                last_val = dataframe[col].iloc[-1] if len(dataframe) > 0 else 0
                df_hash_components.append(f"{col}:{first_val:.8f}:{mid_val:.8f}:{last_val:.8f}")
        
        df_hash = hashlib.md5('|'.join(df_hash_components).encode()).hexdigest()
        
        # Create parameter hash
        param_str = '|'.join(f"{k}:{v}" for k, v in sorted(params.items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        
        return f"{indicator_name}_{df_hash}_{param_hash}"
    
    def get(self, 
            dataframe: pd.DataFrame,
            indicator_name: str,
            params: Dict[str, Any],
            calculate_fn: Callable) -> pd.DataFrame:
        """
        Get cached indicator result or calculate and cache it.
        
        :param dataframe: Input dataframe
        :param indicator_name: Name of the indicator
        :param params: Indicator parameters
        :param calculate_fn: Function to calculate the indicator if not cached
        :return: DataFrame with indicator values
        """
        cache_key = self._generate_cache_key(dataframe, indicator_name, params)
        
        with self._lock:
            if cache_key in self.cache:
                self.cache_hits += 1
                logger.debug(f"Cache hit for {indicator_name} (hits: {self.cache_hits})")
                return self.cache[cache_key].copy()
            
            self.cache_misses += 1
            logger.debug(f"Cache miss for {indicator_name} (misses: {self.cache_misses})")
        
        # Calculate indicator outside of lock
        result = calculate_fn(dataframe, **params)
        
        with self._lock:
            # Check cache size and evict if necessary
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entry (simple FIFO eviction)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                logger.debug(f"Evicted cache entry: {oldest_key}")
            
            # Store in cache
            self.cache[cache_key] = result.copy()
        
        return result
    
    def clear(self):
        """Clear all cached indicators."""
        with self._lock:
            self.cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            logger.info("Indicator cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        :return: Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
            
            return {
                'cache_size': len(self.cache),
                'max_size': self.max_cache_size,
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }


def cached_indicator(cache: Optional[IndicatorCache] = None):
    """
    Decorator for caching indicator calculations.
    
    This decorator can be applied to indicator functions to automatically
    cache their results based on input parameters.
    
    :param cache: IndicatorCache instance to use (creates new if None)
    :return: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        # Use provided cache or create a new one
        indicator_cache = cache or IndicatorCache()
        
        @wraps(func)
        def wrapper(dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
            # Extract indicator name from function
            indicator_name = func.__name__
            
            # Use cache to get or calculate result
            return indicator_cache.get(
                dataframe,
                indicator_name,
                kwargs,
                func
            )
        
        # Attach cache to wrapper for access
        wrapper.cache = indicator_cache
        return wrapper
    
    return decorator


class SignalCache:
    """
    Cache for buy/sell signal calculations.
    
    This class caches the results of strategy signal calculations
    to avoid recalculating signals for the same data and parameters.
    """
    
    def __init__(self, max_cache_size: int = 500):
        """
        Initialize the signal cache.
        
        :param max_cache_size: Maximum number of cached signal sets
        """
        self.cache: Dict[str, Tuple[pd.Series, pd.Series]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = max_cache_size
        self._lock = Lock()
        logger.info(f"Signal cache initialized with max size: {max_cache_size}")
    
    def _generate_cache_key(self,
                           dataframe: pd.DataFrame,
                           strategy_name: str,
                           params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for signal calculations.
        
        :param dataframe: Input dataframe with indicators
        :param strategy_name: Name of the strategy
        :param params: Strategy parameters
        :return: Unique cache key
        """
        # Create hash from dataframe characteristics
        df_components = [
            str(dataframe.shape),
            str(dataframe.index[0]) if len(dataframe) > 0 else "",
            str(dataframe.index[-1]) if len(dataframe) > 0 else "",
        ]
        
        # Sample key indicator columns for uniqueness
        indicator_cols = [col for col in dataframe.columns 
                         if col not in ['open', 'high', 'low', 'close', 'volume', 'date']]
        
        for col in indicator_cols[:5]:  # Sample first 5 indicators
            if col in dataframe.columns:
                first_val = dataframe[col].iloc[0] if len(dataframe) > 0 else np.nan
                last_val = dataframe[col].iloc[-1] if len(dataframe) > 0 else np.nan
                if pd.notna(first_val) and pd.notna(last_val):
                    df_components.append(f"{col}:{first_val:.8f}:{last_val:.8f}")
        
        df_hash = hashlib.md5('|'.join(df_components).encode()).hexdigest()
        
        # Create parameter hash
        param_str = '|'.join(f"{k}:{v}" for k, v in sorted(params.items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        
        return f"{strategy_name}_{df_hash}_{param_hash}"
    
    def get(self,
            dataframe: pd.DataFrame,
            strategy_name: str,
            params: Dict[str, Any],
            calculate_fn: Callable) -> Tuple[pd.Series, pd.Series]:
        """
        Get cached signals or calculate and cache them.
        
        :param dataframe: Input dataframe with indicators
        :param strategy_name: Name of the strategy
        :param params: Strategy parameters
        :param calculate_fn: Function to calculate signals if not cached
        :return: Tuple of (buy_signals, sell_signals)
        """
        cache_key = self._generate_cache_key(dataframe, strategy_name, params)
        
        with self._lock:
            if cache_key in self.cache:
                self.cache_hits += 1
                logger.debug(f"Signal cache hit for {strategy_name} (hits: {self.cache_hits})")
                buy_signals, sell_signals = self.cache[cache_key]
                return buy_signals.copy(), sell_signals.copy()
            
            self.cache_misses += 1
            logger.debug(f"Signal cache miss for {strategy_name} (misses: {self.cache_misses})")
        
        # Calculate signals outside of lock
        buy_signals, sell_signals = calculate_fn(dataframe, **params)
        
        with self._lock:
            # Check cache size and evict if necessary
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                logger.debug(f"Evicted signal cache entry: {oldest_key}")
            
            # Store in cache
            self.cache[cache_key] = (buy_signals.copy(), sell_signals.copy())
        
        return buy_signals, sell_signals
    
    def clear(self):
        """Clear all cached signals."""
        with self._lock:
            self.cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            logger.info("Signal cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        :return: Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
            
            return {
                'cache_size': len(self.cache),
                'max_size': self.max_cache_size,
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }


# Global cache instances for shared use
_global_indicator_cache = None
_global_signal_cache = None


def get_indicator_cache() -> IndicatorCache:
    """Get or create the global indicator cache instance."""
    global _global_indicator_cache
    if _global_indicator_cache is None:
        _global_indicator_cache = IndicatorCache()
    return _global_indicator_cache


def get_signal_cache() -> SignalCache:
    """Get or create the global signal cache instance."""
    global _global_signal_cache
    if _global_signal_cache is None:
        _global_signal_cache = SignalCache()
    return _global_signal_cache


def clear_all_caches():
    """Clear all global cache instances."""
    if _global_indicator_cache:
        _global_indicator_cache.clear()
    if _global_signal_cache:
        _global_signal_cache.clear()
    logger.info("All caches cleared")
