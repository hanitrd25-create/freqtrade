"""
Vectorized backtesting for simple strategies.
Provides fast backtesting for strategies without complex state management.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.constants import DATETIME_PRINT_FORMAT, Config
from freqtrade.enums import ExitType, TradingMode
from freqtrade.persistence import LocalTrade

logger = logging.getLogger(__name__)


class VectorizedBacktester:
    """
    Vectorized backtesting engine for simple strategies.
    
    Requirements for vectorized processing:
    - No position stacking
    - Simple entry/exit signals (no custom exit logic)
    - No trailing stops or complex order management
    - Fixed stake amount
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.stake_amount = config.get('stake_amount', 0.001)
        self.fee = config.get('fee', 0.001)
        self.max_open_trades = config.get('max_open_trades', 1)
        self.minimal_roi = config.get('minimal_roi', {})
        self.stoploss = config.get('stoploss', -1.0)
        self.trailing_stop = config.get('trailing_stop', False)
        self.trading_mode = config.get('trading_mode', TradingMode.SPOT)
        
    def can_use_vectorized(self, strategy) -> bool:
        """
        Check if a strategy can use vectorized backtesting.
        
        :param strategy: Strategy instance to check
        :return: True if strategy is compatible with vectorized processing
        """
        # Check for incompatible features
        if self.trailing_stop:
            return False
        if hasattr(strategy, 'custom_exit') and callable(strategy.custom_exit):
            return False
        if hasattr(strategy, 'custom_stoploss') and callable(strategy.custom_stoploss):
            return False
        if self.config.get('position_stacking', False):
            return False
        if self.trading_mode != TradingMode.SPOT:
            return False
            
        return True
    
    def vectorized_backtest(
        self,
        processed: Dict[str, DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> DataFrame:
        """
        Run vectorized backtesting on processed data.
        
        :param processed: Dictionary of processed DataFrames with signals
        :param start_date: Start date for backtesting
        :param end_date: End date for backtesting
        :return: DataFrame with trade results
        """
        all_trades = []
        
        for pair, df in processed.items():
            # Filter to backtest period
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            df_period = df[mask].copy()
            
            if df_period.empty:
                continue
                
            # Process trades for this pair
            trades = self._process_pair_vectorized(pair, df_period)
            all_trades.extend(trades)
        
        # Convert trades to DataFrame
        if all_trades:
            return pd.DataFrame(all_trades)
        else:
            return pd.DataFrame()
    
    def _process_pair_vectorized(self, pair: str, df: DataFrame) -> List[dict]:
        """
        Process all trades for a pair using vectorized operations.
        
        :param pair: Trading pair
        :param df: DataFrame with OHLCV and signals
        :return: List of trade dictionaries
        """
        trades = []
        
        # Get entry and exit signals as numpy arrays
        enter_long = df['enter_long'].values if 'enter_long' in df else np.zeros(len(df), dtype=bool)
        exit_long = df['exit_long'].values if 'exit_long' in df else np.zeros(len(df), dtype=bool)
        
        # Get price data
        dates = df['date'].values
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        
        # Vectorized trade processing
        in_position = False
        position_idx = -1
        
        # Find all entry points
        entry_points = np.where(enter_long)[0]
        
        for i in entry_points:
            if in_position:
                continue
                
            # Open trade
            entry_date = dates[i]
            entry_price = open_prices[i + 1] if i + 1 < len(open_prices) else open_prices[i]
            
            # Find exit point
            exit_idx = self._find_exit_vectorized(
                i, exit_long, high_prices, low_prices, close_prices, entry_price
            )
            
            if exit_idx is not None and exit_idx < len(dates):
                # Close trade
                exit_date = dates[exit_idx]
                exit_price = self._calculate_exit_price(
                    exit_idx, exit_long[exit_idx] if exit_idx < len(exit_long) else False,
                    open_prices, high_prices, low_prices, close_prices, entry_price
                )
                
                # Calculate profit
                profit_ratio = (exit_price / entry_price - 1) - (2 * self.fee)
                profit_abs = self.stake_amount * profit_ratio
                
                # Create trade record
                trade = {
                    'pair': pair,
                    'stake_amount': self.stake_amount,
                    'amount': self.stake_amount / entry_price,
                    'open_date': pd.Timestamp(entry_date),
                    'close_date': pd.Timestamp(exit_date),
                    'open_rate': entry_price,
                    'close_rate': exit_price,
                    'fee_open': self.fee,
                    'fee_close': self.fee,
                    'profit_ratio': profit_ratio,
                    'profit_abs': profit_abs,
                    'exit_reason': self._determine_exit_reason(
                        exit_idx, exit_long[exit_idx] if exit_idx < len(exit_long) else False,
                        high_prices[exit_idx], low_prices[exit_idx], entry_price
                    ),
                    'trade_duration': int((pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).total_seconds() / 60),
                    'is_open': False,
                    'is_short': False,
                }
                trades.append(trade)
                
                in_position = False
                position_idx = -1
        
        return trades
    
    def _find_exit_vectorized(
        self,
        entry_idx: int,
        exit_signals: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        entry_price: float
    ) -> Optional[int]:
        """
        Find the exit point for a trade using vectorized operations.
        
        :param entry_idx: Index of entry point
        :param exit_signals: Array of exit signals
        :param high_prices: Array of high prices
        :param low_prices: Array of low prices
        :param close_prices: Array of close prices
        :param entry_price: Entry price for the trade
        :return: Index of exit point or None
        """
        # Start looking from the next candle after entry
        start_idx = entry_idx + 1
        if start_idx >= len(close_prices):
            return None
        
        # Calculate stoploss price
        stoploss_price = entry_price * (1 + self.stoploss)
        
        # Find stoploss hits
        stoploss_hit = low_prices[start_idx:] <= stoploss_price
        
        # Find signal exits
        signal_exits = exit_signals[start_idx:]
        
        # Find ROI exits (simplified - check if profit exceeds minimum ROI)
        if self.minimal_roi:
            min_roi = min(self.minimal_roi.values())
            roi_price = entry_price * (1 + min_roi)
            roi_hit = high_prices[start_idx:] >= roi_price
        else:
            roi_hit = np.zeros(len(signal_exits), dtype=bool)
        
        # Find first exit (any condition)
        any_exit = stoploss_hit | signal_exits | roi_hit
        
        if any_exit.any():
            # Return the index of the first exit
            exit_offset = np.argmax(any_exit)
            return start_idx + exit_offset
        
        # If no exit found, close at the last candle
        return len(close_prices) - 1
    
    def _calculate_exit_price(
        self,
        exit_idx: int,
        is_signal_exit: bool,
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        entry_price: float
    ) -> float:
        """
        Calculate the exit price based on exit conditions.
        
        :param exit_idx: Index of exit candle
        :param is_signal_exit: Whether exit is due to signal
        :param open_prices: Array of open prices
        :param high_prices: Array of high prices
        :param low_prices: Array of low prices  
        :param close_prices: Array of close prices
        :param entry_price: Entry price for the trade
        :return: Exit price
        """
        if exit_idx >= len(close_prices):
            return close_prices[-1]
        
        # Check for stoploss hit
        stoploss_price = entry_price * (1 + self.stoploss)
        if low_prices[exit_idx] <= stoploss_price:
            # Stoploss hit - use the maximum of stoploss price or low price (gap scenarios)
            return max(stoploss_price, low_prices[exit_idx])
        
        # Check for ROI hit
        if self.minimal_roi:
            min_roi = min(self.minimal_roi.values())
            roi_price = entry_price * (1 + min_roi)
            if high_prices[exit_idx] >= roi_price:
                # ROI hit - use the minimum of ROI price or high price (gap scenarios)
                return min(roi_price, high_prices[exit_idx])
        
        # Signal exit or end of data - use open price of next candle
        if is_signal_exit and exit_idx + 1 < len(open_prices):
            return open_prices[exit_idx + 1]
        
        return close_prices[exit_idx]
    
    def _determine_exit_reason(
        self,
        exit_idx: int,
        is_signal_exit: bool,
        high_price: float,
        low_price: float,
        entry_price: float
    ) -> str:
        """
        Determine the reason for trade exit.
        
        :param exit_idx: Index of exit candle
        :param is_signal_exit: Whether exit is due to signal
        :param high_price: High price of exit candle
        :param low_price: Low price of exit candle
        :param entry_price: Entry price for the trade
        :return: Exit reason string
        """
        # Check for stoploss hit
        stoploss_price = entry_price * (1 + self.stoploss)
        if low_price <= stoploss_price:
            return ExitType.STOP_LOSS.value
        
        # Check for ROI hit
        if self.minimal_roi:
            min_roi = min(self.minimal_roi.values())
            roi_price = entry_price * (1 + min_roi)
            if high_price >= roi_price:
                return ExitType.ROI.value
        
        # Signal exit
        if is_signal_exit:
            return ExitType.EXIT_SIGNAL.value
        
        # Force exit (end of data)
        return ExitType.FORCE_EXIT.value
