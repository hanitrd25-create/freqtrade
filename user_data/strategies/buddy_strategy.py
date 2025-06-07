# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

"""
Example strategy with weighted indicators for GitHub CI/CD testing
"""
import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    informative,
    DecimalParameter,
    IntParameter,
    CategoricalParameter
)


class BuddyStrategy(IStrategy):
    """
    Example strategy with weighted indicators
    
    You can:
    - Add new indicators in populate_indicators()
    - Modify buy/sell conditions with weights
    - Optimize parameters using hyperopt
    """
    
    # Strategy interface version
    INTERFACE_VERSION = 3
    
    # Optimal timeframe for the strategy
    timeframe = '5m'
    
    # Can this strategy go short?
    can_short: bool = False
    
    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.10,
        "40": 0.05,
        "80": 0.025,
        "120": 0
    }
    
    # Optimal stoploss
    stoploss = -0.10
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30
    
    # Strategy parameters - these can be optimized
    buy_rsi = IntParameter(20, 40, default=30, space='buy')
    buy_rsi_weight = DecimalParameter(0.1, 1.0, default=0.4, space='buy')
    
    buy_macd_weight = DecimalParameter(0.1, 1.0, default=0.6, space='buy')
    
    sell_rsi = IntParameter(60, 80, default=70, space='sell')
    sell_rsi_weight = DecimalParameter(0.1, 1.0, default=0.5, space='sell')
    
    sell_bb_weight = DecimalParameter(0.1, 1.0, default=0.5, space='sell')
    
    # Threshold for weighted signals
    buy_threshold = DecimalParameter(0.5, 1.0, default=0.7, space='buy')
    sell_threshold = DecimalParameter(0.5, 1.0, default=0.7, space='sell')
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_upperband'] = bollinger['upperband']
        
        # EMA
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        
        # Volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        """
        # Calculate weighted buy signal
        dataframe['buy_signal_strength'] = (
            # RSI signal with weight
            ((dataframe['rsi'] < self.buy_rsi.value) * self.buy_rsi_weight.value) +
            
            # MACD signal with weight
            ((dataframe['macd'] > dataframe['macdsignal']) * self.buy_macd_weight.value)
        )
        
        # Normalize weights
        total_weight = self.buy_rsi_weight.value + self.buy_macd_weight.value
        dataframe['buy_signal_strength'] = dataframe['buy_signal_strength'] / total_weight
        
        # Generate buy signal when weighted score exceeds threshold
        dataframe.loc[
            (
                (dataframe['buy_signal_strength'] > self.buy_threshold.value) &
                (dataframe['volume'] > dataframe['volume_mean'])  # Volume confirmation
            ),
            'enter_long'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        """
        # Calculate weighted sell signal
        dataframe['sell_signal_strength'] = (
            # RSI signal with weight
            ((dataframe['rsi'] > self.sell_rsi.value) * self.sell_rsi_weight.value) +
            
            # Bollinger Band signal with weight
            ((dataframe['close'] > dataframe['bb_upperband']) * self.sell_bb_weight.value)
        )
        
        # Normalize weights
        total_weight = self.sell_rsi_weight.value + self.sell_bb_weight.value
        dataframe['sell_signal_strength'] = dataframe['sell_signal_strength'] / total_weight
        
        # Generate sell signal when weighted score exceeds threshold
        dataframe.loc[
            (
                (dataframe['sell_signal_strength'] > self.sell_threshold.value) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1
        
        return dataframe