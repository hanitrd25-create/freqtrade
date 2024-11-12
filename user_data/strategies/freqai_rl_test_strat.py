import logging
from functools import reduce

import talib.abstract as ta
from pandas import DataFrame
import numpy as np
from freqtrade.strategy import IStrategy


#添加的库
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd 

logger = logging.getLogger(__name__)


class freqai_rl_test_strat(IStrategy):
    """
    Test strategy - used for testing freqAI functionalities.
    DO not use in production.
    """

    # ROI table:
    minimal_roi = {
        "0": 0.041,
        "10": 0.022,
        "32": 0.007,
        "39": 0
    }


    process_only_new_candles = True
    stoploss = -0.024
    use_exit_signal = True
    startup_candle_count: int = 300
    can_short = False



    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ):

        dataframe['ma'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=2)
        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = ta.MACD(dataframe['close'], slowperiod=12,
                                                                                    fastperiod=26)
        dataframe['momentum'] = ta.MOM(dataframe, timeperiod=4)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=10)
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['stoch'] = ta.STOCH(dataframe)['slowk']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['ma_100'] = ta.SMA(dataframe, timeperiod=100)

        # Step 1: Normalize Indicators:
        # Why? Normalizing the indicators will make them comparable and allow us to assign weights to them.
        # How? We will calculate the z-score of each indicator by subtracting the rolling mean and dividing by the
        # rolling standard deviation. This will give us a normalized value that is centered around 0 with a standard
        # deviation of 1.
        dataframe['%-normalized_stoch'] = (dataframe['stoch'] - dataframe['stoch'].rolling(window=14).mean()) / dataframe['stoch'].rolling(window=14).std()
        dataframe['%-normalized_atr'] = (dataframe['atr'] - dataframe['atr'].rolling(window=14).mean()) / dataframe['atr'].rolling(window=14).std()
        dataframe['%-normalized_obv'] = (dataframe['obv'] - dataframe['obv'].rolling(window=14).mean()) / dataframe['obv'].rolling(window=14).std()
        dataframe['%-normalized_ma'] = (dataframe['close'] - dataframe['close'].rolling(window=10).mean()) / dataframe['close'].rolling(window=10).std()
        dataframe['%-normalized_macd'] = (dataframe['macd'] - dataframe['macd'].rolling(window=26).mean()) / dataframe['macd'].rolling(window=26).std()
        dataframe['%-normalized_roc'] = (dataframe['roc'] - dataframe['roc'].rolling(window=2).mean()) / dataframe['roc'].rolling(window=2).std()
        dataframe['%-normalized_momentum'] = (dataframe['momentum'] - dataframe['momentum'].rolling(window=4).mean()) / \
                                           dataframe['momentum'].rolling(window=4).std()
        dataframe['%-normalized_rsi'] = (dataframe['rsi'] - dataframe['rsi'].rolling(window=10).mean()) / dataframe['rsi'].rolling(window=10).std()
        dataframe['%-normalized_bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(
            window=20).mean() / (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(window=20).std()
        dataframe['%-normalized_cci'] = (dataframe['cci'] - dataframe['cci'].rolling(window=20).mean()) / dataframe['cci'].rolling(window=20).std()
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs):
        dataframe["%-pct-change"] = (dataframe["close"].pct_change() - dataframe["close"].pct_change().rolling(window=20).mean()) / dataframe["close"].pct_change().rolling(window=20).std()

        dataframe["%-raw_volume"] = (dataframe["volume"] - dataframe["volume"].rolling(window=20).mean()) / dataframe["volume"].rolling(window=20).std()

        dataframe["%-raw_price"] = (dataframe["close"] - dataframe["close"].rolling(window=20).mean()) / dataframe["close"].rolling(window=20).std()
       

        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs):
        dataframe['date'] = pd.to_datetime(dataframe['date'])
        dataframe["day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["hour_of_day"] = dataframe["date"].dt.hour

        dataframe["%-normalized_day_of_week"] = (dataframe["day_of_week"] - dataframe["day_of_week"].rolling(window=20).mean()) / dataframe["day_of_week"].rolling(window=20).std()
        dataframe["%-normalized_hour_of_day"] = (dataframe["hour_of_day"] - dataframe["hour_of_day"].rolling(window=20).mean()) / dataframe["hour_of_day"].rolling(window=20).std()

        dataframe["%-normalized_close"] = (dataframe["close"] - dataframe["close"].rolling(window=20).mean()) / dataframe["close"].rolling(window=20).std()
        dataframe["%-normalized_open"] = (dataframe["open"] - dataframe["open"].rolling(window=20).mean()) / dataframe["open"].rolling(window=20).std()
        dataframe["%-normalized_high"] = (dataframe["high"] - dataframe["high"].rolling(window=20).mean()) / dataframe["high"].rolling(window=20).std()
        dataframe["%-normalized_low"] = (dataframe["low"] - dataframe["low"].rolling(window=20).mean()) / dataframe["low"].rolling(window=20).std()

        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs):
        dataframe["&-action"] = 0

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # 检查这两列的是否都为1
        enter_long_conditions = [df["do_predict"] == 1, df["&-action"] == 1]


        # 如果都为1，则将enter_long和enter_tag列设置为1和long
        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        # 检查这两列是否为1,3
        enter_short_conditions = [df["do_predict"] == 1, df["&-action"] == 3]


        # 如果都为1,3，则将enter_short和enter_tag列设置为1和short
        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # 检查这两列是否为1,2
        exit_long_conditions = [df["do_predict"] == 1, df["&-action"] == 2]
        # 如果都为1,2，则将exit_long列设置为1
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        # 检查这两列是否为1,4
        exit_short_conditions = [df["do_predict"] == 1, df["&-action"] == 4]
        # 如果都为1,4，则将exit_short列设置为1
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df
