"""
SharpeHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""

from datetime import datetime

from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.data.metrics import calculate_account_value_history, calculate_sharpe
from freqtrade.optimize.hyperopt import IHyperOptLoss


class SharpeHyperOptLoss(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation uses the Sharpe Ratio calculation.
    """

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Config,
        processed: dict[str, DataFrame],
        *args,
        **kwargs,
    ) -> float:
        """
        Objective function, returns smaller number for more optimal results.

        Uses Sharpe Ratio calculation.
        """
        start_balance = config["dry_run_wallet"]
        timeframe = config["timeframe"]
        stake_currency = config["stake_currency"]
        pairlist = config["pairs"]
        account_value_history = calculate_account_value_history(
            processed,
            results,
            min_date,
            max_date,
            timeframe,
            stake_currency,
            start_balance,
            pairlist,
        )
        sharp_ratio = calculate_sharpe(account_value_history, timeframe)
        # print(expected_returns_mean, up_stdev, sharp_ratio)
        return -sharp_ratio
