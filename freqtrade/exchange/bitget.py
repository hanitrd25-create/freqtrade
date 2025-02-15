import logging
from datetime import datetime

import ccxt

from freqtrade.constants import BuySell
from freqtrade.enums import CandleType, MarginMode, PriceType, TradingMode
from freqtrade.exceptions import DDosProtection, ExchangeError, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Bitget(Exchange):
    """
    Bitget exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 1000,
        "ohlcv_has_history": True,
        "order_time_in_force": ["GTC", "FOK", "IOC"],
        "trades_has_history": True,
    }
    _ft_has_futures: FtHas = {
        "ohlcv_has_history": True,
        "mark_ohlcv_timeframe": "4h",
        "funding_fee_timeframe": "8h",
        "stoploss_on_exchange": True,
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        "stop_price_prop": "stopPrice",
        "stop_price_type_field": "triggerBy",
        "stop_price_type_value_mapping": {
            PriceType.LAST: "last_price",
            PriceType.MARK: "mark_price",
            PriceType.INDEX: "index_price",
        },
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.FUTURES, MarginMode.ISOLATED)
    ]

    @retrier
    def additional_exchange_init(self) -> None:
        try:
            if not self._config["dry_run"]:
                if self.trading_mode == TradingMode.FUTURES:
                    # Set position mode to one-way (hedged = False)
                    self._api.set_position_mode(False, None, {"productType": "USDT-FUTURES"})
                    logger.info("Bitget: Position mode set to one-way.")

        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"Error in additional_exchange_init due to {e.__class__.__name__}. Message: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _get_params(
        self,
        side: BuySell,
        ordertype: str,
        leverage: float,
        reduceOnly: bool,
        time_in_force: str = "GTC",
    ) -> dict:
        params = super()._get_params(
            side=side,
            ordertype=ordertype,
            leverage=leverage,
            reduceOnly=reduceOnly,
            time_in_force=time_in_force,
        )
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode:
            params["marginMode"] = self.margin_mode.lower()
        return params

    def ohlcv_candle_limit(
        self, timeframe: str, candle_type: CandleType, since_ms: int | None = None
    ) -> int:
        if candle_type == CandleType.FUNDING_RATE:
            return 100  # Bitget seems to limit funding rate data to 100 entries
        if candle_type == CandleType.MARK:
            return 200
        # For other candle types, use the default or previously defined limit
        return super().ohlcv_candle_limit(timeframe, candle_type, since_ms)

    def get_funding_fees(
        self, pair: str, amount: float, is_short: bool, open_date: datetime
    ) -> float:
        if self.trading_mode == TradingMode.FUTURES:
            try:
                return self._fetch_and_calculate_funding_fees(pair, amount, is_short, open_date)
            except ExchangeError:
                logger.warning(f"Could not update funding fees for {pair}.")
        return 0.0

    def get_max_pair_stake_amount(self, pair: str, price: float, leverage: float = 1.0) -> float:
        if self.trading_mode == TradingMode.SPOT:
            return float("inf")  # Not actually inf, but this probably won't matter for SPOT

        if pair not in self._leverage_tiers:
            return 1.0

        pair_tiers = self._leverage_tiers[pair]
        return pair_tiers[-1]["maxNotional"] / leverage
