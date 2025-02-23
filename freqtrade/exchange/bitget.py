import logging
from datetime import datetime, timezone

import ccxt

from freqtrade.constants import BuySell
from freqtrade.enums import CandleType, MarginMode, PriceType, TradingMode
from freqtrade.exceptions import DDosProtection, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange_types import FtHas, OHLCVResponse


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
        "order_time_in_force": ["GTC", "FOK", "IOC"],
        "trades_has_history": True,
        "funding_fee_candle_limit": 100,
    }
    _ft_has_futures: FtHas = {
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

    def timeframe_to_milliseconds(self, timeframe: str) -> int:
        return ccxt.Exchange.parse_timeframe(timeframe) * 1000

    async def _async_get_historic_ohlcv(
        self,
        pair: str,
        timeframe: str,
        since_ms: int,
        candle_type: CandleType,
        raise_: bool = False,
        until_ms: int | None = None,
    ) -> OHLCVResponse:
        try:
            pair_data = await super()._async_get_historic_ohlcv(
                pair, timeframe, since_ms, candle_type, raise_, until_ms
            )

            pair, _, candle_type, data, partial_candle = pair_data

            if candle_type in [CandleType.MARK, CandleType.INDEX]:
                current_time = int(datetime.now(timezone.utc).timestamp() * 1000)
                timeframe_ms = self.timeframe_to_milliseconds(timeframe)
                last_candle_time = data[-1][0] if data else 0

                if current_time - last_candle_time >= timeframe_ms:
                    newest_candle_start = current_time - (current_time % timeframe_ms)

                    params = {}
                    if candle_type == CandleType.MARK:
                        params["price"] = "mark"
                    elif candle_type == CandleType.INDEX:
                        params["price"] = "index"

                    latest_candles = await self._api_async.fetch_ohlcv(
                        pair, timeframe, newest_candle_start, 1, params
                    )

                    if latest_candles:
                        data.append(latest_candles[0])
                    else:
                        estimated_candle = await self._estimate_current_candle(
                            pair, timeframe, newest_candle_start
                        )
                        if estimated_candle:
                            data.append(estimated_candle)
                        else:
                            logger.warning(f"can't for {pair} get candle of {timeframe}")

            return pair, timeframe, candle_type, data, partial_candle

        except Exception as e:
            logger.error(
                f"params：{pair}, {timeframe}, {since_ms}, {candle_type}, "
                f"can't get_historic_ohlcv: {e}"
            )
            raise

    async def _estimate_current_candle(
        self, pair: str, timeframe: str, start_time: int
    ) -> list[float]:
        timeframe_map: dict[str, tuple[str, int]] = {
            "5m": ("1m", 5),
            "15m": ("3m", 5),
            "30m": ("5m", 6),
            "1h": ("15m", 4),
            "4h": ("1h", 4),
            "6h": ("1h", 6),
            "12h": ("3h", 4),
            "1d": ("6h", 4),
            "1w": ("1d", 7),
            "1m": ("1w", 4),
        }

        if timeframe not in timeframe_map:
            raise Exception(f"{timeframe} not in timeframe list")

        smaller_tf, max_candles = timeframe_map[timeframe]

        try:
            smaller_candles = await self._api_async.fetch_ohlcv(
                pair, smaller_tf, start_time, max_candles
            )

            if not smaller_candles:
                logger.warning(f"can't get  {smaller_tf} candle data for {pair}")
                return []
            open_price = smaller_candles[0][1]
            high_price = max(candle[2] for candle in smaller_candles)
            low_price = min(candle[3] for candle in smaller_candles)
            close_price = smaller_candles[-1][4]
            volume = sum(candle[5] for candle in smaller_candles)
            return [start_time, open_price, high_price, low_price, close_price, volume]

        except Exception as e:
            raise Exception(f"can't for {pair} get candle of {timeframe}: {e}")

    def ohlcv_candle_limit(
        self, timeframe: str, candle_type: CandleType, since_ms: int | None = None
    ) -> int:
        if candle_type == CandleType.MARK:
            return 200
        # For other candle types, use the default or previously defined limit
        return super().ohlcv_candle_limit(timeframe, candle_type, since_ms)
