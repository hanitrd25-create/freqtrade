cd /mnt/d/develop/freqtrade
source .venv/bin/activate

freqtrade download-data \
    --exchange binance \
    --pairs BTC/USDT ETH/USDT \
    --timeframes 1m 5m 15m 30m 1h 4h 1d 1w 1M \
    --trading-mode spot \
    --timerange 20170801-20240810 \

    # --data-format-ohlcv hdf5 \
    # --data-format-trades hdf5
# freqtrade download-data --exchange binance -p BTC/USDT -t 1m 3m 5m 15m 30m 1h 4h 1d 1w 1M --timerange 20170801-
