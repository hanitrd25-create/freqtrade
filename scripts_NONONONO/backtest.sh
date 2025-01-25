
FHOME="/mnt/d/develop/freqtrade"
cd $FHOME
source .venv/bin/activate


freqtrade backtesting \
    --config $FHOME/user_data/config.json\
    --dry-run-wallet 10000 \
    --strategy-list FibonacciRetracementStrategy \
    --strategy-path $FHOME/user_data/strategies/copied_strategies/berlinguyinca \
    --timeframe 1w \
    --timerange 20180101-

    # --config $FHOME/user_data/config_spot.json\

# freqtrade backtesting \
#     --dry-run-wallet 10000 \
#     --strategy-list Simple AverageStrategy MACDStrategy ReinforcedQuickie \
#     --strategy-path ./user_data/strategies/copied_strategies/berlinguyinca \
#     --timeframe 5m \
#     --timerange 20180101-20180301

# /mnt/d/develop/freqtrade/user_data/strategies/copied_strategies/Strategy001.py

# freqtrade backtesting --strategy Strategy001 --strategy-path /mnt/d/develop/freqtrade/user_data/strategies/copied_strategies
