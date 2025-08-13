# TIQ Complete Strategy Setup Guide

## 🎯 Quick Reference: What You MUST Change for Each Strategy

### For EACH new strategy/bot, you need:

1. **Unique Config File**: `config_strategyname.json`
2. **Unique Values in Config**:
   - `bot_name`: Different name
   - `strategy`: Your strategy file name
   - `db_url`: Different database file
   - `listen_port`: Different port (8080, 8081, 8082...)
   - `jwt_secret_key`: Different random string
   - `ws_token`: Different random string

3. **Strategy File**: `user_data/strategies/YourStrategy.py`

---

## 📁 File Structure You Need

```
freqtrade/
└── user_data/
    ├── strategies/
    │   ├── AlwaysTrade.py      (Strategy 1)
    │   ├── RSIStrategy.py       (Strategy 2)
    │   └── MACDStrategy.py      (Strategy 3)
    ├── config_alwaystrade.json  (Config for Strategy 1)
    ├── config_rsi.json          (Config for Strategy 2)
    ├── config_macd.json         (Config for Strategy 3)
    └── data/                    (Historical data - shared)
```

---

## 🚀 Step-by-Step Setup for New Strategy

### Step 1: Create Your Strategy File
Create: `user_data/strategies/YourStrategyName.py`

```python
from freqtrade.strategy import IStrategy
from pandas import DataFrame

class YourStrategyName(IStrategy):
    # REQUIRED: Set these based on your strategy
    can_short = True  # Enable shorting for futures
    
    # Stop loss (negative value)
    stoploss = -0.01  # 1% stop loss
    
    # Take profit
    minimal_roi = {
        "0": 0.02  # 2% take profit
    }
    
    # Timeframe
    timeframe = '5m'
    
    # Your strategy logic here
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Add your indicators
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Define entry conditions
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Define exit conditions
        return dataframe
```

### Step 2: Create Config File
Copy the master template and modify:

```bash
# Copy template
cp MASTER_CONFIG_TEMPLATE.json config_yourstrategy.json

# Edit and change these fields:
# 1. bot_name: "TIQ_YourStrategy"
# 2. strategy: "YourStrategyName"
# 3. db_url: "sqlite:///tradesv3_yourstrategy.sqlite"
# 4. listen_port: 8081 (or next available)
# 5. jwt_secret_key: Generate with: openssl rand -hex 32
# 6. ws_token: Generate with: openssl rand -hex 16
# 7. password: Your secure password

# Remove all "//" comment lines!
```

### Step 3: Clean Config (Remove Comments)
Use this Python script to clean comments:

```python
import json

# Read the template
with open('config_yourstrategy.json', 'r') as f:
    lines = f.readlines()

# Remove comment lines
clean_lines = [line for line in lines if not line.strip().startswith('"//"')]

# Save clean version
with open('config_yourstrategy_clean.json', 'w') as f:
    f.writelines(clean_lines)
```

---

## 🧪 Testing Configurations

### For Backtesting
```bash
# Download historical data first
freqtrade download-data \
    --userdir user_data \
    --config config_yourstrategy.json \
    --days 60 \
    --timeframe 5m

# Run backtest
freqtrade backtesting \
    --userdir user_data \
    --config config_yourstrategy.json \
    --strategy YourStrategyName \
    --timerange 20240101-20241231
```

### For Dry Run (Paper Trading)
```json
{
    "dry_run": true,
    "dry_run_wallet": 10000
}
```

### For Live Trading
```json
{
    "dry_run": false,
    "exchange": {
        "key": "YOUR_ACTUAL_API_KEY",
        "secret": "YOUR_ACTUAL_SECRET_KEY"
    }
}
```

---

## 🏃 Running Multiple Strategies

### Option 1: Multiple Terminals
```bash
# Terminal 1
freqtrade trade --userdir user_data --config config_strategy1.json

# Terminal 2  
freqtrade trade --userdir user_data --config config_strategy2.json

# Terminal 3
freqtrade trade --userdir user_data --config config_strategy3.json
```

### Option 2: Background with Logging
```bash
# Start all strategies
nohup freqtrade trade --userdir user_data --config config_strategy1.json > strategy1.log 2>&1 &
nohup freqtrade trade --userdir user_data --config config_strategy2.json > strategy2.log 2>&1 &
nohup freqtrade trade --userdir user_data --config config_strategy3.json > strategy3.log 2>&1 &

# Check logs
tail -f strategy1.log
tail -f strategy2.log

# Stop a specific bot
ps aux | grep config_strategy1
kill <PID>
```

### Option 3: Screen Sessions (Recommended)
```bash
# Create screen for each strategy
screen -S strategy1
freqtrade trade --userdir user_data --config config_strategy1.json
# Press Ctrl+A, then D to detach

screen -S strategy2
freqtrade trade --userdir user_data --config config_strategy2.json
# Press Ctrl+A, then D to detach

# List screens
screen -ls

# Reattach to screen
screen -r strategy1

# Kill screen
screen -X -S strategy1 quit
```

---

## 🌐 Web UI Access

After starting all bots:

1. **Main Bot**: http://localhost:8080
2. **Second Bot**: http://localhost:8081
3. **Third Bot**: http://localhost:8082

To add multiple bots to UI:
1. Login to first bot
2. Click "+" or "Add Bot" 
3. Enter: `http://localhost:8081`
4. Login with that bot's credentials
5. Repeat for all bots

---

## 📊 Quick Config Examples

### Conservative Bot Config
```json
{
    "bot_name": "TIQ_Conservative",
    "strategy": "ConservativeStrategy",
    "db_url": "sqlite:///tradesv3_conservative.sqlite",
    "listen_port": 8081,
    "stake_amount": 50,
    "max_open_trades": 3,
    "tradable_balance_ratio": 0.5
}
```

### Aggressive Bot Config
```json
{
    "bot_name": "TIQ_Aggressive",
    "strategy": "AggressiveStrategy",
    "db_url": "sqlite:///tradesv3_aggressive.sqlite",
    "listen_port": 8082,
    "stake_amount": 200,
    "max_open_trades": 20,
    "tradable_balance_ratio": 0.95
}
```

### Scalping Bot Config
```json
{
    "bot_name": "TIQ_Scalper",
    "strategy": "ScalpingStrategy",
    "db_url": "sqlite:///tradesv3_scalper.sqlite",
    "listen_port": 8083,
    "timeframe": "1m",
    "stake_amount": 100,
    "max_open_trades": 15
}
```

---

## ⚠️ Important Notes

### What You DON'T Need to Change:
- Exchange settings (always Binance Futures)
- Whitelist (same for all strategies)
- CCXT configurations
- Price settings
- Pairlist configuration

### What You MIGHT Want to Change:
- `stake_amount`: Trade size
- `max_open_trades`: Number of simultaneous trades
- `timeframe`: Based on strategy (1m, 5m, 15m, etc.)
- `tradable_balance_ratio`: How much of wallet to use

### Files Outside config.json:
1. **Strategy File**: `user_data/strategies/YourStrategy.py`
   - Contains actual trading logic
   - Must match "strategy" in config

2. **Database**: Auto-created, no manual setup needed
   - Location defined in `db_url`

3. **Logs**: Auto-created when running
   - `user_data/logs/freqtrade.log`

---

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Find what's using port 8080
lsof -i :8080
# Kill the process
kill -9 <PID>
```

### Strategy Not Found
- Check file name matches config
- Check file is in `user_data/strategies/`
- Check class name matches file name

### Database Locked
- Each bot needs unique database
- Check `db_url` is different for each config

### API Connection Failed
- Check port is unique
- Check firewall allows connection
- Verify username/password

---

## 📝 Config Validation

Before running, validate your config:
```bash
freqtrade test-config --userdir user_data --config config_yourstrategy.json
```

---

## 🎯 Complete Example: Setting Up RSI Strategy

1. **Create Strategy** (`user_data/strategies/RSIStrategy.py`):
```python
class RSIStrategy(IStrategy):
    can_short = True
    stoploss = -0.01
    minimal_roi = {"0": 0.015}
    timeframe = '5m'
    # ... strategy logic
```

2. **Create Config** (`config_rsi.json`):
```json
{
    "bot_name": "TIQ_RSI",
    "strategy": "RSIStrategy",
    "db_url": "sqlite:///tradesv3_rsi.sqlite",
    "api_server": {
        "listen_port": 8081
    }
    // ... rest of config
}
```

3. **Test**:
```bash
freqtrade backtesting --userdir user_data --config config_rsi.json
```

4. **Run**:
```bash
freqtrade trade --userdir user_data --config config_rsi.json
```

5. **Access**: http://localhost:8081

---

## 🚦 Pre-Launch Checklist

Before starting each bot:
- [ ] Strategy file exists in `user_data/strategies/`
- [ ] Config file has unique `bot_name`
- [ ] Config file has unique `db_url`
- [ ] Config file has unique `listen_port`
- [ ] Config file has unique `jwt_secret_key`
- [ ] Config file has unique `ws_token`
- [ ] All comment lines removed from config
- [ ] Config validated with `test-config`
- [ ] Historical data downloaded (for backtesting)
- [ ] Dry run tested first

---

## 💡 Pro Tips

1. **Always test in dry_run first!**
2. **Start with small stake_amount**
3. **Monitor logs**: `tail -f user_data/logs/freqtrade.log`
4. **Use different Telegram bots for each strategy** (optional)
5. **Backup your configs regularly**
6. **Document your strategy parameters**

---

## 📞 Need Help?

Check:
1. Logs: `user_data/logs/freqtrade.log`
2. Config validation: `freqtrade test-config`
3. Strategy syntax: `freqtrade backtesting --strategy YourStrategy`
4. Web UI: Clear browser cache if issues