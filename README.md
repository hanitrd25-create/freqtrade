# ![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade_poweredby.svg)

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/workflows/Freqtrade%20CI/badge.svg)](https://github.com/freqtrade/freqtrade/actions/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![Coverage Status](https://coveralls.io/repos/github/freqtrade/freqtrade/badge.svg?branch=develop&service=github)](https://coveralls.io/github/freqtrade/freqtrade?branch=develop)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Maintainability](https://api.codeclimate.com/v1/badges/5737e6d668200b7518ff/maintainability)](https://codeclimate.com/github/freqtrade/freqtrade/maintainability)
[![Discord](https://img.shields.io/discord/700048804539400213?logo=discord&logoColor=white)](https://discord.gg/p7nuUNVfP7)

## 🚀 Welcome to Freqtrade

**Freqtrade** is a powerful, feature-rich, and highly customizable free and open-source cryptocurrency trading bot written in Python. It empowers traders of all experience levels to automate their trading strategies across multiple exchanges with confidence and precision.

With an intuitive interface accessible via Telegram or the built-in WebUI, Freqtrade provides comprehensive tools for backtesting, data visualization, portfolio management, and strategy optimization powered by machine learning.

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## 💡 Why Freqtrade?

- **Free and Open Source**: Full transparency with no hidden fees or costs
- **Highly Customizable**: Create and fine-tune strategies to match your trading style
- **Comprehensive Backtesting**: Test your strategies against historical data before risking real capital
- **Machine Learning Integration**: Leverage AI to optimize your trading parameters
- **Active Community**: Join thousands of traders sharing strategies and insights
- **Multi-Exchange Support**: Trade on numerous popular cryptocurrency exchanges
- **Security Focused**: Your API keys remain on your machine, with no data sent to external servers

## Disclaimer

This software is for educational purposes only. Do not risk money which
you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS
AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

Always start by running a trading bot in Dry-run and do not engage money
before you understand how it works and what profit/loss you should
expect.

We strongly recommend you to have coding and Python knowledge. Do not
hesitate to read the source code and understand the mechanism of this bot.

## Supported Exchange marketplaces

Please read the [exchange specific notes](docs/exchanges.md) to learn about eventual, special configurations needed for each exchange.

- [X] [Binance](https://www.binance.com/)
- [X] [Bitmart](https://bitmart.com/)
- [X] [BingX](https://bingx.com/invite/0EM9RX)
- [X] [Bybit](https://bybit.com/)
- [X] [Gate.io](https://www.gate.io/ref/6266643)
- [X] [HTX](https://www.htx.com/)
- [X] [Hyperliquid](https://hyperliquid.xyz/) (A decentralized exchange, or DEX)
- [X] [Kraken](https://kraken.com/)
- [X] [OKX](https://okx.com/)
- [X] [MyOKX](https://okx.com/) (OKX EEA)
- [ ] [potentially many others](https://github.com/ccxt/ccxt/). _(We cannot guarantee they will work)_

### Supported Futures Exchanges (experimental)

- [X] [Binance](https://www.binance.com/)
- [X] [Gate.io](https://www.gate.io/ref/6266643)
- [X] [Hyperliquid](https://hyperliquid.xyz/) (A decentralized exchange, or DEX)
- [X] [OKX](https://okx.com/)
- [X] [Bybit](https://bybit.com/)

Please make sure to read the [exchange specific notes](docs/exchanges.md), as well as the [trading with leverage](docs/leverage.md) documentation before diving in.

### Community tested

Exchanges confirmed working by the community:

- [X] [Bitvavo](https://bitvavo.com/)
- [X] [Kucoin](https://www.kucoin.com/)

## Documentation

We invite you to read the bot documentation to ensure you understand how the bot is working.

Please find the complete documentation on the [freqtrade website](https://www.freqtrade.io).

## 🔧 Key Features

- [x] **Cross-Platform Python Foundation**: Built on Python 3.10+, ensuring compatibility across Windows, macOS, and Linux systems for maximum flexibility.

- [x] **Reliable Data Storage**: Secure and efficient data persistence through SQLite, maintaining your trading history and configuration even after restarts.

- [x] **Risk-Free Testing with Dry-Run Mode**: Safely test your strategies in real-time market conditions without risking actual capital - perfect for beginners and strategy development.

- [x] **Comprehensive Backtesting Engine**: Rigorously test your strategies against historical data to validate performance before deploying with real funds.

- [x] **AI-Powered Strategy Optimization**: Leverage machine learning to fine-tune your trading parameters based on real market data, discovering optimal configurations automatically.

- [X] **Advanced FreqAI Prediction Modeling**: Build sophisticated strategies with FreqAI that continuously adapt to changing market conditions through state-of-the-art machine learning methods. [Learn more](https://www.freqtrade.io/en/stable/freqai/)

- [x] **Smart Position Sizing with Edge**: Optimize your capital allocation by calculating win rates, risk-reward ratios, and ideal stop-loss levels for each market. [Learn more](https://www.freqtrade.io/en/stable/edge/)

- [x] **Flexible Asset Selection**: Fine-tune your trading universe with customizable whitelists and blacklists, including dynamic options that adapt to market conditions.

- [x] **Modern Web Interface**: Monitor and control your trading operations through an intuitive, built-in web UI dashboard.

- [x] **Telegram Integration**: Manage your bot remotely via Telegram messaging, allowing you to control operations from anywhere.

- [x] **Real-Time Performance Metrics**: Track your trading performance with detailed reports and profit/loss calculations in your preferred fiat currency.

## 🚀 Getting Started

### Quick Installation Options

Choose the installation method that works best for your environment:

1. **🐳 Docker (Recommended for Beginners)**
   - The fastest way to get up and running with minimal configuration
   - [Docker Quickstart Guide](https://www.freqtrade.io/en/stable/docker_quickstart/)

2. **💻 Native Installation**
   - For users who prefer direct installation on their systems
   - [Detailed Installation Instructions](https://www.freqtrade.io/en/stable/installation/)

3. **☁️ Cloud Deployment**
   - Deploy on your favorite cloud platform
   - Follow our [Cloud Deployment Guide](https://www.freqtrade.io/en/stable/advanced-setup/)

### First Steps After Installation

1. Create a configuration file: `freqtrade new-config`
2. Download historical data: `freqtrade download-data --timeframe 5m`
3. Choose a trading strategy or create your own
4. Run backtesting: `freqtrade backtesting --strategy YourStrategy`
5. Start trading in dry-run mode: `freqtrade trade --strategy YourStrategy --dry-run`

## Basic Usage

### Bot commands

```
usage: freqtrade [-h] [-V]
                 {trade,create-userdir,new-config,show-config,new-strategy,download-data,convert-data,convert-trade-data,trades-to-ohlcv,list-data,backtesting,backtesting-show,backtesting-analysis,edge,hyperopt,hyperopt-list,hyperopt-show,list-exchanges,list-markets,list-pairs,list-strategies,list-hyperoptloss,list-freqaimodels,list-timeframes,show-trades,test-pairlist,convert-db,install-ui,plot-dataframe,plot-profit,webserver,strategy-updater,lookahead-analysis,recursive-analysis}
                 ...

Free, open source crypto trading bot

positional arguments:
  {trade,create-userdir,new-config,show-config,new-strategy,download-data,convert-data,convert-trade-data,trades-to-ohlcv,list-data,backtesting,backtesting-show,backtesting-analysis,edge,hyperopt,hyperopt-list,hyperopt-show,list-exchanges,list-markets,list-pairs,list-strategies,list-hyperoptloss,list-freqaimodels,list-timeframes,show-trades,test-pairlist,convert-db,install-ui,plot-dataframe,plot-profit,webserver,strategy-updater,lookahead-analysis,recursive-analysis}
    trade               Trade module.
    create-userdir      Create user-data directory.
    new-config          Create new config
    show-config         Show resolved config
    new-strategy        Create new strategy
    download-data       Download backtesting data.
    convert-data        Convert candle (OHLCV) data from one format to
                        another.
    convert-trade-data  Convert trade data from one format to another.
    trades-to-ohlcv     Convert trade data to OHLCV data.
    list-data           List downloaded data.
    backtesting         Backtesting module.
    backtesting-show    Show past Backtest results
    backtesting-analysis
                        Backtest Analysis module.
    edge                Edge module.
    hyperopt            Hyperopt module.
    hyperopt-list       List Hyperopt results
    hyperopt-show       Show details of Hyperopt results
    list-exchanges      Print available exchanges.
    list-markets        Print markets on exchange.
    list-pairs          Print pairs on exchange.
    list-strategies     Print available strategies.
    list-hyperoptloss   Print available hyperopt loss functions.
    list-freqaimodels   Print available freqAI models.
    list-timeframes     Print available timeframes for the exchange.
    show-trades         Show trades.
    test-pairlist       Test your pairlist configuration.
    convert-db          Migrate database to different system
    install-ui          Install FreqUI
    plot-dataframe      Plot candles with indicators.
    plot-profit         Generate plot showing profits.
    webserver           Webserver module.
    strategy-updater    updates outdated strategy files to the current version
    lookahead-analysis  Check for potential look ahead bias.
    recursive-analysis  Check for potential recursive formula issue.

options:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
```

### Telegram RPC commands

Telegram is not mandatory. However, this is a great way to control your bot. More details and the full command list on the [documentation](https://www.freqtrade.io/en/latest/telegram-usage/)

- `/start`: Starts the trader.
- `/stop`: Stops the trader.
- `/stopentry`: Stop entering new trades.
- `/status <trade_id>|[table]`: Lists all or specific open trades.
- `/profit [<n>]`: Lists cumulative profit from all finished trades, over the last n days.
- `/forceexit <trade_id>|all`: Instantly exits the given trade (Ignoring `minimum_roi`).
- `/fx <trade_id>|all`: Alias to `/forceexit`
- `/performance`: Show performance of each finished trade grouped by pair
- `/balance`: Show account balance per currency.
- `/daily <n>`: Shows profit or loss per day, over the last n days.
- `/help`: Show help message.
- `/version`: Show version.

## Development branches

The project is currently setup in two main branches:

- `develop` - This branch has often new features, but might also contain breaking changes. We try hard to keep this branch as stable as possible.
- `stable` - This branch contains the latest stable release. This branch is generally well tested.
- `feat/*` - These are feature branches, which are being worked on heavily. Please don't use these unless you want to test a specific feature.

## 🤝 Community & Support

### Join Our Thriving Community

Freqtrade has a vibrant, active community of traders, developers, and enthusiasts from around the world. Connect with us to share strategies, get help, and participate in the ongoing development of the platform.

### 💬 Discord Community

Our primary community hub is Discord, where thousands of Freqtrade users gather to:
- Get real-time help from experienced users
- Share trading strategies and insights
- Discuss feature ideas and development
- Connect with like-minded crypto traders

[Join the Freqtrade Discord Server](https://discord.gg/p7nuUNVfP7)

### 📚 Documentation

Comprehensive documentation is available at [freqtrade.io](https://www.freqtrade.io), covering everything from basic setup to advanced features.

### 🐛 Reporting Issues

Found a bug? We appreciate your help in improving Freqtrade!

1. First, [search existing issues](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue) to see if it's already reported
2. If not, [create a new issue](https://github.com/freqtrade/freqtrade/issues/new/choose) following the template
3. Provide detailed information to help us reproduce and fix the problem
4. Follow up on your issue and mark it resolved when fixed

Please follow GitHub's [community policy](https://docs.github.com/en/site-policy/github-terms/github-community-code-of-conduct) in all interactions.

### 💡 Feature Requests

Have ideas to make Freqtrade even better?

1. Check if your idea was [already suggested](https://github.com/freqtrade/freqtrade/labels/enhancement)
2. If not, [submit a feature request](https://github.com/freqtrade/freqtrade/issues/new/choose) with details about your idea
3. Use the template to clearly explain the feature and its benefits

### 🔄 Contributing Code

We enthusiastically welcome contributions from developers of all skill levels! Freqtrade thrives thanks to our community contributors.

#### How to Contribute:

1. **Read the [Contributing Guidelines](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md)** to understand our development process
2. **Start Small**: Look for issues labeled [good first issue](https://github.com/freqtrade/freqtrade/labels/good%20first%20issue) to get familiar with the codebase
3. **Documentation Matters**: Improving documentation is a valuable contribution that doesn't require coding expertise
4. **Discuss First**: Before starting work on major features, [open an issue](https://github.com/freqtrade/freqtrade/issues/new/choose) or discuss in the [#dev channel on Discord](https://discord.gg/p7nuUNVfP7) to get feedback
5. **Submit PRs to `develop`**: Always create pull requests against the `develop` branch, not `stable`

Every contribution, whether it's code, documentation, or bug reports, helps make Freqtrade better for everyone. We look forward to your contributions!

## ⚙️ System Requirements

### Hardware Recommendations

For optimal performance, we recommend:

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM       | 2GB     | 4GB+        |
| Disk Space| 1GB     | 5GB+        |
| CPU       | 2 vCPU  | 4+ vCPU     |

Cloud instances meeting these specifications work well for most users. For high-frequency trading or when using advanced machine learning features, consider higher specifications.

### Software Prerequisites

- **[Python 3.10+](http://docs.python-guide.org/en/latest/starting/installation/)**: The foundation of Freqtrade
- **[pip](https://pip.pypa.io/en/stable/installing/)**: For package management
- **[git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)**: For version control
- **[TA-Lib](https://ta-lib.github.io/ta-lib-python/)**: For technical analysis functions
- **[virtualenv](https://virtualenv.pypa.io/en/stable/installation.html)**: Recommended for isolated environments
- **[Docker](https://www.docker.com/products/docker)**: Recommended for easy deployment

### System Configuration

- **Synchronized Clock**: Ensure your system clock is accurately synchronized with an NTP server to prevent API communication issues with exchanges
- **Stable Internet Connection**: A reliable internet connection is essential for consistent trading operations

## 📈 Success Stories & Showcase

Freqtrade is used by thousands of traders worldwide, from beginners to professionals. While we don't guarantee profits (see our disclaimer), many users have successfully automated their trading strategies using our platform.

Share your success stories, strategies (without sensitive details), or showcase your Freqtrade setup in our Discord community!

## 🙏 Acknowledgements

Freqtrade exists thanks to the contributions of many individuals who have dedicated their time and expertise to make this project possible. We extend our gratitude to all contributors, testers, and community members who help make Freqtrade better every day.

Special thanks to all the [contributors](https://github.com/freqtrade/freqtrade/graphs/contributors) who have helped shape Freqtrade into what it is today.
