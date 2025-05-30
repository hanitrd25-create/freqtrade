# Freqtrade Bot Project

## 1. Project Overview

A Dockerized Freqtrade setup with Binance integration, Telegram notifications, FreqUI, and a simple web UI for configuration. This project aims to provide an easy-to-deploy and manage Freqtrade instance suitable for both beginners and experienced users.

## 2. Features

*   **Freqtrade Trading Bot:** Leverages the powerful Freqtrade crypto trading bot.
*   **Binance Integration:** Pre-configured for trading on Binance (spot market).
*   **Telegram Bot:** Integrated for notifications (buy, sell, status, errors) and basic bot control.
*   **FreqUI:** Includes the Freqtrade web interface for monitoring trades, logs, and bot performance.
*   **Docker-compose Setup:** All components (Freqtrade, Config Backend, Config Frontend) are containerized and orchestrated with Docker Compose for easy deployment and management.
*   **Web UI for Configuration:** A simple, dedicated web interface to securely manage sensitive configurations like API keys, Telegram tokens, and FreqUI credentials, which are then written to `user_data/config.json`.

## 3. Prerequisites

Before you begin, ensure you have the following installed and configured:

*   **Docker and Docker Compose:** Essential for running the containerized application. Visit the [official Docker website](https://www.docker.com/get-started) for installation instructions.
*   **A Binance Account:** You will need to generate an API Key and Secret.
    *   Ensure your API Key has permissions for Spot & Margin Trading.
    *   **Crucially, for security, DO NOT enable withdrawals for this API key.**
*   **A Telegram Account:** You will need to:
    *   Create a new Telegram Bot using BotFather.
    *   Note down the **Bot Token**.
    *   Find your **Chat ID**. You can get this by sending a message to your bot and then querying the bot's updates API, or by using a helper bot like `@userinfobot`.

## 4. Directory Structure

A brief overview of the key directories and files:

*   `./user_data/`: This is the most important directory for Freqtrade. It stores:
    *   `config.json`: The main configuration file for Freqtrade, including your API keys, strategy settings, etc.
    *   `strategies/`: Your custom trading strategies (e.g., `ExampleStrategy.py`).
    *   `logs/`: Freqtrade operational logs.
    *   `data/`: Downloaded market data for backtesting.
    *   `backtest_results/`: Results from your backtests.
    *   `tradesv3.sqlite`: The database storing trade history.
*   `./config_backend/`: Contains the Python Flask backend application that powers the configuration UI.
    *   `app.py`: The Flask application logic.
    *   `requirements.txt`: Python dependencies for the backend.
*   `./config_frontend/`: Contains the static HTML, CSS, and JavaScript files for the configuration UI.
    *   `index.html`: The main page for the configuration UI.
    *   `style.css`: Styles for the UI.
    *   `script.js`: JavaScript for interacting with the backend API.
*   `Dockerfile`: Defines the Docker image for the Freqtrade bot itself, using an official Freqtrade image as a base.
*   `docker-compose.yml`: Defines and orchestrates all the services: `freqtrade` (the bot), `config-api` (the backend for config UI), and `config-ui` (the Nginx server for the frontend UI).

## 5. Setup and Configuration

Follow these steps to get your Freqtrade bot up and running:

**1. Clone the Repository:**

   If you haven't already, clone this repository to your local machine:
   ```bash
   # Replace <repository_url> with the actual URL and <repository_name> with the directory name
   git clone <repository_url>
   cd <repository_name>
   ```

**2. Initial Configuration (Recommended - Using Web UI):**

   This method is recommended for ease of use and to avoid manual JSON editing errors.

   *   **Start all services:**
       This command will build the images (if not already built) and start all containers in detached mode.
       ```bash
       docker-compose up -d
       ```
   *   **Access the Configuration UI:**
       Open your web browser and navigate to: `http://localhost:8081`
   *   **Enter Your Credentials:**
       *   Fill in your Binance API Key.
       *   Fill in your Binance API Secret.
       *   Fill in your Telegram Bot Token.
       *   Fill in your Telegram Chat ID.
       *   Choose a secure Username for FreqUI.
       *   Choose a secure Password for FreqUI.
   *   **Save the Configuration:**
       Click the "Save Configuration" button. This will send your details to the backend, which then updates the `user_data/config.json` file.
   *   **Restart Freqtrade:**
       For the Freqtrade bot to pick up the new settings from `config.json`, you must restart its container:
       ```bash
       docker-compose restart freqtrade
       ```

**3. Manual Configuration (Alternative):**

   If you prefer, you can manually edit the `user_data/config.json` file before running the bot for the first time.

   *   Locate the `user_data/config.json` file. If it doesn't exist (e.g., on a fresh clone where `user_data` might be empty or only contain `.gitkeep` files), the Freqtrade container will create a default one on its first run (as per the `docker-compose.yml` command). You might want to run `docker-compose up -d freqtrade` once to generate it, then `docker-compose down`, then edit. Or, if a `config_binance.example.json` was copied as `config.json` earlier, edit that.
   *   Replace the placeholder values for:
       *   `exchange.key`
       *   `exchange.secret`
       *   `telegram.token`
       *   `telegram.chat_id`
       *   `api_server.username`
       *   `api_server.password`
   *   After saving your manual changes, you can start all services:
       ```bash
       docker-compose up -d
       ```
       The Freqtrade bot should pick up these settings on its initial start.

## 6. Running the Bot

Once configured, you can manage the bot using Docker Compose:

*   **Start all services (Freqtrade, Config Backend, Config UI):**
    ```bash
    docker-compose up -d
    ```
*   **Stop all services:**
    This command stops and removes the containers. Your data in `user_data/` will persist.
    ```bash
    docker-compose down
    ```
*   **View Freqtrade logs:**
    Useful for monitoring bot activity and troubleshooting.
    ```bash
    docker-compose logs -f freqtrade
    ```
*   **View Config Backend (Flask API) logs:**
    ```bash
    docker-compose logs -f config-api
    ```
*   **View Config UI (Nginx) logs:**
    ```bash
    docker-compose logs -f config-ui
    ```
*   **Restart a specific service (e.g., Freqtrade):**
    ```bash
    docker-compose restart freqtrade
    ```

## 7. Accessing Interfaces

*   **Configuration UI:** `http://localhost:8081`
    *   Use this interface to update your API keys, Telegram tokens, or FreqUI credentials. Remember to restart the `freqtrade` container after saving changes here if they affect Freqtrade's core operation.
*   **FreqUI (Freqtrade Web Interface):** `http://localhost:8080`
    *   Login with the username and password you set via the Configuration UI (or manually in `config.json`).
    *   FreqUI allows you to monitor trades, view charts, check logs, and interact with the bot.
*   **Telegram Bot:**
    *   Open Telegram and find the bot you created.
    *   You can interact with it using commands like:
        *   `/status` or `/status table`: View open trades.
        *   `/profit`: Check overall profit/loss.
        *   `/start`: Start the bot (if stopped via Telegram).
        *   `/stop`: Stop the bot (trading will halt, but open trades remain).
        *   `/forceexit <trade_id>`: Manually sell a specific trade.
        *   Refer to the [official Freqtrade Telegram documentation](https://www.freqtrade.io/en/stable/telegram-usage/) for a full list of commands.

## 8. Trading Strategy

*   **Default Strategy:** The bot is configured by default to use the `ExampleStrategy` located in `user_data/strategies/ExampleStrategy.py`. This is specified in the `command` section of the `freqtrade` service in `docker-compose.yml` and should also be reflected in `user_data/config.json` under the `strategy` key.
*   **Developing Your Own Strategies:**
    *   Create your strategy Python file (e.g., `MyAwesomeStrategy.py`) inheriting from `IStrategy`.
    *   Place this file in the `user_data/strategies/` directory.
*   **Using a Different Strategy:**
    1.  Update the `strategy` field in `user_data/config.json`. For example:
        ```json
        "strategy": "MyAwesomeStrategy",
        ```
        (You can do this via the Configuration UI if it's extended to support strategy selection, or manually edit `config.json`.)
    2.  Ensure the corresponding `MyAwesomeStrategy.py` file is in `user_data/strategies/`.
    3.  Restart the Freqtrade container to apply the change:
        ```bash
        docker-compose restart freqtrade
        ```

## 9. Production & Security Considerations

*   **API Keys:**
    *   **Permissions:** When creating Binance API keys, grant only necessary permissions: "Enable Reading" and "Enable Spot & Margin Trading". **Crucially, DO NOT enable Withdrawals.**
    *   **Git Commits:** **NEVER commit your `user_data/config.json` file (or any file containing live API keys or secrets) to a public Git repository.** The provided `.gitignore` file should already exclude the `user_data/` directory, but always double-check.
*   **FreqUI Credentials:** Use a strong, unique username and password for FreqUI. These are stored in `config.json`.
*   **Dry-run vs. Live Trading:**
    *   The bot is configured for `dry_run: true` by default (paper trading with simulated funds). This is highly recommended for testing.
    *   Thoroughly test your strategies and bot configuration in dry-run mode before considering live trading.
    *   To switch to live trading:
        1.  Set `dry_run: false` in `user_data/config.json` (you can use the Configuration UI or edit manually).
        2.  Ensure your live Binance API keys (with trading permissions) are correctly configured.
        3.  Restart Freqtrade: `docker-compose restart freqtrade`.
    *   **RISK WARNING: Live trading involves real financial risk. You could lose money. Understand the risks and your strategy's behavior before proceeding.**
*   **Server Deployment:** For continuous 24/7 operation, deploy this setup on a reliable server or Virtual Private Server (VPS), not your personal computer.
*   **Configuration UI Security:**
    *   The provided Configuration UI (`config-frontend` and `config-api`) is designed for ease of use in a trusted environment.
    *   The backend API (`config-api`) currently returns API keys and secrets when the "Load" button is pressed. While this is convenient for personal setup, be mindful if others have access to your network or the machine running the Docker containers.
    *   For enhanced security in less trusted environments, you might consider:
        *   Not displaying secrets in the UI after they've been set (e.g., show "********" or "Set").
        *   Adding authentication to the `config-api` itself.
        *   Restricting network access to the `config-api` port (5001).
    *   For typical personal use on a local machine or a secured VPS, the current setup offers a good balance of convenience and security.

## 10. Troubleshooting

*   **Port Conflicts:** If ports `8080` (FreqUI), `8081` (Config Frontend), or `5001` (Config Backend) are already in use on your system, Docker Compose will fail to start the respective service. You can change the host-side port mapping in `docker-compose.yml`. For example, to change FreqUI to port `8888` on your host, modify `ports` for the `freqtrade` service to `"- 8888:8080"`.
*   **Docker Issues:**
    *   Ensure the Docker daemon is running.
    *   Run `docker-compose ps` to check the status of your containers.
    *   If a container is not starting or is repeatedly exiting, check its logs (e.g., `docker-compose logs -f freqtrade`).
*   **Freqtrade Logs:** Detailed operational logs for the Freqtrade bot can be found in `user_data/logs/freqtrade.log` (inside the container, mapped to your local `user_data/logs/`) or viewed directly via Docker logs.
*   **Strategy Issues:** If Freqtrade fails to start and logs indicate a strategy error:
    *   Ensure your strategy file (e.g., `ExampleStrategy.py`) is correctly placed in the `user_data/strategies/` directory.
    *   Check for Python syntax errors in your strategy file.
    *   Verify that the strategy name in `config.json` matches the filename (without `.py`) and the class name within the strategy file.
*   **"Config file not found" by Flask App:** If the `config-api` logs show errors related to not finding `/app/user_data/config.json`, double-check the `volumes` section for the `config-api` service in `docker-compose.yml`. It should correctly map `./user_data` from your host to `/app/user_data` in the container.

---

Happy Trading! Remember to trade responsibly.

## 11. Advanced Strategy Development & FreqAI

Beyond the `ExampleStrategy.py`, Freqtrade offers powerful tools for crafting sophisticated trading approaches.

**1. Custom Strategy Development:**

*   The `ExampleStrategy.py` provided is a basic SMA crossover strategy, intended as a starting point.
*   You are encouraged to develop your own strategies by creating new `.py` files within the `user_data/strategies/` directory.
*   Strategies are Python classes that inherit from `IStrategy` and implement specific methods to define indicators, buy signals, and sell signals.
*   For detailed guidance, refer to the official Freqtrade strategy development documentation: [Strategy Customization](https://www.freqtrade.io/en/stable/strategy-customization/)

**2. Introduction to FreqAI:**

*   FreqAI is Freqtrade's integrated machine learning module. It allows you to build strategies that can learn from market data and adapt over time.
*   This can involve predicting market movements, identifying optimal entry/exit points based on complex patterns, or even managing risk more dynamically.
*   Setting up and training FreqAI models is an advanced topic that requires a good understanding of both trading concepts and machine learning principles.
*   To explore FreqAI, visit the official documentation: [FreqAI](https://www.freqtrade.io/en/stable/freqai/)

**3. Importance of Backtesting and Hyperoptimization:**

*   **Backtesting:** Before deploying any strategy with real funds, it is crucial to test its performance on historical data. Freqtrade provides robust backtesting tools to simulate how your strategy would have performed in the past.
    *   Learn more: [Backtesting](https://www.freqtrade.io/en/stable/backtesting/)
*   **Hyperoptimization (Hyperopt):** Most strategies have parameters (e.g., SMA periods, ROI levels, stoploss values). Hyperopt helps you find the optimal values for these parameters by systematically testing different combinations against historical data.
    *   Learn more: [Hyperopt](https://www.freqtrade.io/en/stable/hyperopt/)

**4. Realistic Expectations:**

*   There is no "guaranteed profitable" or "best strategy." Market conditions change, and past performance is not indicative of future results.
*   Algorithmic trading, especially with advanced techniques like machine learning, is a field of continuous learning, experimentation, and adaptation. Be prepared to invest time in understanding the tools, developing your strategies, and managing risk.
