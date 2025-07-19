import tkinter as tk
from tkinter import ttk
import subprocess
import os
import psutil

class PaperTradingTab:
    def __init__(self, notebook):
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="Paper Trading")

        self.create_widgets()
        self.start_paper_trading()

    def create_widgets(self):
        # Create a dashboard frame
        self.dashboard_frame = ttk.LabelFrame(self.frame, text="Dashboard")
        self.dashboard_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Create a frame for the chart
        self.chart_frame = ttk.Frame(self.dashboard_frame)
        self.chart_frame.pack(side="left", fill="both", expand=True)
        self.chart_label = ttk.Label(self.chart_frame, text="Candlestick Chart")
        self.chart_label.pack()

        # Create a frame for the open positions and equity curve
        self.right_frame = ttk.Frame(self.dashboard_frame)
        self.right_frame.pack(side="right", fill="y")

        # Create a table for open positions
        self.positions_table = ttk.Treeview(self.right_frame, columns=("id", "entry", "exit", "pnl"), show="headings")
        self.positions_table.heading("id", text="ID")
        self.positions_table.heading("entry", text="Entry")
        self.positions_table.heading("exit", text="Exit")
        self.positions_table.heading("pnl", text="PnL %")
        self.positions_table.pack(fill="both", expand=True)

        # Create a frame for the equity curve
        self.equity_frame = ttk.Frame(self.right_frame)
        self.equity_frame.pack(fill="both", expand=True)
        self.equity_label = ttk.Label(self.equity_frame, text="Equity Curve")
        self.equity_label.pack()

        # Create a frame for the safety switches
        self.safety_switches_frame = ttk.LabelFrame(self.frame, text="Safety Switches")
        self.safety_switches_frame.pack(padx=10, pady=10, fill="x")

        # Panic Close button
        self.panic_button = ttk.Button(self.safety_switches_frame, text="Panic Close All", command=self.panic_close)
        self.panic_button.pack(side="left", padx=5)

        # Pause Strategy checkbox
        self.pause_strategy_var = tk.BooleanVar()
        self.pause_strategy_check = ttk.Checkbutton(self.safety_switches_frame, text="Pause Strategy", variable=self.pause_strategy_var, command=self.toggle_pause_strategy)
        self.pause_strategy_check.pack(side="left", padx=5)

    def start_paper_trading(self):
        # Create a dummy config file for paper trading
        paper_config = {
            "exchange": {
                "name": "binance",
                "key": "",
                "secret": "",
                "dry_run": True
            },
            "stake_currency": "USDT",
            "stake_amount": 1000,
            "strategy": "SampleStrategy"
        }
        with open("user_data/gui_paper_config.json", "w") as f:
            f.write(str(paper_config).replace("'", '"'))

        # Start freqtrade in the background
        process = subprocess.Popen(
            ["freqtrade", "trade", "--config", "user_data/gui_paper_config.json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy()
        )
        p = psutil.Process(process.pid)
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        self.process = p


    def panic_close(self):
        # TODO: Implement panic close
        pass

    def toggle_pause_strategy(self):
        # TODO: Implement pause strategy
        pass
