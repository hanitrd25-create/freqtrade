import tkinter as tk
from tkinter import ttk
import rapidjson
from pathlib import Path

class ConfigurationTab:
    def __init__(self, notebook, config_path="config.json"):
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="Configuration")

        self.config_path = config_path
        self.config = self.load_config()

        self.create_widgets()

    def load_config(self):
        try:
            with Path(self.config_path).open("r") as file:
                return rapidjson.load(file, parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS)
        except FileNotFoundError:
            return {}

    def create_widgets(self):
        # Create a canvas and a scrollbar
        self.canvas = tk.Canvas(self.frame)
        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Create widgets for each configuration section
        self.create_telegram_widgets()

    def create_telegram_widgets(self):
        telegram_frame = ttk.LabelFrame(self.scrollable_frame, text="Telegram")
        telegram_frame.pack(padx=10, pady=10, fill="x")

        telegram_config = self.config.get("telegram", {})

        # Enabled
        self.telegram_enabled_var = tk.BooleanVar(value=telegram_config.get("enabled", False))
        ttk.Checkbutton(telegram_frame, text="Enabled", variable=self.telegram_enabled_var).grid(row=0, column=0, sticky="w")

        # Token
        ttk.Label(telegram_frame, text="Token:").grid(row=1, column=0, sticky="w")
        self.telegram_token_entry = ttk.Entry(telegram_frame, width=50)
        self.telegram_token_entry.insert(0, telegram_config.get("token", ""))
        self.telegram_token_entry.grid(row=1, column=1, sticky="w")

        # Chat ID
        ttk.Label(telegram_frame, text="Chat ID:").grid(row=2, column=0, sticky="w")
        self.telegram_chat_id_entry = ttk.Entry(telegram_frame, width=50)
        self.telegram_chat_id_entry.insert(0, telegram_config.get("chat_id", ""))
        self.telegram_chat_id_entry.grid(row=2, column=1, sticky="w")
