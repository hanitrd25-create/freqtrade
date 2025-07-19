import tkinter as tk
from tkinter import ttk

class SplashScreen:
    def __init__(self, root):
        self.root = root
        self.root.overrideredirect(True)
        self.root.geometry("400x200")
        self.root.eval('tk::PlaceWindow . center')

        self.create_widgets()

    def create_widgets(self):
        # Create a frame for the splash screen content
        self.frame = ttk.Frame(self.root, relief="solid", borderwidth=1)
        self.frame.pack(expand=True, fill="both")

        # Create a label for the title
        self.title_label = ttk.Label(self.frame, text="Freqtrade GUI", font=("Helvetica", 16))
        self.title_label.pack(pady=10)

        # Create a progress bar
        self.progress = ttk.Progressbar(self.frame, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)

        # Create a label for the log messages
        self.log_label = ttk.Label(self.frame, text="Initializing...")
        self.log_label.pack(pady=10)

    def set_progress(self, value):
        self.progress["value"] = value

    def set_log_message(self, message):
        self.log_label.config(text=message)
