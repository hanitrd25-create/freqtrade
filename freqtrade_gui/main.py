import tkinter as tk
from tkinter import ttk
import psutil
import logging
from logging.handlers import QueueHandler, QueueListener
import queue
import time
import asyncio
import httpx
from freqtrade_gui.splash import SplashScreen
from freqtrade_gui.command_audit import get_cli_commands, get_rpc_commands
from freqtrade_gui.configuration import ConfigurationTab
from freqtrade_gui.paper_trading import PaperTradingTab
from freqtrade_gui.settings import SettingsTab

class FreqtradeGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Freqtrade GUI")
        self.root.geometry("1200x800")
        self.offline_mode = False

        self.create_widgets()
        self.update_status_bar()

        # Set up logging
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.queue_handler)

        # Create a listener to process log records from the queue
        self.log_listener = QueueListener(self.log_queue, self)
        self.log_listener.start()

        # Start the async event loop for API calls
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.api_sync_loop())


    def create_widgets(self):
        # Create a notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both")
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # Create a dictionary to store the tabs that have been loaded
        self.loaded_tabs = {}

        # Add tabs
        self.add_tab("Logs", self.create_logs_tab)
        self.add_tab("Command Audit", self.create_audit_tab)
        self.add_tab("Configuration", self.create_config_tab)
        self.add_tab("Paper Trading", self.create_paper_trading_tab)
        self.add_tab("Settings", self.create_settings_tab)


        # Create the status bar
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side="bottom", fill="x")
        self.create_status_bar()

        # Load the first tab
        self.on_tab_changed(None)


    def add_tab(self, text, creation_func):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=text)
        self.loaded_tabs[text] = {"frame": frame, "loaded": False, "creation_func": creation_func}

    def on_tab_changed(self, event):
        selected_tab = self.notebook.select()
        if not selected_tab:
            return
        tab_name = self.notebook.tab(selected_tab, "text")
        if not self.loaded_tabs[tab_name]["loaded"]:
            self.loaded_tabs[tab_name]["creation_func"]()
            self.loaded_tabs[tab_name]["loaded"] = True

    def create_logs_tab(self):
        frame = self.loaded_tabs["Logs"]["frame"]
        # Create a text widget for logs
        self.log_text = tk.Text(frame, wrap="word", state="disabled")
        self.log_text.pack(expand=True, fill="both")

        # Create a search bar
        self.search_bar = ttk.Entry(frame)
        self.search_bar.pack(side="bottom", fill="x")
        self.search_bar.bind("<KeyRelease>", self.search_logs)
        self.log_text.after(100, self.poll_log_queue)

    def create_audit_tab(self):
        frame = self.loaded_tabs["Command Audit"]["frame"]
        # Create a text widget for the audit log
        self.audit_text = tk.Text(frame, wrap="word", state="disabled")
        self.audit_text.pack(expand=True, fill="both")

        # Get the commands and display them
        cli_commands = get_cli_commands()
        rpc_commands = get_rpc_commands()

        self.audit_text.config(state="normal")
        self.audit_text.insert(tk.END, "CLI Commands:\n")
        for cmd in cli_commands:
            self.audit_text.insert(tk.END, f"  - File: {cmd['file']}, Command: {cmd['command']}\n")
        self.audit_text.insert(tk.END, "\nRPC Commands:\n")
        for cmd in rpc_commands:
            self.audit_text.insert(tk.END, f"  - File: {cmd['file']}, Command: {cmd['command']}\n")
        self.audit_text.config(state="disabled")

    def create_config_tab(self):
        frame = self.loaded_tabs["Configuration"]["frame"]
        self.config_tab = ConfigurationTab(frame)

    def create_paper_trading_tab(self):
        frame = self.loaded_tabs["Paper Trading"]["frame"]
        self.paper_trading_tab = PaperTradingTab(frame)

    def create_settings_tab(self):
        frame = self.loaded_tabs["Settings"]["frame"]
        self.settings_tab = SettingsTab(frame, self)


    def create_status_bar(self):
        # Create labels for memory and CPU usage
        self.mem_label = ttk.Label(self.status_bar, text="Mem: N/A")
        self.mem_label.pack(side="left")
        self.cpu_label = ttk.Label(self.status_bar, text="CPU: N/A")
        self.cpu_label.pack(side="left")
        self.api_status_label = ttk.Label(self.status_bar, text="API: N/A")
        self.api_status_label.pack(side="right")


    def update_status_bar(self):
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        self.mem_label.config(text=f"Mem: {mem.percent}%")
        self.cpu_label.config(text=f"CPU: {cpu}%")
        self.root.after(1000, self.update_status_bar)

    def search_logs(self, event):
        # TODO: Implement log searching
        pass

    def handle(self, record):
        """
        This method is called by the QueueListener for each log record.
        """
        if hasattr(self, 'log_text'):
            self.display_log_record(record)

    def display_log_record(self, record):
        msg = self.queue_handler.format(record)
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.config(state="disabled")
        self.log_text.see(tk.END)

    def poll_log_queue(self):
        # Check for log messages
        while True:
            try:
                record = self.log_queue.get(block=False)
            except queue.Empty:
                break
            else:
                self.handle(record)
        if hasattr(self, 'log_text'):
            self.root.after(100, self.poll_log_queue)

    async def api_sync_loop(self):
        while True:
            if not self.offline_mode:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get("http://127.0.0.1:8080/api/v1/status")
                        if response.status_code == 200:
                            self.api_status_label.config(text="API: Connected", foreground="green")
                            # TODO: Update GUI fields with data from response.json()
                        else:
                            self.api_status_label.config(text=f"API: Error {response.status_code}", foreground="red")
                except httpx.ConnectError:
                    self.api_status_label.config(text="API: Disconnected", foreground="red")
            else:
                self.api_status_label.config(text="API: Offline", foreground="orange")
            await asyncio.sleep(2)


def main():
    root = tk.Tk()
    root.withdraw()  # Hide the main window initially

    splash = tk.Toplevel(root)
    splash_screen = SplashScreen(splash)

    def update_splash(progress, message):
        splash_screen.set_progress(progress)
        splash_screen.set_log_message(message)
        splash.update()

    # Simulate pre-loading phases
    update_splash(10, "Validating environment...")
    time.sleep(1)
    update_splash(30, "Discovering config files...")
    time.sleep(1)
    update_splash(50, "Testing exchange connectivity...")
    time.sleep(1)
    update_splash(70, "Loading strategies...")
    time.sleep(1)
    update_splash(100, "GUI is ready.")
    time.sleep(0.5)

    splash.destroy()
    root.deiconify()  # Show the main window

    logging.basicConfig(level=logging.INFO)
    app = FreqtradeGui(root)
    logging.info("Freqtrade GUI started.")

    # This is a bit of a hack to run the tkinter mainloop and the asyncio event loop together.
    # A better solution would be to use a library like `quamash`.
    async def main_loop():
        while True:
            root.update()
            await asyncio.sleep(0.01)

    app.loop.run_until_complete(main_loop())


if __name__ == "__main__":
    main()
