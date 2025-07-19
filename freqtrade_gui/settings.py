import tkinter as tk
from tkinter import ttk
import os
import difflib
import shutil

class SettingsTab:
    def __init__(self, notebook, app):
        self.app = app
        self.frame = ttk.Frame(notebook)
        notebook.add(self.frame, text="Settings")

        self.create_widgets()
        self.generate_compliance_report()

    def create_widgets(self):
        # Create an offline mode frame
        self.offline_frame = ttk.LabelFrame(self.frame, text="Offline Mode")
        self.offline_frame.pack(padx=10, pady=10, fill="x")

        self.offline_mode_var = tk.BooleanVar()
        self.offline_mode_check = ttk.Checkbutton(
            self.offline_frame,
            text="Enable Offline Mode",
            variable=self.offline_mode_var,
            command=self.toggle_offline_mode,
        )
        self.offline_mode_check.pack(side="left", padx=5)

        # Create a compliance report frame
        self.compliance_frame = ttk.LabelFrame(self.frame, text="Compliance Report")
        self.compliance_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Create a text widget for the diff report
        self.diff_text = tk.Text(self.compliance_frame, wrap="word", state="disabled")
        self.diff_text.pack(expand=True, fill="both")

        # Create a rollback button
        self.rollback_button = ttk.Button(self.compliance_frame, text="Rollback All Patches", command=self.rollback_patches)
        self.rollback_button.pack(pady=5)

    def generate_compliance_report(self):
        self.diff_text.config(state="normal")
        self.diff_text.delete("1.0", tk.END)

        patches_dir = "user_data/gui_patches"
        if not os.path.exists(patches_dir):
            self.diff_text.insert(tk.END, "No patches found.")
            self.diff_text.config(state="disabled")
            return

        for patch_file in os.listdir(patches_dir):
            if patch_file.endswith(".patch"):
                original_file_path = os.path.join("freqtrade", patch_file.replace(".patch", ""))
                patched_file_path = os.path.join(patches_dir, patch_file)

                with open(original_file_path, "r") as f1, open(patched_file_path, "r") as f2:
                    diff = difflib.unified_diff(
                        f1.readlines(),
                        f2.readlines(),
                        fromfile=original_file_path,
                        tofile=patched_file_path,
                    )
                    self.diff_text.insert(tk.END, f"--- {original_file_path}\n")
                    self.diff_text.insert(tk.END, f"+++ {patched_file_path}\n")
                    self.diff_text.insert(tk.END, "".join(diff))
                    self.diff_text.insert(tk.END, "\n\n")

        self.diff_text.config(state="disabled")


    def rollback_patches(self):
        patches_dir = "user_data/gui_patches"
        if not os.path.exists(patches_dir):
            return

        for patch_file in os.listdir(patches_dir):
            if patch_file.endswith(".patch"):
                original_file_path = os.path.join("freqtrade", patch_file.replace(".patch", ""))
                patched_file_path = os.path.join(patches_dir, patch_file)

                # For simplicity, we'll just remove the patch files.
                # A more robust solution would be to apply the patches in reverse.
                os.remove(patched_file_path)

        self.generate_compliance_report()

    def toggle_offline_mode(self):
        self.app.offline_mode = self.offline_mode_var.get()
