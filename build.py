import PyInstaller.__main__
import os
import shutil

def build():
    # Clean up previous builds
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    if os.path.exists("build"):
        shutil.rmtree("build")

    # Run PyInstaller
    PyInstaller.__main__.run([
        "freqtrade_gui/main.py",
        "--name=FreqtradeGUI",
        "--onefile",
        "--windowed",
        "--add-data=freqtrade_gui:freqtrade_gui"
    ])

if __name__ == "__main__":
    build()
