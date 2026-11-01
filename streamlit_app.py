"""Original swing UI entry point; the full research app runs separately on 8502."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name("swing_app.py")),run_name="__main__")
