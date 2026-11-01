"""Market direction research, with optional project-local dependencies."""
import sys
from pathlib import Path

local_deps = Path(__file__).resolve().parents[1] / ".deps"
if local_deps.exists():
    sys.path.append(str(local_deps))

__version__ = "0.1.0"
