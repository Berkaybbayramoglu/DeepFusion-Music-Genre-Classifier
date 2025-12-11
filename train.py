#!/usr/bin/env python3
"""Top-level runner for the modular propes_model package."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from propes_model.train import main

if __name__ == "__main__":
    main()
