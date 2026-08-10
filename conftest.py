"""Put the repo root on sys.path so tests can import the top-level modules.

The project has no package layout — data.py, SNES.py, TNEP.py etc. live at
the root — so pytest's prepend import mode would otherwise only see tests/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
