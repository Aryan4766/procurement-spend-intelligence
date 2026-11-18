try:  # noqa: SIM105 - torch first (Windows DLL load order), then numpy/pandas
    import torch  # noqa: F401
except Exception:
    pass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
