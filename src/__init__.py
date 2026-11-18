"""Cantill spend-intelligence engine.

torch must be imported before numpy/pandas on Windows (otherwise its DLL initialisation can fail with an access
violation and the pipeline would silently lose the semantic model). Importing it here guarantees the order for
every entry point (main.py, tests, notebooks) that imports the package first.
"""
try:  # noqa: SIM105
    import torch  # noqa: F401
except Exception:  # torch unavailable: the engine falls back to lexical similarity and reports it
    pass
