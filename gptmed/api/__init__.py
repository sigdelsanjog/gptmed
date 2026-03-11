"""
gptmed.api
──────────
This package serves two purposes:

1. Re-exports gptmed's original high-level training/generation API so that
   ``from gptmed.api import create_config`` continues to work now that api.py
   is shadowed by this package directory.

2. Hosts the FastAPI application for local agentic-framework testing.
   Import the app with ``from gptmed.api.app import app``.
"""

# ── Re-export original high-level training API ────────────────────────────────
# Guarded: the model architecture deps (torch, etc.) may not be installed in
# lightweight environments (e.g. when only the agentic API is used).
try:
    from ._high_level import create_config, train_from_config, generate
    __all__ = ["create_config", "train_from_config", "generate"]
except ImportError:
    pass
