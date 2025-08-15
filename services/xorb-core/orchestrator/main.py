# COMPATIBILITY SHIM - DEPRECATED
# This file has been consolidated to src/orchestrator/main.py
# This shim will be removed in P06 - update your imports

import warnings
warnings.warn(
    "services.xorb-core.orchestrator.main is deprecated. "
    "Use src.orchestrator.main instead.",
    DeprecationWarning,
    stacklevel=2
)

from src.orchestrator.main import *  # noqa