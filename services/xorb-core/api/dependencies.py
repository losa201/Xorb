# COMPATIBILITY SHIM - DEPRECATED
# This file has been consolidated to src/api/dependencies.py
# This shim will be removed in P06 - update your imports

import warnings
warnings.warn(
    "services.xorb-core.api.dependencies is deprecated. "
    "Use src.api.dependencies instead.",
    DeprecationWarning,
    stacklevel=2
)

from src.api.dependencies import *  # noqa