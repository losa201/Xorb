# COMPATIBILITY SHIM - DEPRECATED
# This file has been consolidated to src/api/gateway.py
# This shim will be removed in P06 - update your imports

import warnings
warnings.warn(
    "services.xorb-core.api.gateway is deprecated. "
    "Use src.api.gateway instead.",
    DeprecationWarning,
    stacklevel=2
)

from src.api.gateway import *  # noqa