# COMPATIBILITY SHIM - DEPRECATED
# This file has been consolidated to src/api/db_management.py
# This shim will be removed in P06 - update your imports

import warnings
warnings.warn(
    "services.xorb-core.api.db_management is deprecated. "
    "Use src.api.db_management instead.",
    DeprecationWarning,
    stacklevel=2
)

from src.api.db_management import *  # noqa