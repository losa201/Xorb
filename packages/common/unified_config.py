# COMPATIBILITY SHIM - DEPRECATED
# This file has been consolidated to src/common/unified_config.py
# This shim will be removed in P06 - update your imports

import warnings
warnings.warn(
    "packages.common.unified_config is deprecated. "
    "Use src.common.unified_config instead.",
    DeprecationWarning,
    stacklevel=2
)

from src.common.unified_config import *  # noqa