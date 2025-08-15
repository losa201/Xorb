# COMPATIBILITY SHIMS - DEPRECATED
# All common modules have been consolidated to src/common/
# These shims will be removed in P06 - update your imports

import warnings
warnings.warn(
    "packages.common is deprecated. Use src.common instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all common modules for compatibility
try:
    from src.common.config import *  # noqa
    from src.common.encryption import *  # noqa  
    from src.common.vault_manager import *  # noqa
    from src.common.secret_manager import *  # noqa
    from src.common.config_manager import *  # noqa
    from src.common.jwt_manager import *  # noqa
    from src.common.performance_monitor import *  # noqa
    from src.common.unified_config import *  # noqa
except ImportError as e:
    warnings.warn(f"Failed to import from src.common: {e}", ImportWarning)