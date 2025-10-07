"""Configuration presets collection."""

from .configuration import PRINT_STREAM, DEFAULT_LOCAL_ENV_FILE

# Minimal handy configuration preset
# for non-production usage (simple scripts, small experiments, etc.)
MIN_SETUP = dict(
    INTERACTIVE_SETUP=True,
    DOT_ENV_FILE=DEFAULT_LOCAL_ENV_FILE,
    USE_LOGGING=PRINT_STREAM,
)
