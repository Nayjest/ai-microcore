"""Configuration presets collection."""

from .configuration import PRINT_STREAM, DEFAULT_LOCAL_ENV_FILE, EmbeddingDbType
from .llm_backends import ApiType

# Minimal handy configuration preset
# for non-production usage (simple scripts, small experiments, etc.)
MIN_SETUP = dict(
    INTERACTIVE_SETUP=True,
    DOT_ENV_FILE=DEFAULT_LOCAL_ENV_FILE,
    USE_LOGGING=PRINT_STREAM,
)

# No LLM, No embedding DB
EMPTY = dict(
    LLM_API_TYPE=ApiType.NONE,
    EMBEDDING_DB_TYPE=EmbeddingDbType.NONE
)
