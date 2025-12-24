import importlib.metadata

_openai_version = importlib.metadata.version("openai")

if not _openai_version.startswith("0."):
    from ._openai_llm_v1 import make_llm_functions
else:
    from ._openai_llm_v0 import make_llm_functions

__all__ = ["make_llm_functions"]
