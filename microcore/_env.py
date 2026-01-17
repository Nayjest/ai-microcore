"""
MicroCore environment object / initialization.
"""
import os.path
from dataclasses import dataclass, field, asdict, fields
from importlib.util import find_spec
from typing import TYPE_CHECKING

import jinja2

from .embedding_db import AbstractEmbeddingDB
from .configuration import (
    Config,
    LLMConfigError,
    EmbeddingDbType,
    PRINT_STREAM,
)
from .llm_backends import ApiType
from .presets import MIN_SETUP
from .lm_client import BaseAIClient
from .types import TplFunctionType, LLMAsyncFunctionType, LLMFunctionType
from .templating.jinja2 import make_jinja2_env, make_tpl_function
from .llm.local_llm import make_llm_functions as make_local_llm_functions

if TYPE_CHECKING:
    from .wrappers.llm_response_wrapper import LLMResponse  # noqa: F401
    from transformers import PreTrainedModel, PreTrainedTokenizer  # noqa: F401
    from .mcp import MCPRegistry


@dataclass
class Env:
    config: Config = field(default_factory=Config)
    jinja_env: jinja2.Environment = None
    tpl_function: TplFunctionType = None
    llm_async_function: LLMAsyncFunctionType = None
    llm_function: LLMFunctionType = None
    llm_before_handlers: list[callable] = field(default_factory=list)
    llm_after_handlers: list[callable] = field(default_factory=list)
    texts: AbstractEmbeddingDB = None
    model: "PreTrainedModel" = field(
        default=None, init=False, repr=False
    )  # noqa
    tokenizer: "PreTrainedTokenizer" = field(  # noqa
        default=None, init=False, repr=False
    )
    default_client: BaseAIClient | None = None
    _mcp_registry: "MCPRegistry" = field(init=False, default=None)

    def __post_init__(self):
        global _env
        _env = self
        self.init_templating()
        self.init_llm()
        if self.config.USE_LOGGING:
            from .logging import use_logging

            use_logging(stream=self.config.USE_LOGGING == PRINT_STREAM)
        self.init_similarity_search()

    def make_stopping_criteria(self, seq: str | list[str]) -> list[callable]:
        raise NotImplementedError

    def init_templating(self):
        """Initialize Jinja2 environment and template function for templates rendering."""
        self.jinja_env = make_jinja2_env(self)
        self.tpl_function = make_tpl_function(self)

    @property
    def mcp_registry(self) -> "MCPRegistry":
        """Lazily initialize and return the registry of preconfigured MCP servers."""
        if self._mcp_registry is None:
            from .mcp import MCPRegistry
            self._mcp_registry = MCPRegistry(self.config.MCP_SERVERS)
        return self._mcp_registry

    def init_llm(self):
        """Initialize language model functions based on configuration."""

        def default_llm(*args, **kwargs) -> "LLMResponse":
            if self.default_client:
                return self.default_client.generate(*args, **kwargs)
            raise LLMConfigError("Language model is not configured")

        async def aio_default_llm(*args, **kwargs) -> "LLMResponse":
            if self.default_client:
                return await self.default_client.aio.generate(*args, **kwargs)
            raise LLMConfigError("Language model is not configured")

        self.llm_function, self.llm_async_function = (
            default_llm,
            aio_default_llm,
        )

        if self.config.LLM_API_TYPE == ApiType.NONE:
            pass
        elif self.config.LLM_API_TYPE == ApiType.FUNCTION:
            self.llm_function, self.llm_async_function = make_local_llm_functions(
                self.config
            )
        elif self.config.LLM_API_TYPE == ApiType.TRANSFORMERS:
            try:
                from .llm.local_transformers import (
                    make_llm_functions as make_transformers_llm_functions,
                )
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "To use local Transformers language models, "
                    "you need to install the `transformers`, `pytorch` and `accelerate` packages. "
                ) from e
            (
                self.llm_function,
                self.llm_async_function,
            ) = make_transformers_llm_functions(self.config, self)
        elif self.config.LLM_API_TYPE == ApiType.ANTHROPIC:
            try:
                from .llm.anthropic import (
                    make_llm_functions as make_anthropic_llm_functions,
                )
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "To use the Anthropic language models, "
                    "you need to install the `anthropic` package. "
                    "Run `pip install anthropic`."
                ) from e
            self.llm_function, self.llm_async_function = make_anthropic_llm_functions(
                self.config
            )
        elif self.config.LLM_API_TYPE in (
            ApiType.GOOGLE,
            ApiType.GOOGLE_AI_STUDIO,  # @deprecated
            ApiType.GOOGLE_VERTEX_AI  # @deprecated
        ):
            try:
                from .llm.google_genai import GoogleClient
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "To use the Google Gemini language models via Google GenAI SDK, "
                    "you need to install the `google-genai` package. "
                    "Run `pip install google-genai`."
                ) from e
            self.default_client = GoogleClient(self.config)
        else:
            from .llm.openai import OpenAIClient
            self.default_client = OpenAIClient(self.config)

    def init_similarity_search(self):
        if (
            self.config.EMBEDDING_DB_TYPE == EmbeddingDbType.CHROMA
            and find_spec("chromadb") is not None
        ):
            from .embedding_db.chromadb import ChromaEmbeddingDB

            self.texts = ChromaEmbeddingDB(self.config)
            return

        if self.config.EMBEDDING_DB_TYPE == EmbeddingDbType.QDRANT:
            if find_spec("qdrant_client") is None:
                raise ModuleNotFoundError(
                    "To use Qdrant, install the `qdrant-client` package. "
                    "Run `pip install qdrant-client`."
                )
            from .embedding_db.qdrant import QdrantEmbeddingDB
            self.texts = QdrantEmbeddingDB(self.config)
            return


@dataclass
class _Configure(Config):
    def __post_init__(self):
        global _env
        _env = None
        super().__post_init__()
        Env(self)


configure: callable = _Configure
"""Applies configuration to MicroCore environment"""

if True:  # pylint: disable=W0125
    # This block is inside a condition to avoid breaking IDE autocompletion

    _fields = list(map(lambda f: f.name, fields(Config)))

    def _config_builder_wrapper(cfg: Config | dict | str = None, **kwargs):
        """
        - Convert configuration keys to uppercase
        - Add LLM_ prefix to keys if necessary
        - Allow to configure from Config instance or dictionary
        """
        if cfg:
            assert not kwargs, "Cannot pass both cfg and kwargs"
        if isinstance(cfg, dict):
            return _config_builder_wrapper(**cfg)
        if isinstance(cfg, str):
            if not os.path.isfile(cfg):
                raise LLMConfigError(f"Configuration file not found: {cfg}")
            return _config_builder_wrapper(Config(USE_DOT_ENV=True, DOT_ENV_FILE=cfg))
        kwargs = {str(k).upper(): v for k, v in kwargs.items()}
        for k in list(kwargs.keys()):
            if not hasattr(Config, k) and (
                hasattr(Config, key := f"LLM_{k}") or key in _fields
            ):
                kwargs[key] = kwargs.pop(k)
        return _Configure(**(cfg and asdict(cfg) or kwargs))

    configure = _config_builder_wrapper


def min_setup():
    """
    Minimal handy setup for non-production usage
    (simple scripts, small experiments, etc.)
    """
    return configure(MIN_SETUP)


_env: Env | None = None


def env() -> Env:
    """Return current MicroCore environment object."""
    return _env or Env()


def config() -> Config:
    """Resolve current configuration."""
    return env().config
