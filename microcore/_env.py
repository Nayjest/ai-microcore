import os.path
from dataclasses import dataclass, field, asdict, fields
from importlib.util import find_spec
from typing import TYPE_CHECKING

import jinja2

from .embedding_db import AbstractEmbeddingDB
from .configuration import Config, ApiType, LLMConfigError
from .types import TplFunctionType, LLMAsyncFunctionType, LLMFunctionType
from .templating.jinja2 import make_jinja2_env, make_tpl_function
from .llm.openai_llm import make_llm_functions as make_openai_llm_functions
from .llm.local_llm import make_llm_functions as make_local_llm_functions
if TYPE_CHECKING:
    from .wrappers.llm_response_wrapper import LLMResponse  # noqa: F401

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
    model: "transformers.PreTrainedModel" = field(default=None, init=False, repr=False)  # noqa
    tokenizer: "transformers.PreTrainedTokenizer" = field(  # noqa
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        global _env
        _env = self
        self.init_templating()
        self.init_llm()
        if self.config.USE_LOGGING:
            from .logging import use_logging

            use_logging()
        self.init_similarity_search()

    def make_stopping_criteria(self, seq: str | list[str]) -> list[callable]:
        raise NotImplementedError

    def init_templating(self):
        self.jinja_env = make_jinja2_env(self)
        self.tpl_function = make_tpl_function(self)

    def init_llm(self):
        if self.config.LLM_API_TYPE == ApiType.NONE:
            def not_configured(*args, **kwargs) -> "LLMResponse":
                raise LLMConfigError("Language model is not configured")

            async def a_not_configured(*args, **kwargs) -> "LLMResponse":
                raise LLMConfigError("Language model is not configured")

            self.llm_function, self.llm_async_function = (
                not_configured,
                a_not_configured,
            )

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
        elif self.config.LLM_API_TYPE == ApiType.GOOGLE_VERTEX_AI:
            try:
                from .llm.google_vertex_ai import (
                    make_llm_functions as make_google_vertex_llm_functions,
                )
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "To use the Google Vertex language models, "
                    "you need to install the `vertexai` package "
                    "and authenticate with Google Cloud cli."
                    "Run `pip install vertexai`."
                ) from e
            (
                self.llm_function,
                self.llm_async_function,
            ) = make_google_vertex_llm_functions(self.config)
        elif self.config.LLM_API_TYPE == ApiType.GOOGLE_AI_STUDIO:
            try:
                from .llm.google_genai import (
                    make_llm_functions as make_google_genai_llm_functions,
                )
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "To use the Google Gemini language models via AI Studio, "
                    "you need to install the `google-generativeai` package. "
                    "Run `pip install google-generativeai`."
                ) from e
            (
                self.llm_function,
                self.llm_async_function,
            ) = make_google_genai_llm_functions(self.config)
        else:
            self.llm_function, self.llm_async_function = make_openai_llm_functions(
                self.config
            )

    def init_similarity_search(self):
        if find_spec("chromadb") is not None:
            from .embedding_db.chromadb import ChromaEmbeddingDB

            self.texts = ChromaEmbeddingDB(self.config)


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

_env: Env | None = None


def env() -> Env:
    """Returns the current MicroCore environment"""
    return _env or Env()


def config() -> Config:
    """Resolve current configuration"""
    return env().config
