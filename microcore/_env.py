from dataclasses import dataclass, field
from importlib.util import find_spec
import jinja2

from .configuration import Config, ApiType
from . import AbstractEmbeddingDB
from .types import TplFunctionType, LLMAsyncFunctionType, LLMFunctionType
from .templating.jinja2 import make_jinja2_env, make_tpl_function
from .llm.openai_llm import make_llm_functions as make_openai_llm_functions


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

    def __post_init__(self):
        global _env
        _env = self
        self.init_templating()
        self.init_llm()
        if self.config.USE_LOGGING:
            from .logging import use_logging

            use_logging()
        self.init_similarity_search()

    def init_templating(self):
        self.jinja_env = make_jinja2_env(self)
        self.tpl_function = make_tpl_function(self)

    def init_llm(self):
        if self.config.LLM_API_TYPE == ApiType.ANTHROPIC:
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

_env: Env | None = None


def env() -> Env:
    """Returns the current MicroCore environment"""
    return _env or Env()


def config() -> Config:
    """Resolve current configuration"""
    return env().config
