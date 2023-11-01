import os
from dataclasses import dataclass, field, fields
from pathlib import Path
import dotenv

from microcore.utils import get_bool_from_env

_MISSING = object()


def from_env(default=None):
    f = field(default=_MISSING, metadata=dict(_from_env=True, _default=default))
    return f


class ApiType:
    OPEN_AI = "open_ai"
    AZURE = "azure"
    LLM = "llm"
    ANYSCALE = "anyscale"
    DEEP_INFRA = "deep_infra"


_default_dotenv_loaded = False


@dataclass
class BaseConfig:
    USE_DOT_ENV: bool = None
    DOT_ENV_FILE: str | Path = None

    def __post_init__(self):
        self._dot_env_setup()
        self._update_from_env()

    def _dot_env_setup(self):
        global _default_dotenv_loaded

        if self.USE_DOT_ENV is None:
            self.USE_DOT_ENV = get_bool_from_env("USE_DOT_ENV", True)

        if self.USE_DOT_ENV:
            if self.DOT_ENV_FILE or not _default_dotenv_loaded:
                dotenv.load_dotenv(override=True, dotenv_path=self.DOT_ENV_FILE)
            if not self.DOT_ENV_FILE:
                _default_dotenv_loaded = True

    def _update_from_env(self):
        for f in fields(self):
            if f.metadata.get("_from_env") and getattr(self, f.name) is _MISSING:
                setattr(self, f.name, os.getenv(f.name.upper(), f.metadata["_default"]))


@dataclass
class OpenAIEnvVars:
    # OS Environment variables expected by OpenAI library
    # Will be used as defaults for LLM
    # @todo: implement lib_defaults to take default values from openai lib if available
    OPENAI_API_TYPE: str = from_env(ApiType.OPEN_AI)
    OPENAI_API_KEY: str = from_env()
    OPENAI_API_BASE: str = from_env("https://api.openai.com/v1")
    OPENAI_API_VERSION: str = from_env()


@dataclass
class LLMConfig(BaseConfig, OpenAIEnvVars):
    LLM_API_TYPE: str = from_env()
    LLM_API_KEY: str = from_env()
    LLM_API_BASE: str = from_env()
    LLM_API_VERSION: str = from_env()
    LLM_DEPLOYMENT_ID: str = from_env()
    MODEL: str = from_env()
    LLM_DEFAULT_ARGS: dict = field(default_factory=dict)
    AZURE_DEPLOYMENT_ID: str = from_env()

    def __post_init__(self):
        super().__post_init__()
        self._init_llm_options()

    def _init_llm_options(self):
        # Use defaults from ENV variables expected by OpenAI API
        self.LLM_API_TYPE = self.LLM_API_TYPE or self.OPENAI_API_TYPE
        self.LLM_API_KEY = self.LLM_API_KEY or self.OPENAI_API_KEY
        self.LLM_API_BASE = self.LLM_API_BASE or self.OPENAI_API_BASE
        self.LLM_API_VERSION = self.LLM_API_VERSION or self.OPENAI_API_VERSION

        if self.LLM_API_TYPE == ApiType.AZURE:
            self.LLM_DEPLOYMENT_ID = self.LLM_DEPLOYMENT_ID or self.AZURE_DEPLOYMENT_ID

        if self.LLM_API_TYPE == ApiType.ANYSCALE:
            self.LLM_API_BASE = (
                self.LLM_API_BASE or "https://api.endpoints.anyscale.com/v1"
            )
            self.MODEL = self.MODEL or "meta-llama/Llama-2-70b-chat-hf"

        if self.LLM_API_TYPE == ApiType.DEEP_INFRA:
            self.LLM_API_BASE = (
                self.LLM_API_BASE or "https://api.deepinfra.com/v1/openai"
            )
            self.MODEL = self.MODEL or "meta-llama/Llama-2-70b-chat-hf"

        self.MODEL = self.MODEL or "gpt-3.5-turbo"

    def validate(self):
        if not self.LLM_API_KEY:
            raise LLMConfigError("LLM configuration error: LLM_API_KEY is absent")
        if self.LLM_API_TYPE == ApiType.AZURE:
            if not self.LLM_API_BASE:
                raise LLMConfigError(
                    "LLM configuration error: "
                    "LLM_API_BASE is required for using Azure models"
                )
            if not self.LLM_DEPLOYMENT_ID:
                raise LLMConfigError(
                    "LLM configuration error: "
                    "LLM_DEPLOYMENT_ID is required for using Azure models"
                )
            if not self.LLM_API_VERSION:
                raise LLMConfigError(
                    "LLM configuration error: "
                    "LLM_API_VERSION is required for using Azure models"
                )


class LLMConfigError(ValueError):
    pass


@dataclass
class Config(LLMConfig):
    USE_LOGGING: bool = False

    PROMPT_TEMPLATES_PATH: str | Path = from_env("tpl")

    STORAGE_PATH: str | Path = from_env("storage")
    EMBEDDING_DB_FOLDER: str = "embedding_db"
    DEFAULT_ENCODING: str = from_env("utf-8")
