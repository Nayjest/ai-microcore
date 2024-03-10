import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import dotenv

_MISSING = object()

TRUE_VALUES = ["1", "TRUE", "YES", "ON", "ENABLED", "Y", "+"]
"""@private"""


def from_env(default=None):
    """
    Provides default value for the configuration dataclass
    from the environment variable with the name equal to field name in upper case"""
    return field(default=_MISSING, metadata=dict(_from_env=True, _default=default))


def get_bool_from_env(env_var: str, default: bool = False) -> bool:
    """Convert value of environment variable to boolean"""
    return os.getenv(env_var, str(default)).upper() in TRUE_VALUES


class ApiType:
    """LLM API types"""

    OPEN_AI = "open_ai"
    AZURE = "azure"
    """See https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models"""
    LLM = "llm"
    ANYSCALE = "anyscale"
    """See https://www.anyscale.com/endpoints"""
    DEEP_INFRA = "deep_infra"
    """List of text generation models: https://deepinfra.com/models?type=text-generation"""
    ANTHROPIC = "anthropic"
    GOOGLE_VERTEX_AI = "google_vertex_ai"


_default_dotenv_loaded = False


@dataclass
class BaseConfig:
    """Base class for configuration dataclasses"""

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
class _OpenAIEnvVars:
    # OS Environment variables expected by OpenAI library
    # Will be used as defaults for LLM
    # @todo: implement lib_defaults to take default values from openai lib if available
    OPENAI_API_TYPE: str = from_env(ApiType.OPEN_AI)
    OPENAI_API_KEY: str = from_env()
    OPENAI_API_BASE: str = from_env("https://api.openai.com/v1")
    OPENAI_API_VERSION: str = from_env()


@dataclass
class _AnthropicEnvVars:
    ANTHROPIC_API_KEY: str = from_env()


@dataclass
class _GoogleVertexAiEnvVars:
    GOOGLE_VERTEX_ACCESS_TOKEN: str = from_env()
    GOOGLE_VERTEX_PROJECT_ID: str = from_env()
    GOOGLE_VERTEX_LOCATION: str = from_env()
    GOOGLE_VERTEX_GCLOUD_AUTH: bool = None
    GOOGLE_GEMINI_RESPONSE_VALIDATION: bool = None
    GOOGLE_GEMINI_SAFETY_SETTINGS: dict = None


@dataclass
class LLMConfig(BaseConfig, _OpenAIEnvVars, _AnthropicEnvVars, _GoogleVertexAiEnvVars):
    """LLM configuration"""

    LLM_API_TYPE: str = from_env()
    """
    See `ApiType`.
    To use services that is not listed in `ApiType`,
    but provides OpenAPI interface, use `ApiType.OPEN_AI`"""

    LLM_API_KEY: str = from_env()
    LLM_API_BASE: str = from_env()
    """Base URL for the LLM API, e.g. https://api.openai.com/v1"""

    LLM_API_VERSION: str = from_env()
    LLM_DEPLOYMENT_ID: str = from_env()
    """Required by `ApiType.AZURE`"""

    MODEL: str = from_env()
    """Language model name"""

    LLM_DEFAULT_ARGS: dict = field(default_factory=dict)
    """
    You may specify here default arguments for the LLM API calls,
     i. e. temperature, max_tokens, etc.
     """

    AZURE_DEPLOYMENT_ID: str = from_env()

    def __post_init__(self):
        super().__post_init__()
        self._init_llm_options()
        self.validate()

    def _init_llm_options(self):
        # Use defaults from ENV variables expected by OpenAI API
        self.LLM_API_TYPE = self.LLM_API_TYPE or self.OPENAI_API_TYPE
        self.LLM_API_VERSION = self.LLM_API_VERSION or self.OPENAI_API_VERSION

        if self.LLM_API_TYPE == ApiType.AZURE:
            self.LLM_DEPLOYMENT_ID = self.LLM_DEPLOYMENT_ID or self.AZURE_DEPLOYMENT_ID
        elif self.LLM_API_TYPE == ApiType.GOOGLE_VERTEX_AI:
            self.MODEL = self.MODEL or "gemini-1.0-pro"
            if self.GOOGLE_VERTEX_GCLOUD_AUTH is None:
                self.GOOGLE_VERTEX_GCLOUD_AUTH = get_bool_from_env(
                    "GOOGLE_VERTEX_GCLOUD_AUTH", not self.GOOGLE_VERTEX_ACCESS_TOKEN
                )
            if self.GOOGLE_GEMINI_RESPONSE_VALIDATION is None:
                self.GOOGLE_VERTEX_RESPONSE_VALIDATION = get_bool_from_env(
                    "GOOGLE_GEMINI_RESPONSE_VALIDATION", False
                )
        elif self.LLM_API_TYPE == ApiType.ANYSCALE:
            self.LLM_API_BASE = (
                self.LLM_API_BASE or "https://api.endpoints.anyscale.com/v1"
            )
            self.MODEL = self.MODEL or "meta-llama/Llama-2-70b-chat-hf"
        elif self.LLM_API_TYPE == ApiType.DEEP_INFRA:
            self.LLM_API_BASE = (
                self.LLM_API_BASE or "https://api.deepinfra.com/v1/openai"
            )
            self.MODEL = self.MODEL or "meta-llama/Llama-2-70b-chat-hf"
        elif self.LLM_API_TYPE == ApiType.ANTHROPIC:
            self.LLM_API_BASE = self.LLM_API_BASE or "https://api.anthropic.com/"
            self.MODEL = self.MODEL or "claude-3-opus-20240229"
            self.LLM_API_KEY = self.LLM_API_KEY or self.ANTHROPIC_API_KEY
        else:
            self.LLM_API_BASE = self.LLM_API_BASE or self.OPENAI_API_BASE
            self.LLM_API_KEY = self.LLM_API_KEY or self.OPENAI_API_KEY

        self.MODEL = self.MODEL or "gpt-3.5-turbo"

    def validate(self):
        """
        Validate LLM configuration

        Raises:
            LLMConfigError
        """
        if self.LLM_API_TYPE == ApiType.GOOGLE_VERTEX_AI:
            if not self.GOOGLE_VERTEX_ACCESS_TOKEN and not self.GOOGLE_VERTEX_GCLOUD_AUTH:
                raise LLMConfigError(
                    "LLM configuration error: "
                    "GOOGLE_VERTEX_ACCESS_TOKEN should be provided "
                    "or GOOGLE_VERTEX_GCLOUD_AUTH should be enabled"
                )
        else:
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
    """LLM configuration error"""


@dataclass
class Config(LLMConfig):
    """MicroCore configuration"""

    USE_LOGGING: bool = False
    """Whether to use logging or not, see `microcore.use_logging`"""

    PROMPT_TEMPLATES_PATH: str | Path = from_env("tpl")
    """Path to the folder with prompt templates, ./tpl by default"""

    STORAGE_PATH: str | Path = from_env("storage")
    """Path to the folder with file storage, ./storage by default"""

    STORAGE_DEFAULT_FILE_EXT: str = field(default="")

    EMBEDDING_DB_FOLDER: str = "embedding_db"
    """Folder within microcore.config.Config.STORAGE_PATH for storing embeddings"""

    EMBEDDING_DB_FUNCTION: Any = from_env()

    EMBEDDING_DB_ALLOW_DUPLICATES: bool = False

    DEFAULT_ENCODING: str = from_env("utf-8")
    """Used in file system operations, utf-8 by default"""

    JINJA2_AUTO_ESCAPE: bool = None

    ELEVENLABS_API_KEY: str = from_env()

    TEXT_TO_SPEECH_PATH: str | Path = from_env()

    def __post_init__(self):
        if self.JINJA2_AUTO_ESCAPE is None:
            self.JINJA2_AUTO_ESCAPE = get_bool_from_env("JINJA2_AUTO_ESCAPE", False)
        super().__post_init__()
        if self.TEXT_TO_SPEECH_PATH is None:
            self.TEXT_TO_SPEECH_PATH = Path(self.STORAGE_PATH) / "voicing"
