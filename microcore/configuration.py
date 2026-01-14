import json
import logging
import os
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Union, Callable

import dotenv
from colorama import Fore

_MISSING = object()

TRUE_VALUES = ["1", "TRUE", "YES", "ON", "ENABLED", "Y", "+"]
"""@private"""

PRINT_STREAM = "print_stream"
"""Logging method ID, value for 'config.use_logging'"""

DEFAULT_LOCAL_ENV_FILE = ".env"


def from_env(default=None, dtype=None):
    """
    Provides default value for the configuration dataclass
    from the environment variable with the name equal to field name in upper case"""
    return field(
        default=_MISSING, metadata=dict(_from_env=True, _default=default, _dtype=dtype)
    )


def get_bool_from_env(env_var: str, default: bool | None = False) -> bool | None:
    """Convert value of environment variable to boolean"""
    if env_var not in os.environ:
        return default
    return os.getenv(env_var, str(default)).upper() in TRUE_VALUES


def get_object_from_env(env_var: str, dtype: type, default: Any = None):
    val_from_env = os.getenv(env_var, _MISSING)  # pylint: disable=W1508
    if isinstance(val_from_env, str):
        val_from_env = val_from_env.strip()
        if val_from_env:
            try:
                val_from_env = json.loads(val_from_env.strip())
                assert isinstance(
                    val_from_env, dtype
                ), f"Expected {dtype.__name__}, got {type(val_from_env).__name__}"
            except (json.JSONDecodeError, AssertionError) as e:
                raise LLMConfigError(
                    f"Invalid value in environment variable '{env_var}'. "
                    f"Expected: JSON {dtype.__name__}, received: '{val_from_env}'"
                ) from e
        else:
            val_from_env = _MISSING
    if val_from_env is _MISSING:
        if default is None:  # instead of default factory
            default = dtype()
        val_from_env = default
    return val_from_env


class ApiType(str, Enum):
    """LLM API types"""

    OPEN_AI = "open_ai"
    AZURE = "azure"
    """See https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models"""
    ANYSCALE = "anyscale"
    """See https://www.anyscale.com/endpoints"""
    DEEP_INFRA = "deep_infra"
    """List of text generation models: https://deepinfra.com/models?type=text-generation"""
    ANTHROPIC = "anthropic"
    GOOGLE_VERTEX_AI = "google_vertex_ai"  # @Deprecated
    GOOGLE_AI_STUDIO = "google_ai_studio"  # @Deprecated
    GOOGLE = "google"  # new Google SDK

    # Local models
    FUNCTION = "function"
    TRANSFORMERS = "transformers"
    NONE = "none"

    @staticmethod
    def is_local(api_type: str) -> bool:
        return api_type in (ApiType.FUNCTION, ApiType.TRANSFORMERS, ApiType.NONE)

    def __str__(self):
        return self.value


class EmbeddingDbType(str, Enum):
    CHROMA = "chroma"
    QDRANT = "qdrant"
    NONE = ""

    def __str__(self):
        return self.value


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
                fp = self.DOT_ENV_FILE
                if fp and "~" in str(fp):
                    fp = Path(fp).expanduser()
                dotenv.load_dotenv(
                    override=True,
                    dotenv_path=fp
                )
            if not self.DOT_ENV_FILE:
                _default_dotenv_loaded = True

    def _update_from_env(self):
        for f in fields(self):
            if f.metadata.get("_from_env") and getattr(self, f.name) is _MISSING:
                env_name = f.name.upper()
                default = f.metadata["_default"]
                dtype = f.metadata.get("_dtype")
                if dtype is bool:
                    val_from_env = get_bool_from_env(env_name, default)
                elif dtype in [dict, list]:
                    val_from_env = get_object_from_env(env_name, dtype, default)
                else:
                    val_from_env = os.getenv(env_name, default)

                setattr(self, f.name, val_from_env)

    def __iter__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            yield f.name, value


@dataclass
class _OpenAIEnvVars:
    # OS Environment variables expected by OpenAI library
    # Will be used as defaults for LLM
    # @todo: implement lib_defaults to take default values from openai lib if available
    OPENAI_API_TYPE: str = from_env(ApiType.OPEN_AI)
    OPENAI_API_KEY: str = from_env("")
    OPENAI_API_BASE: str = from_env("https://api.openai.com/v1")
    OPENAI_API_VERSION: str = from_env()


@dataclass
class _AnthropicEnvVars:
    ANTHROPIC_API_KEY: str = from_env()


@dataclass
class _GoogleVertexAiEnvVars:
    """@deprecated, use _GoogleGenAiEnvVars instead"""
    GOOGLE_VERTEX_ACCESS_TOKEN: str = from_env()
    GOOGLE_VERTEX_PROJECT_ID: str = from_env()
    GOOGLE_VERTEX_LOCATION: str = from_env()
    GOOGLE_VERTEX_GCLOUD_AUTH: bool = from_env(dtype=bool)

    GOOGLE_VERTEX_RESPONSE_VALIDATION: bool = from_env(dtype=bool, default=False)
    GOOGLE_GEMINI_SAFETY_SETTINGS: dict = from_env(dtype=dict)


@dataclass
class _GoogleGenAiEnvVars:
    # see https://docs.cloud.google.com/docs/authentication/application-default-credentials
    GOOGLE_CLOUD_SERVICE_ACCOUNT: str = from_env()  # # file path (standard GCP name)
    GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON: str = from_env()  # JSON string content
    # see https://googleapis.github.io/python-genai/
    GOOGLE_CLOUD_PROJECT_ID: str = from_env()
    GOOGLE_CLOUD_LOCATION: str = from_env()
    GOOGLE_GENAI_USE_VERTEXAI: bool | None = from_env(default=None, dtype=bool)


@dataclass
class LLMConfig(
    BaseConfig,
    _OpenAIEnvVars,
    _AnthropicEnvVars,
    _GoogleVertexAiEnvVars,
    _GoogleGenAiEnvVars,
):
    """LLM configuration"""

    LLM_API_TYPE: str = from_env()
    """
    See `ApiType`.
    To use services that is not listed in `ApiType`,
    but provides OpenAPI interface, use `ApiType.OPEN_AI`"""

    LLM_API_KEY: str = from_env("")
    LLM_API_BASE: str = from_env()
    """Base URL for the LLM API, e.g. https://api.openai.com/v1"""

    LLM_API_VERSION: str = from_env()
    LLM_DEPLOYMENT_ID: str = from_env()
    """Required by `ApiType.AZURE`"""

    MODEL: str = from_env()
    """Language model name"""

    TIKTOKEN_ENCODING: str = from_env()
    """Will enforce using specific encoding for token size measurement"""

    LLM_DEFAULT_ARGS: dict = from_env(dtype=dict)
    """
    You may specify here default arguments for the LLM API calls,
     i. e. temperature, max_tokens, etc.
     """

    AZURE_DEPLOYMENT_ID: str = from_env()

    INFERENCE_FUNC: Union[Callable, str] = from_env()
    """Inference function for local models"""
    CHAT_MODE: bool = from_env(dtype=bool)
    """Is it a chat or completion model"""
    INIT_PARAMS: dict = from_env(dtype=dict)
    """Custom initialization parameters for the model"""

    HIDDEN_OUTPUT_BEGIN: str = from_env()
    HIDDEN_OUTPUT_END: str = from_env()
    """Remove <think>...</think> from LLM response for models like DeepSeek R1"""
    CALLBACKS: list[Callable] = field(default_factory=list)

    VALIDATE_CONFIG: bool = from_env(dtype=bool, default=True)

    def __post_init__(self):
        super().__post_init__()
        self._init_llm_options()
        self.VALIDATE_CONFIG and self.validate()

    def uses_local_model(self) -> bool:
        return ApiType.is_local(self.LLM_API_TYPE)

    def hiding_output(self) -> bool:
        return bool(self.HIDDEN_OUTPUT_BEGIN and self.HIDDEN_OUTPUT_END)

    def _init_llm_options(self):
        if self.INFERENCE_FUNC:
            if not self.LLM_API_TYPE:
                self.LLM_API_TYPE = ApiType.FUNCTION
        if self.uses_local_model():
            return

        # Use defaults from ENV variables expected by OpenAI API
        self.LLM_API_TYPE = self.LLM_API_TYPE or self.OPENAI_API_TYPE

        if self.LLM_API_TYPE == ApiType.AZURE:
            self.LLM_API_VERSION = self.LLM_API_VERSION or self.OPENAI_API_VERSION
            self.LLM_DEPLOYMENT_ID = self.LLM_DEPLOYMENT_ID or self.AZURE_DEPLOYMENT_ID
        elif self.LLM_API_TYPE in (ApiType.GOOGLE_AI_STUDIO, ApiType.GOOGLE):
            self.MODEL = self.MODEL or "gemini-2.5-pro"
        elif self.LLM_API_TYPE == ApiType.GOOGLE_VERTEX_AI:
            self.MODEL = self.MODEL or "gemini-2.5-pro"
            if self.GOOGLE_VERTEX_GCLOUD_AUTH is None:
                self.GOOGLE_VERTEX_GCLOUD_AUTH = get_bool_from_env(
                    "GOOGLE_VERTEX_GCLOUD_AUTH", not self.GOOGLE_VERTEX_ACCESS_TOKEN
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
            self.LLM_API_VERSION = self.LLM_API_VERSION or self.OPENAI_API_VERSION
            self.MODEL = self.MODEL or "gpt-3.5-turbo"

    def _validate_local_llm(self):
        if self.CHAT_MODE is None:
            logging.warning(
                "When using local models, "
                "(bool)CHAT_MODE configuration option should be explicitly set"
            )
        if self.LLM_API_TYPE == ApiType.FUNCTION:
            if not self.INFERENCE_FUNC:
                raise LLMConfigError(
                    "INFERENCE_FUNC should be provided for local models"
                )
        elif self.LLM_API_TYPE == ApiType.TRANSFORMERS:
            if not self.MODEL:
                raise LLMConfigError(
                    "MODEL should be provided for local transformers models"
                )

    def validate(self):
        """
        Validate LLM configuration

        Raises:
            LLMConfigError
        """
        if self.LLM_API_TYPE == ApiType.NONE:
            return
        if self.uses_local_model():
            self._validate_local_llm()
            return
        if self.INFERENCE_FUNC:
            raise LLMConfigError(
                "INFERENCE_FUNC should be provided only for local models"
            )
        if self.LLM_API_TYPE == ApiType.GOOGLE_VERTEX_AI:
            if (
                not self.GOOGLE_VERTEX_ACCESS_TOKEN
                and not self.GOOGLE_VERTEX_GCLOUD_AUTH
            ):
                raise LLMConfigError(
                    "GOOGLE_VERTEX_ACCESS_TOKEN should be provided "
                    "or GOOGLE_VERTEX_GCLOUD_AUTH should be enabled"
                )
        elif self.LLM_API_TYPE == ApiType.GOOGLE:
            if (
                not self.GOOGLE_CLOUD_SERVICE_ACCOUNT
                and not self.GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON
                and not self.LLM_API_KEY
            ):
                raise LLMCredentialError(
                    "Google API credentials not configured. Provide one of: "
                    "GOOGLE_CLOUD_SERVICE_ACCOUNT (path to service account JSON file), "
                    "GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON (service account JSON content), "
                    "or LLM_API_KEY (for Gemini Developer API only, not Vertex AI)"
                )
        else:
            if not self.LLM_API_KEY:
                raise LLMApiKeyError()
            if self.LLM_API_TYPE == ApiType.AZURE:
                if not self.LLM_API_BASE:
                    raise LLMApiBaseError()
                if not self.LLM_DEPLOYMENT_ID:
                    raise LLMApiDeploymentIdError()
                if not self.LLM_API_VERSION:
                    raise LLMApiVersionError()

    def describe(self, return_dict=False):
        """
        Informal description of the configuration
        """
        prev_env = os.environ.copy()
        os.environ.clear()
        default = Config(LLM_API_TYPE=ApiType.NONE, USE_DOT_ENV=False)
        os.environ.update(prev_env)
        data = {
            k.lower().replace("llm_", ""): v
            for k, v in dict(self).items()
            if v is not None and v != getattr(default, k) and k != "USE_DOT_ENV"
        }
        for k, v in data.items():
            if "_key" in k and isinstance(v, str):
                if len(v) <= 3:
                    continue
                data[k] = v[: 1 if len(v) <= 12 else 3] + "****" + v[-2:]
        if return_dict:
            return data

        print("Config:")
        for k, v in data.items():
            print(f"  {k}: {Fore.GREEN}{v}{Fore.RESET}")
        return None


class LLMConfigError(ValueError):
    """LLM configuration error"""
    BASE_MSG = "LLM configuration error"

    def __init__(self, message: str = None):
        message = f"{self.BASE_MSG}: {message}"
        super().__init__(message)


class LLMCredentialError(LLMConfigError):
    def __init__(self, message: str = None):
        message = message or "LLM credentials are invalid"
        super().__init__(message)


class LLMApiKeyError(LLMCredentialError):
    """LLM API KEY error"""

    def __init__(self, message: str = None):
        message = message or "LLM_API_KEY is absent"
        super().__init__(message)


class LLMApiBaseError(LLMConfigError):
    """LLM API BASE error"""

    def __init__(self, message: str = None):
        message = message or "LLM_API_BASE is required for using Azure models"
        super().__init__(message)


class LLMApiDeploymentIdError(LLMConfigError):
    """LLM API DEPLOYMENT ID error"""

    def __init__(self, message: str = None):
        message = message or "LLM_DEPLOYMENT_ID is required for using Azure models"
        super().__init__(message)


class LLMApiVersionError(LLMConfigError):
    """LLM API VERSION error"""

    def __init__(self, message: str = None):
        message = message or "LLM_API_VERSION is required for using Azure models"
        super().__init__(message)


@dataclass
class Config(LLMConfig):
    """MicroCore configuration"""

    USE_LOGGING: bool = from_env(default=False)
    """Whether to use logging or not, see `microcore.use_logging`"""

    PROMPT_TEMPLATES_PATH: str | Path = from_env("tpl")
    """Path to the folder with prompt templates, ./tpl by default"""

    STORAGE_PATH: str | Path = from_env("storage")
    """Path to the folder with file storage, ./storage by default"""

    STORAGE_DEFAULT_FILE_EXT: str = from_env(default="")

    EMBEDDING_DB_FOLDER: str = from_env(default="embedding_db")
    """Folder within microcore.config.Config.STORAGE_PATH for storing embeddings"""

    EMBEDDING_DB_FUNCTION: Any = from_env()

    EMBEDDING_DB_ALLOW_DUPLICATES: bool = from_env(dtype=bool, default=False)

    EMBEDDING_DB_HOST: str = from_env(default=None)

    EMBEDDING_DB_PORT: str = from_env(default=None)

    EMBEDDING_DB_TYPE: str = from_env(EmbeddingDbType.CHROMA)

    EMBEDDING_DB_TIMEOUT: int = from_env(default=10 * 60)

    EMBEDDING_DB_SIZE: int = from_env(default=0)
    """Used with Qdrant"""

    DEFAULT_ENCODING: str = from_env("utf-8")
    """Used in file system operations, utf-8 by default"""

    JINJA2_AUTO_ESCAPE: bool = from_env(dtype=bool, default=False)

    ELEVENLABS_API_KEY: str = from_env()

    TEXT_TO_SPEECH_PATH: str | Path = from_env()
    """Path to the folder with generated voice files"""

    MAX_CONCURRENT_TASKS: int = from_env(default=None)

    SAVE_MEMORY: bool = from_env(dtype=bool, default=False)
    """
    Some additional data will not be collected:
      - LLMResponse objects will not contain the links to the prompt field
    """

    AI_SYNTAX_FUNCTION_NAME_FIELD: str = from_env(default="call")

    DEFAULT_AI_FUNCTION_SYNTAX: str = from_env(default="json")

    JINJA2_GLOBALS: dict = from_env(dtype=dict)

    MCP_SERVERS: list = from_env(dtype=list)

    INTERACTIVE_SETUP: bool = field(default=False)
    """Whether to run interactive setup if configuration is not valid."""

    def __post_init__(self):
        # Enforce using .env files if interactive_setup is enabled
        if self.INTERACTIVE_SETUP and not self.DOT_ENV_FILE:
            self.DOT_ENV_FILE = DEFAULT_LOCAL_ENV_FILE
        try:
            super().__post_init__()
        except LLMConfigError as e:
            if self.INTERACTIVE_SETUP and not os.path.exists(self.DOT_ENV_FILE):
                from .interactive_setup import interactive_setup
                orig_logging = self.USE_LOGGING  # avoid rewr. logging settings
                config = interactive_setup(self.DOT_ENV_FILE)
                if not config:
                    raise e
                self.__dict__.update(config.__dict__)
                self.USE_LOGGING = orig_logging
            else:
                raise e
        if self.TEXT_TO_SPEECH_PATH is None:
            self.TEXT_TO_SPEECH_PATH = Path(self.STORAGE_PATH) / "voicing"

    def validate(self):
        super().validate()
        if self.EMBEDDING_DB_TYPE == EmbeddingDbType.QDRANT:
            if not self.EMBEDDING_DB_SIZE:
                raise LLMConfigError(
                    "EMBEDDING_DB_SIZE is required configuration parameter for Qdrant"
                )
            if not self.EMBEDDING_DB_HOST:
                raise LLMConfigError(
                    "EMBEDDING_DB_HOST is required configuration parameter for Qdrant"
                )
            if not self.EMBEDDING_DB_FUNCTION:
                raise LLMConfigError(
                    "EMBEDDING_DB_FUNCTION is required configuration parameter for Qdrant"
                )
