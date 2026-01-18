"""LLM API types and platforms definitions"""
from enum import Enum, EnumMeta
from typing import Iterable, Optional, Union


class SafeEnumMeta(EnumMeta):
    """A metaclass for Enums that handles `in` checks safely for Python < 3.12"""
    def __contains__(cls, item):
        try:
            return super().__contains__(item)
        except TypeError:
            return False


class SafeStrEnum(str, Enum, metaclass=SafeEnumMeta):
    """String Enum with safe `in` checks for Python < 3.12"""
    def __str__(self):
        return self.value


class ApiType(SafeStrEnum):
    """LLM API types"""

    OPENAI = "openai"
    AZURE = "azure"  # @Deprecated in favor of LLM_API_PLATFORM parameter
    """See https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models"""
    ANYSCALE = "anyscale"  # @Deprecated in favor of LLM_API_PLATFORM parameter
    """See https://www.anyscale.com/endpoints"""
    DEEP_INFRA = "deep_infra"  # @Deprecated in favor of LLM_API_PLATFORM parameter
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
    def is_local(api_type: Union[str, "ApiType"]) -> bool:
        """
        Check if the given API type is for local models (embedded in the application).
        Args:
            api_type (str|ApiType): The API type to check.
        Returns:
            bool: True if the API type is local, False otherwise.
        """
        return api_type in (ApiType.FUNCTION, ApiType.TRANSFORMERS, ApiType.NONE)

    @staticmethod
    def major_remote() -> list["ApiType"]:
        """Get list of major remote API types"""
        return [ApiType.OPENAI, ApiType.GOOGLE, ApiType.ANTHROPIC]

    @staticmethod
    def label_for(api_type: Union[str, "ApiType"]) -> str:
        """
        Get human-readable label for the API type
        Args:
            api_type (str|ApiType): The API type to get the label for.
        Returns:
            str: The human-readable label for the API type.
        """
        labels = {
            ApiType.OPENAI: "OpenAI",
            ApiType.AZURE: "Azure OpenAI",
            ApiType.FUNCTION: "Local Function",
            ApiType.TRANSFORMERS: "Local Transformers",
            ApiType.NONE: "No LLM",
        }
        return labels.get(api_type, str(api_type).replace('_', ' ').title())

    @staticmethod
    def labels(api_types: Iterable = None) -> dict:
        """
        Get human-readable labels for the API types.
        Args:
            api_types (Iterable, optional): The API types to get labels for.
                If None, all ApiType values are used.
        Returns:
            dict: A dictionary mapping ApiType to its human-readable label.
        """
        if api_types is None:
            api_types = ApiType
        return {api_type: ApiType.label_for(api_type) for api_type in api_types}

    def __str__(self):
        return self.value


class ModelPreset(SafeStrEnum):
    """Model presets."""
    HIGH_END = "high-end"
    LOW_END = "low-end"


class ApiPlatform(SafeStrEnum):
    """LLM platforms / Inference providers"""
    # OpenAI Compatible
    OPENAI = "openai"
    AZURE = "azure"
    ANYSCALE = "anyscale"
    DEEPINFRA = "deepinfra"
    MISTRAL = "mistral"
    FIREWORKS = "fireworks"
    DEEPSEEK = "deepseek"
    XAI = "xai"
    CEREBRAS = "cerebras"
    GROQ = "groq"
    COHERE = "cohere"
    TOGETHER_AI = "together_ai"
    OPENROUTER = "openrouter"
    PERPLEXITY = "perplexity"

    # Google
    GOOGLE_AI_STUDIO = "google_ai_studio"
    GOOGLE_VERTEX_AI = "google_vertex_ai"

    # Anthropic
    ANTHROPIC = "anthropic"

    @staticmethod
    def for_api_type(api_type: ApiType) -> list["ApiPlatform"]:
        """Get list of ApiPlatforms for the given ApiType"""
        return API_PLATFORMS_BY_API_TYPE.get(api_type, [])

    def api_type(self) -> Optional[ApiType]:
        """
        Get ApiType for the platform.
        @todo: handle multiple ApiTypes per platform
        """
        for api_type, platforms in API_PLATFORMS_BY_API_TYPE.items():
            if self in platforms:
                return api_type
        return None

    @staticmethod
    def label_for(platform: Union[str, "ApiPlatform"]) -> str:
        """
        Get human-readable label for the platform.
        Args:
            platform (str|ApiPlatform): The platform to get the label for.
        Returns:
            str: The human-readable label for the platform.
        """
        return _API_PLATFORM_CUSTOM_LABELS.get(platform, str(platform).replace('_', ' ').title())

    @staticmethod
    def labels(platforms: Iterable = None) -> dict:
        """
        Get human-readable labels for the platforms.
        Args:
            platforms (Iterable, optional): The platforms to get labels for.
                If None, all ApiPlatform values are used.
        Returns:
            dict: A dictionary mapping ApiPlatform to its human-readable label.
        """
        if platforms is None:
            platforms = ApiPlatform
        return {platform: ApiPlatform.label_for(platform) for platform in platforms}

    def default_api_base(self) -> Optional[str]:
        """Get default API base URL for the platform"""
        return LLM_API_BASE_URLS.get(self.api_type(), {}).get(self)


_API_PLATFORM_CUSTOM_LABELS: dict[ApiPlatform, str] = {
    ApiPlatform.OPENAI: "OpenAI",
    ApiPlatform.GOOGLE_AI_STUDIO: "Google AI Studio",
    ApiPlatform.GOOGLE_VERTEX_AI: "Google Vertex AI",
    ApiPlatform.DEEPINFRA: "DeepInfra",
    ApiPlatform.FIREWORKS: "Fireworks AI",
    ApiPlatform.DEEPSEEK: "DeepSeek",
    ApiPlatform.XAI: "xAI",
    ApiPlatform.GROQ: "GROQ",
    ApiPlatform.TOGETHER_AI: "Together AI",
    ApiPlatform.OPENROUTER: "OpenRouter",
}
ANTHROPIC_API_PLATFORMS: list[ApiPlatform] = [ApiPlatform.ANTHROPIC]
GOOGLE_API_PLATFORMS: list[ApiPlatform] = [
    ApiPlatform.GOOGLE_AI_STUDIO,
    ApiPlatform.GOOGLE_VERTEX_AI
]
OPENAI_API_PLATFORMS: list[ApiPlatform] = [
    ApiPlatform.OPENAI,
    ApiPlatform.AZURE,
    ApiPlatform.ANYSCALE,
    ApiPlatform.DEEPINFRA,
    ApiPlatform.MISTRAL,
    ApiPlatform.FIREWORKS,
    ApiPlatform.DEEPSEEK,
    ApiPlatform.XAI,
    ApiPlatform.CEREBRAS,
    ApiPlatform.GROQ,
    ApiPlatform.COHERE,
    ApiPlatform.TOGETHER_AI,
    ApiPlatform.OPENROUTER,
    ApiPlatform.PERPLEXITY,
]
API_PLATFORMS_BY_API_TYPE: dict[ApiType, list] = {
    ApiType.OPENAI: OPENAI_API_PLATFORMS,
    ApiType.GOOGLE: GOOGLE_API_PLATFORMS,
    ApiType.ANTHROPIC: ANTHROPIC_API_PLATFORMS,
}
LLM_API_BASE_URLS = {
    ApiType.OPENAI: {
        ApiPlatform.OPENAI: "https://api.openai.com/v1",
        # ApiPlatform.AZURE: "",  # Azure base URL is custom per user
        ApiPlatform.ANYSCALE: "https://api.endpoints.anyscale.com/v1",
        ApiPlatform.DEEPINFRA: "https://api.deepinfra.com/v1/openai",
        ApiPlatform.MISTRAL: "https://api.mistral.ai/v1",
        ApiPlatform.FIREWORKS: "https://api.fireworks.ai/inference/v1",
        ApiPlatform.DEEPSEEK: "https://api.deepseek.com/v1",
        ApiPlatform.XAI: "https://api.x.ai/v1",
        ApiPlatform.CEREBRAS: "https://api.cerebras.ai/v1",
        ApiPlatform.GROQ: "https://api.groq.com/openai/v1",
        ApiPlatform.COHERE: "https://api.cohere.ai/compatibility/v1",
        ApiPlatform.TOGETHER_AI: "https://api.together.xyz/v1",
        ApiPlatform.OPENROUTER: "https://openrouter.ai/api/v1",
        ApiPlatform.PERPLEXITY: "https://api.perplexity.ai",
    },
    ApiType.ANTHROPIC: {
        ApiPlatform.ANTHROPIC: "https://api.anthropic.com/",
    }
}
DEFAULT_PLATFORMS = {
    ApiType.OPENAI: ApiPlatform.OPENAI,
    ApiType.GOOGLE: ApiPlatform.GOOGLE_AI_STUDIO,
    ApiType.ANTHROPIC: ApiPlatform.ANTHROPIC,
}

HIGH_END_MODELS: dict[ApiPlatform, str] = {
    ApiPlatform.OPENAI: "gpt-5.2",
    ApiPlatform.ANTHROPIC: "claude-opus-4-5",
    ApiPlatform.GOOGLE_AI_STUDIO: "gemini-2.5-pro",
    ApiPlatform.GOOGLE_VERTEX_AI: "gemini-2.5-pro",
    ApiPlatform.MISTRAL: "mistral-large-latest",
    ApiPlatform.XAI: "grok-4",  # I/O: $3 $15 /M tokens
    ApiPlatform.DEEPSEEK: "deepseek-chat",
    ApiPlatform.CEREBRAS: "gpt-oss-120b",  # I/O: $0.35/$0.75 /M tokens
    ApiPlatform.GROQ: "openai/gpt-oss-120b",  # I/O: $0.15/$0.60 /M tokens
    ApiPlatform.FIREWORKS: "accounts/fireworks/models/kimi-k2-thinking",  # I/O: $0.60/$2.50 /M
    ApiPlatform.PERPLEXITY: "sonar-deep-research",  # I/O: $2/$8 /M tokens
}
LOW_END_MODELS: dict[ApiPlatform, str] = {
    ApiPlatform.OPENAI: "gpt-5-nano",
    ApiPlatform.ANTHROPIC: "claude-haiku-4-5",
    ApiPlatform.GOOGLE_AI_STUDIO: "gemini-2.5-flash-lite",
    ApiPlatform.GOOGLE_VERTEX_AI: "gemini-2.5-flash-lite",
    ApiPlatform.XAI: "grok-4-1-fast",  # I/O: $0.20 $0.50 /M tokens
    ApiPlatform.MISTRAL: "ministral-3b-2512",
    ApiPlatform.DEEPSEEK: "deepseek-chat",
    ApiPlatform.CEREBRAS: "llama3.1-8b",  # I/O: $0.10/M tokens
    ApiPlatform.GROQ: "llama-3.1-8b-instant",    # I/O: $0.05/$0.08 /M tokens
    ApiPlatform.FIREWORKS: "accounts/fireworks/models/gpt-oss-20b",  # I/O: $0.07 / $0.30 /M tokens
    ApiPlatform.PERPLEXITY: "sonar",  # I/O: $1/$1 /M tokens
}
MID_END_MODELS: dict[ApiPlatform, str] = {
    ApiPlatform.PERPLEXITY: "sonar-pro",  # I/O: $3/$15 /M tokens
}
MODEL_PRESETS: dict[ModelPreset, dict[ApiPlatform, str]] = {
    ModelPreset.HIGH_END: HIGH_END_MODELS,
    ModelPreset.LOW_END: LOW_END_MODELS,
}
DEFAULT_MODELS: dict[ApiPlatform, str] = HIGH_END_MODELS


def llm_api_base_required(api_type: ApiType, platform: ApiPlatform | None) -> bool:
    """
    Determine if the given API type and platform require an API base URL.
    """
    if api_type == ApiType.OPENAI:
        if platform == ApiPlatform.AZURE or not platform:
            return True
        if LLM_API_BASE_URLS[ApiType.OPENAI].get(platform) is None:
            return True
    return False


def llm_api_key_required(api_type: ApiType | str, platform: ApiPlatform | None) -> bool:
    """
    Determine if the given API type and platform require an API key.
    """
    if ApiType.is_local(api_type):
        return False
    if platform == ApiPlatform.GOOGLE_VERTEX_AI:
        return False
    return True


def platform_by_api_base(api_base: str) -> tuple[Optional[ApiType], Optional[ApiPlatform]]:
    """
    Determine ApiPlatform by the given API base URL.
    """
    for api_type, platforms in LLM_API_BASE_URLS.items():
        for platform, base_url in platforms.items():
            if api_base == base_url:
                return api_type, platform
    return None, None
