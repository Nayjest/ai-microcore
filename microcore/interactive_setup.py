import os

from .configuration import (
    EmbeddingDbType,
    Config,
)
from .llm_backends import (
    ApiPlatform,
    ApiType,
    llm_api_base_required,
    llm_api_key_required,
)
from .ui import ask_choose, ask_non_empty, ask_yn, error, yellow, ask
from ._env import configure
from ._llm_functions import llm
from .utils import file_link


def prompt_api_type(
    question: str = "Which language model API should be used?",
    api_types: dict[ApiType, str] | list[str] = None
) -> ApiType:
    """
    Prompt user to choose an LLM API type.
    """
    return ask_choose(question, api_types or ApiType.labels(ApiType.major_remote()))


def prompt_api_platform(
    api_type: ApiType,
    question: str = "Select your LLM inference provider.",
) -> ApiPlatform | None:
    platforms: list = ApiPlatform.for_api_type(api_type)
    if len(platforms) == 1:
        return platforms[0]
    platform_labels = ApiPlatform.labels(platforms)
    if api_type == ApiType.OPENAI:
        platform_labels["other"] = "Other"

    result = ask_choose(question, platform_labels)
    if result == "other":
        return None
    return result


def prompt_api_key(
    question: str = "Enter API Key:",
) -> str:
    """
    Prompt user to enter an LLM API key.
    """
    return ask_non_empty(question).strip()


def prompt_model_name(
    question: str = "Enter model name:",
) -> str:
    """
    Prompt user to enter an LLM model name.
    """
    return ask_non_empty(question).strip()


def interactive_setup(
    file_path: str,
    defaults: dict = None,
    extras: dict | list = None,
) -> Config | None:
    """
    Start interactive CLI setup for LLM API configuration.
    Prompts user for configuration details such as API type, key, model name,
    and base URL. Tests the LLM API with a sample query and saves the configuration
    to a specified file if the user chooses to do so.
    Args:
        file_path (str): Path to the configuration file.
        defaults (dict, optional): Default configuration values.
            If provided, user will not be prompted for those values.
            Additional values for storing in the file can be added to defaults.
        extras (dict | list, optional): Additional configuration fields to prompt for.
    """
    raw_config = dict(defaults) if defaults else dict()
    if "LLM_API_TYPE" not in raw_config:
        raw_config["LLM_API_TYPE"] = prompt_api_type()
    api_type = raw_config["LLM_API_TYPE"]

    if "LLM_API_PLATFORM" not in raw_config:
        raw_config["LLM_API_PLATFORM"] = prompt_api_platform(api_type)
    platform = raw_config["LLM_API_PLATFORM"]

    if "LLM_API_BASE" not in raw_config and llm_api_base_required(api_type, platform):
        raw_config["LLM_API_BASE"] = ask(
            "API Base URL (may be empty for some API types): "
        ).strip()

    if platform == ApiPlatform.AZURE:
        if "LLM_DEPLOYMENT_ID" not in raw_config:
            raw_config["LLM_DEPLOYMENT_ID"] = ask_non_empty("Enter deployment ID:")
        if "LLM_API_VERSION" not in raw_config:
            raw_config["LLM_API_VERSION"] = ask_non_empty("Enter API version:")

    if "LLM_API_KEY" not in raw_config and llm_api_key_required(api_type, platform):
        raw_config["LLM_API_KEY"] = prompt_api_key()

    if platform == ApiPlatform.GOOGLE_VERTEX_AI:
        if "GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON" not in raw_config:
            raw_config["GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON"] = ask_non_empty(
                "Enter Google Cloud Service Account JSON:"
            )
        if "GOOGLE_CLOUD_PROJECT_ID" not in raw_config:
            raw_config["GOOGLE_CLOUD_PROJECT_ID"] = ask(
                "Enter Google Cloud Project ID:"
            )
        if "GOOGLE_CLOUD_LOCATION" not in raw_config:
            raw_config["GOOGLE_CLOUD_LOCATION"] = ask(
                "Enter Google Cloud Location (e.g. us-central1):"
            )

    if "MODEL" not in raw_config:
        raw_config["MODEL"] = prompt_model_name()

    if extras:
        if isinstance(extras, list):
            extras = {
                i: (
                    str(i)
                    .replace('_', ' ')
                    .capitalize()
                    .replace('Llm', 'LLM')
                    .replace('Api', 'API')
                )
                for i in extras
            }
        for field, title in extras.items():
            if field not in raw_config:
                raw_config[field] = ask_non_empty(f"{title}: ").strip()
    if not (config := test_llm_connection(raw_config)):
        if ask_yn("Restart configuring?"):
            return interactive_setup(file_path, defaults, extras)
        return None

    config_body = ''.join(f"{k}={v}\n" for k, v in raw_config.items())
    print(f"Configuration:\n{yellow(config_body)}")
    if ask_yn(f"Save configuration to file {file_link(file_path)}?"):
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(config_body)
        print(f"Saved to {file_link(file_path)}")
    return config


def test_llm_connection(config_dict: dict) -> Config | None:
    """
    Test LLM connection with given configuration dictionary.
    Args:
        config_dict (dict): Configuration dictionary for LLM setup.
    Returns:
        Config | None: Configuration object if the LLM responds correctly, None otherwise.
    """
    try:
        final_config_dict = dict(
            USE_DOT_ENV=False,
            EMBEDDING_DB_TYPE=EmbeddingDbType.NONE,
            USE_LOGGING=True,
        )
        final_config_dict.update(config_dict)
        config = configure(final_config_dict)
        print("Testing LLM...")
        q = "What is capital of France?\n(!) IMPORTANT: Answer only with one word"
        return config if "pari" in llm(q).lower() else None
    except Exception as e:  # pylint: disable=W0718
        error(f"Error testing LLM API: {e}")
        return None
