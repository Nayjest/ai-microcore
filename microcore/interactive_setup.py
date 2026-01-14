import os

from .configuration import EmbeddingDbType, ApiType, Config
from .ui import ask_choose, ask_non_empty, ask_yn, error, yellow
from ._env import configure
from ._llm_functions import llm
from .utils import file_link


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
        raw_config["LLM_API_TYPE"] = ask_choose(
            "Choose LLM API Type:",
            list(i.value for i in ApiType if not ApiType.is_local(i)),
        )
    if "LLM_API_KEY" not in raw_config:
        raw_config["LLM_API_KEY"] = ask_non_empty("API Key: ").strip()
    if "MODEL" not in raw_config:
        raw_config["MODEL"] = ask_non_empty("Model Name: ").strip()
    if "LLM_API_BASE" not in raw_config:
        raw_config["LLM_API_BASE"] = input(
            "API Base URL (may be empty for some API types): "
        ).strip()
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
