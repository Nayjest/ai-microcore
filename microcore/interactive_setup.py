from .configuration import EmbeddingDbType, ApiType
from .ui import ask_choose, ask_non_empty, ask_yn, error, yellow
from ._env import configure
from ._llm_functions import llm
from .utils import file_link


def interactive_setup(file_path: str):
    raw_config = dict()
    raw_config["LLM_API_TYPE"] = ask_choose(
        "Choose LLM API Type:",
        list(i.value for i in ApiType if not ApiType.is_local(i)),
    )
    raw_config["LLM_API_KEY"] = ask_non_empty("API Key: ")
    raw_config["MODEL"] = ask_non_empty("Model Name: ")
    raw_config["LLM_API_BASE"] = input("API Base URL (may be empty for some API types): ")
    try:
        configure(
            USE_DOT_ENV=False,
            EMBEDDING_DB_TYPE=EmbeddingDbType.NONE,
            USE_LOGGING=True,
            **raw_config
        )
        print("Testing LLM...")
        q = "What is capital of France?\n(!) IMPORTANT: Answer only with one word"
        assert "pari" in llm(q).lower()
    except Exception as e:  # pylint: disable=W0718
        error(f"Error testing LLM API: {e}")
        if ask_yn("Restart configuring?"):
            interactive_setup(file_path)
        return

    config_body = ''.join(f"{k}={v}\n" for k, v in raw_config.items())
    print(f"Configuration:\n{yellow(config_body)}")
    if ask_yn("Save configuration to file?"):
        print(f"Saved to {file_link(file_path)}")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(config_body)
