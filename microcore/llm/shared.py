import re

from ..configuration import Config


def make_remove_hidden_output(config: Config) -> callable:
    pattern = re.compile(
        f"{config.HIDDEN_OUTPUT_BEGIN}.*?{config.HIDDEN_OUTPUT_END}", flags=re.DOTALL
    )

    def remove_hidden_output(text: str) -> str:
        return pattern.sub("", text)

    return remove_hidden_output
