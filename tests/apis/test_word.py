import logging
from colorama import Fore as C
from microcore import config, llm
from .setup_env import setup_env  # noqa


def test_return_word(setup_env):  # noqa
    word = "UNICORN"
    prompt = (
        "You are LLM subsystem of AI software.\n"
        "Respond to following request in straightforward manner."
        'Avoid any additional words like "sure", "here is my answer", etc (!).\n'
        f"\nRequest: REPLAY WITH FOLLOWING WORD: {word}\nResponse:"
    )

    out = llm(prompt)

    logging.info(
        f"\n{C.BLUE}Model: {C.LIGHTGREEN_EX}{config().MODEL}{C.RESET}"
        f"\n{C.BLUE}Prompt: {C.LIGHTGREEN_EX}{prompt}{C.RESET}"
        f"\n{C.BLUE}Response: {C.LIGHTGREEN_EX}{out}{C.RESET}"
    )
    for i in [".", '"', "'", "\n"]:
        out = out.replace(i, "")
    assert word.upper() == out.upper().strip()
