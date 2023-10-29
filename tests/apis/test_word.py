from microcore import llm, env
from .setup_env import setup_env # noqa
import logging
from colorama import Fore as c


def test_return_word(setup_env): # noqa
    word = 'UNICORN'
    prompt = (
        'You are LLM subsystem of AI software.\n'
        'Respond to following request in straightforward manner.'
        'Avoid any additional words like "sure", "here is my answer", etc (!).\n'
        f'\nRequest: REPLAY WITH FOLLOWING WORD: {word}\nResponse:'
    )

    out = llm(prompt)

    logging.info(
        f"\n{c.BLUE}Model: {c.LIGHTGREEN_EX}{env().config.MODEL}{c.RESET}"
        f"\n{c.BLUE}Prompt: {c.LIGHTGREEN_EX}{prompt}{c.RESET}"
        f"\n{c.BLUE}Response: {c.LIGHTGREEN_EX}{out}{c.RESET}"
    )
    for i in ['.', '"', '\'', '\n']:
        out = out.replace(i, '')
    assert word.upper() == out.upper().strip()
