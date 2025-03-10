from datetime import datetime

from .utils import run_parallel
from .wrappers.llm_response_wrapper import LLMResponse
from .types import TPrompt
from ._env import env


def llm(prompt: TPrompt, **kwargs) -> str | LLMResponse:
    """
    Request Large Language Model synchronously

    Args:
        prompt (str | Msg | dict | list[str | Msg | dict]): Text to send to LLM
        **kwargs (dict): Parameters supported by the LLM API

            See parameters supported by the OpenAI:

            - https://platform.openai.com/docs/api-reference/completions/create
            - https://platform.openai.com/docs/api-reference/chat/create

            **Additional parameters:**

                - callback: callable - callback function
                to be called on each chunk of text,
                enables response streaming if supported by the LLM API
                - callbacks: list[callable] - collection of callbacks
                to be called on each chunk of text,
                enables response streaming if supported by the LLM API

    Returns:

        Text generated by the LLM as string
        with all fields returned by API accessible as an attributes.

        See fields returned by the OpenAI:

        - https://platform.openai.com/docs/api-reference/completions/object
        - https://platform.openai.com/docs/api-reference/chat/object
    """
    [h(prompt, **kwargs) for h in env().llm_before_handlers]
    start = datetime.now()
    response = env().llm_function(prompt, **kwargs)
    try:
        response.gen_duration = (datetime.now() - start).total_seconds()
        if not env().config.SAVE_MEMORY:
            response.prompt = prompt
    except AttributeError:
        ...
    [h(response) for h in env().llm_after_handlers]
    return response


async def allm(prompt: TPrompt, **kwargs) -> str | LLMResponse:
    """
    Request Large Language Model asynchronously

    Args:
        prompt (str | Msg | dict | list[str | Msg | dict]): Text to send to LLM
        **kwargs (dict): Parameters supported by the LLM API

            See parameters supported by the OpenAI:

            - https://platform.openai.com/docs/api-reference/completions/create
            - https://platform.openai.com/docs/api-reference/chat/create

            **Additional parameters:**

            - callback: callable - callback function
            to be called on each chunk of text,
            enables response streaming if supported by the LLM API
            - callbacks: list[callable] - collection of callbacks
            to be called on each chunk of text,
            enables response streaming if supported by the LLM API

            Note: async callbacks are supported only for async LLM API calls

    Returns:

        Text generated by the LLM as string
        with all fields returned by API accessible as an attributes.

        See fields returned by the OpenAI:

        - https://platform.openai.com/docs/api-reference/completions/object
        - https://platform.openai.com/docs/api-reference/chat/object
    """
    [h(prompt, **kwargs) for h in env().llm_before_handlers]
    start = datetime.now()
    response = await env().llm_async_function(prompt, **kwargs)
    try:
        response.gen_duration = (datetime.now() - start).total_seconds()
        if not env().config.SAVE_MEMORY:
            response.prompt = prompt
    except AttributeError:
        ...
    [h(response) for h in env().llm_after_handlers]
    return response


async def llm_parallel(
    prompts: list[TPrompt], max_concurrent_tasks: int = None, **kwargs
) -> list[str | LLMResponse]:
    """
    Execute multiple LLM requests in parallel

    Returns (list[LLMResponse | str]): a list of responses in the same order as the prompts
    """
    tasks = [allm(prompt, **kwargs) for prompt in prompts]

    if max_concurrent_tasks is None:
        max_concurrent_tasks = int(env().config.MAX_CONCURRENT_TASKS)
    if not max_concurrent_tasks:
        max_concurrent_tasks = len(tasks)

    return await run_parallel(tasks, max_concurrent_tasks=max_concurrent_tasks)
