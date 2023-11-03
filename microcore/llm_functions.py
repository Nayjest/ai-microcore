from .wrappers.llm_response_wrapper import LLMResponse
from .internal_env import env


async def allm(prompt, **kwargs) -> str | LLMResponse:
    [h(prompt, **kwargs) for h in env().llm_before_handlers]
    response = await env().llm_async_function(prompt, **kwargs)
    [h(response) for h in env().llm_after_handlers]
    return response


def llm(prompt, **kwargs) -> str | LLMResponse:
    [h(prompt, **kwargs) for h in env().llm_before_handlers]
    response = env().llm_function(prompt, **kwargs)
    [h(response) for h in env().llm_after_handlers]
    return response
