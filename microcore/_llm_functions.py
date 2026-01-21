import re
import logging
from datetime import datetime
from typing import Any

from .utils import run_parallel, RETURN_EXCEPTION
from .wrappers.llm_response_wrapper import (
    LLMResponse,
    DictFromLLMResponse,
    ImageGenerationResponse,
)
from .types import (
    TPrompt,
    LLMContextLengthExceededError,
    LLMQuotaExceededError,
    LLMAuthError,
)
from .file_cache import (
    cache_hit,
    load_cache,
    save_cache,
    build_cache_name,
    delete_cache,
)
from ._env import env


# pylint: disable=too-many-return-statements,too-many-branches
def convert_exception(e: Exception, model: str = None) -> Exception | None:
    """
    Convert LLM exceptions microcore-specific exceptions if possible.
    Args:
        e (Exception): Original exception
        model (str): LLM model name, used for better error messages
    Returns:
        Converted exception or None if no conversion is possible
    """

    def with_cause(new_exception: Exception) -> Exception:
        """
        Attach a cause to an exception without raising it.

        Equivalent to `raise new_exc from cause` but returns the exception
        instead of raising, preserving the exception chain for later use.
        """
        new_exception.__cause__ = e
        return new_exception

    if not isinstance(e, Exception):
        return None
    t, msg = f"{type(e).__module__}.{type(e).__name__}", str(e)
    max_tokens, actual_tokens = None, None
    if t == "openai.BadRequestError":
        if "context_length_exceeded" in msg:
            match = re.search(
                r"maximum context length is (\d+) tokens.*?resulted in (\d+) tokens",
                msg,
            )
            if match:
                max_tokens = int(match.group(1))
                actual_tokens = int(match.group(2))
            return with_cause(
                LLMContextLengthExceededError(
                    actual_tokens=actual_tokens, max_tokens=max_tokens, model=model
                )
            )
        if (
            "Please reduce the length of the messages or completion." in msg
        ):  # Groq, no details
            return with_cause(LLMContextLengthExceededError(model=model))

        # x.ai grok-fast
        if (
            "This model's maximum prompt length is" in msg
            and "but the request contains" in msg
            and "tokens" in msg
        ):
            match = re.search(
                r"maximum prompt length is (\d+) but the request contains (\d+) tokens",
                msg,
            )
            if match:
                max_tokens = int(match.group(1))
                actual_tokens = int(match.group(2))
            return with_cause(
                LLMContextLengthExceededError(
                    actual_tokens=actual_tokens, max_tokens=max_tokens, model=model
                )
            )

        if "maximum context length" in msg:  # Mistral, # DeepSeek
            if match := re.search(
                r"Prompt contains (\d+) tokens.*?model with (\d+) maximum context length",
                msg,
            ):  # Mistral
                max_tokens = int(match.group(2))
                actual_tokens = int(match.group(1))
            elif match := re.search(
                r"maximum context length is (\d+) tokens.*? you requested (\d+) tokens",
                msg,
            ):  # DeepSeek
                max_tokens = int(match.group(1))
                actual_tokens = int(match.group(2))
            return with_cause(
                LLMContextLengthExceededError(
                    actual_tokens=actual_tokens, max_tokens=max_tokens, model=model
                )
            )
        if "too_many_prompt_tokens" in msg:  # Perplexity
            if match := re.search(r"User input tokens exceeds (\d+) tokens", msg):
                max_tokens = int(match.group(1))
            return with_cause(
                LLMContextLengthExceededError(
                    actual_tokens=actual_tokens, max_tokens=max_tokens, model=model
                )
            )

    if (
        t == "openai.APIStatusError" and "413 Request Entity Too Large" in msg
    ):  # Cerebras
        return with_cause(LLMContextLengthExceededError(model=model))

    if t == "openai.APIStatusError" and "Payload Too Large" in msg:  # Fireworks
        return with_cause(LLMContextLengthExceededError(model=model))

    if t == "anthropic.BadRequestError" and "prompt is too long:" in msg:
        if match := re.search(r"(\d+)\s+tokens\s+>\s+(\d+)\s+maximum", msg):
            max_tokens = int(match.group(2))
            actual_tokens = int(match.group(1))
        return with_cause(
            LLMContextLengthExceededError(
                actual_tokens=actual_tokens, max_tokens=max_tokens, model=model
            )
        )
    if t == "google.genai.errors.ClientError":

        if "429" in msg and "RESOURCE_EXHAUSTED" in msg:
            return with_cause(LLMQuotaExceededError(details=msg))

        if (
            "input token count" in msg
            and "exceeds the maximum number of tokens allowed" in msg
        ):
            # ai studio
            if match := re.search(
                r"input token count exceeds the maximum number of tokens allowed (\d+)",
                msg,
            ):
                max_tokens = int(match.group(1))
            # vertex
            elif match := re.search(
                r"input token count \((\d+)\) "
                r"exceeds the maximum number of tokens allowed \((\d+)\)",
                msg,
            ):
                actual_tokens = int(match.group(1))
                max_tokens = int(match.group(2))
            return with_cause(
                LLMContextLengthExceededError(
                    actual_tokens=actual_tokens, max_tokens=max_tokens, model=model
                )
            )
    if t in (
        "openai.AuthenticationError",
        "anthropic.AuthenticationError",
        "google.auth.exceptions.MalformedError",  # Vertex AI, wrong service acc. json
    ):
        return with_cause(LLMAuthError(msg))
    if t == "google.genai.errors.ClientError":
        if "API_KEY_INVALID" in msg:
            return with_cause(LLMAuthError(msg))
        if "PERMISSION_DENIED" in msg:  # invalid project in service account json
            return with_cause(LLMAuthError(msg))
    return None


def llm(
    prompt: TPrompt,
    retries: int = 0,
    parse_json: bool | dict = False,
    file_cache: bool | str = False,
    **kwargs,
) -> str | LLMResponse | ImageGenerationResponse:
    """
    Request Large Language Model synchronously

    Args:
        prompt (str | Msg | dict | list[str | Msg | dict]): Text to send to LLM.
        retries (int):
            Number of retries in case of error.
            Default is 0 (no retries).
        parse_json (bool|dict):
            If True, parses response as JSON,
            alternatively non-empty dict can be used as parse_json arguments
            Default is False (no parsing).
        file_cache (bool | str):
            If True or non-empty string, enables file caching of LLM responses.
            If string, it will be used as cache prefix.
            When enabled, identical requests with identical parameters
            will return cached responses instead of making new API calls.
            Default is False (no caching).
        **kwargs: Parameters supported by the LLM API.

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

    if file_cache and cache_hit(
        cache_name := build_cache_name(
            prompt,
            kwargs,
            prefix=file_cache if isinstance(file_cache, str) else "llm_requests",
        )
    ):
        response: LLMResponse = load_cache(cache_name)
        response.from_file_cache = True
        tries = 0
    else:
        tries = retries + 1
        while tries > 0:
            try:
                tries -= 1
                response = env().llm_function(prompt, **kwargs)
                break
            except Exception as e:  # pylint: disable=W0718
                converted_exception = convert_exception(e)
                # If context length exceeded, or no tries left --> do not retry
                if tries == 0 or isinstance(
                    converted_exception, (LLMContextLengthExceededError, LLMAuthError)
                ):
                    if converted_exception:
                        raise converted_exception from e
                    raise e
                logging.error(f"LLM error: {e}")
                logging.info(f"Retrying... {tries} retries left")
                continue
        try:
            response.gen_duration = (datetime.now() - start).total_seconds()
            if not env().config.SAVE_MEMORY:
                response.prompt = prompt
        except AttributeError:
            ...
        if file_cache:
            save_cache(cache_name, response)
    [h(response) for h in env().llm_after_handlers]
    if tries > 0:
        setattr(response, "_retry_callback", lambda: llm(prompt, retries=tries - 1, **kwargs))
    if parse_json:
        parsing_params = parse_json if isinstance(parse_json, dict) else {}
        return response.parse_json(**parsing_params)
    return response


async def allm(
    prompt: TPrompt,
    retries: int = 0,
    parse_json: bool | dict = False,
    file_cache: bool | str = False,
    **kwargs,
) -> str | LLMResponse | DictFromLLMResponse | ImageGenerationResponse:
    """
    Request Large Language Model asynchronously

    Args:
        prompt (str | Msg | dict | list[str | Msg | dict]): Text to send to LLM.
        retries (int):
            Number of retries in case of error.
            Default is 0 (no retries).
        parse_json (bool|dict):
            If True, parses response as JSON,
            alternatively non-empty dict can be used as parse_json arguments.
            Default is False (no parsing).
        file_cache (bool | str):
            If True or non-empty string, enables file caching of LLM responses.
            If string, it will be used as cache prefix.
            When enabled, identical requests with identical parameters
            will return cached responses instead of making new API calls.
            Default is False (no caching).
        **kwargs: Parameters supported by the LLM API.

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

    if file_cache and cache_hit(
        cache_name := build_cache_name(
            prompt,
            kwargs,
            prefix=file_cache if isinstance(file_cache, str) else "llm_requests",
        )
    ):
        response: LLMResponse = load_cache(cache_name)
        response.from_file_cache = True
        tries = 0
    else:
        tries = retries + 1
        while tries > 0:
            try:
                tries -= 1
                response = await env().llm_async_function(prompt, **kwargs)
                break
            except Exception as e:  # pylint: disable=W0718
                converted_exception = convert_exception(e)
                # If context length exceeded, or no tries left --> do not retry
                if tries == 0 or isinstance(
                    converted_exception, (LLMContextLengthExceededError, LLMAuthError)
                ):
                    if converted_exception:
                        raise converted_exception from e
                    raise e
                logging.error(f"LLM error: {e}")
                logging.info(f"Retrying... {tries} retries left")
                continue
        try:
            response.gen_duration = (datetime.now() - start).total_seconds()
            if not env().config.SAVE_MEMORY:
                response.prompt = prompt
        except AttributeError:
            ...
        if file_cache:
            save_cache(cache_name, response)
    [h(response) for h in env().llm_after_handlers]
    if parse_json:
        try:
            parsing_params = parse_json if isinstance(parse_json, dict) else {}
            return response.parse_json(**parsing_params)
        except Exception as e:  # pylint: disable=W0718
            if tries > 0:
                logging.error(f"LLM error: {e}")
                logging.info(f"Retrying... {tries} retries left")
                if file_cache:
                    delete_cache(cache_name)
                return await allm(
                    prompt,
                    retries=tries - 1,
                    parse_json=parse_json,
                    file_cache=file_cache,
                    **kwargs
                )
            raise e
    return response


async def llm_parallel(
    prompts: list[TPrompt],
    max_concurrent_tasks: int = None,
    allow_failures: bool = False,
    return_on_failure: Any = RETURN_EXCEPTION,
    log_errors: bool = True,
    **kwargs,
) -> list[str | LLMResponse]:
    """
    Execute multiple LLM requests in parallel

    Returns (list[LLMResponse | str]): a list of responses in the same order as the prompts
    """
    tasks = [allm(prompt, **kwargs) for prompt in prompts]

    if max_concurrent_tasks is None:
        max_concurrent_tasks = int(env().config.MAX_CONCURRENT_TASKS or 0)
    if not max_concurrent_tasks:
        max_concurrent_tasks = len(tasks)

    return await run_parallel(
        tasks,
        max_concurrent_tasks=max_concurrent_tasks,
        allow_failures=allow_failures,
        return_on_failure=return_on_failure,
        log_errors=log_errors,
    )
