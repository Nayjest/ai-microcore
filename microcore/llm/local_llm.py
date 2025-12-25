import asyncio
import inspect
import threading
from typing import Awaitable, Optional, Any, TypeVar

from ..configuration import Config, LLMConfigError
from .._prepare_llm_args import prompt_to_message_dicts, prepare_prompt
from ..types import LLMAsyncFunctionType, LLMFunctionType
from ..utils import resolve_callable
from ..wrappers.llm_response_wrapper import LLMResponse
from .shared import prepare_callbacks

T = TypeVar("T")


class _sync_await:
    def __init__(self):
        self._loop = None
        self._looper = None

    def __enter__(self) -> "_sync_await":
        self._loop = asyncio.new_event_loop()
        self._looper = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._looper.start()
        return self

    def __call__(self, coro: Awaitable[T], timeout: Optional[float] = None) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def __exit__(self, *exc_info: Any) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._looper.join()
        self._loop.close()


def _prepare_llm_arguments(config: Config, kwargs: dict):
    args = {**config.LLM_DEFAULT_ARGS, **kwargs}
    args.pop("model", None)
    callbacks = prepare_callbacks(config, args, set_stream=False)
    return args, {"callbacks": callbacks}


def make_llm_functions(
    config: Config, overriden_inference_func: callable = None
) -> tuple[LLMFunctionType, LLMAsyncFunctionType]:
    try:
        inference_fn = resolve_callable(
            overriden_inference_func or config.INFERENCE_FUNC
        )
    except ValueError as e:
        raise LLMConfigError(f"Invalid inference function, {e}") from e

    if inspect.iscoroutinefunction(inference_fn):

        async def allm(prompt, **kwargs):
            args, options = _prepare_llm_arguments(config, kwargs)
            prompt = (
                prompt_to_message_dicts(prompt)
                if config.CHAT_MODE
                else prepare_prompt(prompt)
            )
            response = await inference_fn(prompt, **args)
            for cb in options["callbacks"]:
                cb(response)
            if not isinstance(response, LLMResponse):
                response = LLMResponse(response)
            return response

        def llm(prompt, **kwargs):
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    with _sync_await() as sa:
                        return sa(allm(prompt, **kwargs))
                else:
                    # not tested
                    return loop.run_until_complete(allm(prompt, **kwargs))
            except RuntimeError:
                return asyncio.run(allm(prompt, **kwargs))

    else:

        def llm(prompt, **kwargs):
            args, options = _prepare_llm_arguments(config, kwargs)
            prompt = (
                prompt_to_message_dicts(prompt)
                if config.CHAT_MODE
                else prepare_prompt(prompt)
            )
            response = inference_fn(prompt, **args)
            for cb in options["callbacks"]:
                cb(response)
            if not isinstance(response, LLMResponse):
                response = LLMResponse(response)
            return response

        async def allm(prompt, **kwargs):
            return llm(prompt, **kwargs)

    return llm, allm
