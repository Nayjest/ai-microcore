"""
Command-line interface LLM client implementation.
"""
import asyncio
import shlex
import shutil
import threading

from ..lm_client import BaseAIChatClient, BaseAsyncAIClient
from ..configuration import Config
from ..llm_backends import ApiType
from ..types import BadAIAnswer, TPrompt
from ..wrappers.llm_response_wrapper import (
    LLMResponse,
    ImageGenerationResponse,
    StoredImageGenerationResponse
)
from .shared import prepare_callbacks


class CommandLineLLMError(BadAIAnswer):
    def __init__(
        self, message: str = "Error executing LLM CLI command", details=None
    ):
        super().__init__(message, details)


def _run_sync(coro):
    """Run a coroutine to completion from sync code, whether or not a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # no loop: simplest path

    # loop already running in this thread: run on a fresh loop in another thread
    result = {}

    def runner():
        result["value"] = asyncio.run(coro)

    t = threading.Thread(target=runner)
    t.start()
    t.join()
    return result["value"]


async def run_streaming(argv, callback) -> str:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    chunks = []
    first = True

    async def pump_stdout():
        nonlocal first
        async for raw in proc.stdout:        # yields complete lines as they arrive
            text = raw.decode()
            if first:
                first = False
                if text.startswith(">"):
                    text = text[1:].lstrip()
            chunks.append(text)
            await callback(text)

    stderr_data = bytearray()

    async def pump_stderr():
        async for raw in proc.stderr:
            stderr_data.extend(raw)

    await asyncio.gather(pump_stdout(), pump_stderr())
    returncode = await proc.wait()
    output = "".join(chunks).strip()
    if returncode != 0:
        raise CommandLineLLMError(stderr_data.decode().strip() or output)
    return output


class CommandLineClient(BaseAIChatClient):
    PLACEHOLDER = "<request>"
    aio: "AsyncCommandLineClient"

    def __init__(self, config: Config):
        super().__init__(config)
        self.aio = AsyncCommandLineClient(self)

    def load_models(self, **_kwargs) -> dict:
        return {
            self.config.MODEL: {
                "id": self.config.MODEL,
                "supports_functions": False,
                "supports_vision": False,
            }
        }

    def prepare_run(self, prompt: TPrompt, kwargs: dict):
        """Build the command line and streaming callback shared by sync/async."""
        args = {**self.config.LLM_DEFAULT_ARGS, **kwargs}
        args["model"] = args.get("model", self.config.MODEL)
        callbacks = prepare_callbacks(self.config, args)
        messages = self.convert_prompt_to_chat_input(prompt)

        if len(messages) > 1:
            prompt_str = ""
            for msg in messages:
                prompt_str += f"\n[{msg['role']}]:{msg['content'][0]}"
        else:
            prompt_str = messages[0]['content'][0]

        async def callback(text):
            for cb in callbacks:
                if asyncio.iscoroutinefunction(cb):
                    await cb(text)
                else:
                    cb(text)

        argv = shlex.split(self.config.LLM_CLI, posix=True)
        argv = [a.replace(self.PLACEHOLDER, prompt_str) for a in argv]
        resolved = shutil.which(argv[0])
        if resolved:
            argv[0] = resolved
        return argv, callback

    def generate(
        self,
        prompt: TPrompt,
        **kwargs
    ) -> LLMResponse | ImageGenerationResponse | StoredImageGenerationResponse:
        argv, callback = self.prepare_run(prompt, kwargs)
        result: str = _run_sync(run_streaming(argv, callback))
        return LLMResponse(
            result,
            api_type=ApiType.CLI,
        )


class AsyncCommandLineClient(BaseAsyncAIClient):
    sync_client: "CommandLineClient"

    def __init__(self, sync_client: "CommandLineClient"):
        self.sync_client = sync_client

    async def load_models(self, **kwargs) -> dict:
        return self.sync_client.load_models(**kwargs)

    async def generate(
        self,
        prompt: TPrompt,
        **kwargs
    ) -> LLMResponse:
        argv, callback = self.sync_client.prepare_run(prompt, kwargs)
        result: str = await run_streaming(argv, callback)
        return LLMResponse(
            result,
            api_type=ApiType.CLI,
        )
