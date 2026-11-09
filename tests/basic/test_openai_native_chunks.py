import asyncio

from microcore.llm.openai import (
    _a_process_streamed_response,
    _process_streamed_response,
)


class Chunk:
    def __init__(self, content=None, tool_calls=None, finish_reason=None):
        self.choices = [
            type(
                "Choice",
                (),
                {
                    "delta": type("Delta", (), {"content": content})(),
                    "finish_reason": finish_reason,
                },
            )()
        ]
        self.content = content
        self.tool_calls = tool_calls
        self.finish_reason = finish_reason
        self.usage = None

    def model_dump(self, mode, exclude_none):
        delta = {}
        if self.content is not None:
            delta["content"] = self.content
        if self.tool_calls is not None:
            delta["tool_calls"] = self.tool_calls
        return {
            "object": "chat.completion.chunk",
            "choices": [{"delta": delta, "finish_reason": self.finish_reason}],
        }


def test_native_chunks_do_not_change_text_callbacks():
    text = []
    native = []

    def callback(chunk):
        text.append(chunk)

    callback.on_openai_chunk = native.append
    tool_call = [{"index": 0, "function": {"name": "ping", "arguments": "{"}}]
    result = _process_streamed_response(
        [
            Chunk(content="Hello"),
            Chunk(tool_calls=tool_call),
            Chunk(finish_reason="tool_calls"),
        ],
        [callback],
        chat_model_used=True,
        native_chunk_callbacks=[callback.on_openai_chunk],
    )

    assert result == "Hello"
    assert text == ["Hello"]
    assert native[1]["choices"][0]["delta"]["tool_calls"] == tool_call
    assert native[2]["choices"][0]["finish_reason"] == "tool_calls"


def test_async_native_chunks_preserve_argument_fragments():
    native = []

    async def response():
        yield Chunk(tool_calls=[{"index": 0, "function": {"arguments": "{"}}])
        yield Chunk(tool_calls=[{"index": 0, "function": {"arguments": "}"}}])
        yield Chunk(finish_reason="tool_calls")

    result = asyncio.run(
        _a_process_streamed_response(
            response(),
            [],
            chat_model_used=True,
            native_chunk_callbacks=[native.append],
        )
    )

    assert result == ""
    assert (
        native[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
        == "{"
    )
    assert (
        native[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
        == "}"
    )
    assert native[2]["choices"][0]["finish_reason"] == "tool_calls"
