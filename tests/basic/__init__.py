import importlib.metadata
import os

import pytest
import microcore as mc


@pytest.fixture()
def setup(request, mocker):
    os.environ.clear()
    mock_openai_chat(mocker)
    mc.configure(
        USE_DOT_ENV=False,
        PROMPT_TEMPLATES_PATH="tests/basic/tpl",
        LLM_API_TYPE=mc.ApiType.OPEN_AI,
        LLM_API_KEY="123",
    )
    yield


def mock_openai_chat(mocker):
    _openai_version = importlib.metadata.version("openai")
    if _openai_version.startswith("1."):
        mocker.patch(
            "openai.resources.AsyncCompletions.create",
            side_effect=_aside_effect_compl_parrot,
        )
        mocker.patch(
            "openai.resources.chat.AsyncCompletions.create",
            side_effect=_aside_effect_chat_parrot,
        )
        mocker.patch(
            "openai.resources.Completions.create", side_effect=_side_effect_compl_parrot
        )
        mocker.patch(
            "openai.resources.chat.Completions.create",
            side_effect=_side_effect_chat_parrot,
        )
    else:
        mocker.patch("openai.Completion.create", side_effect=_side_effect_compl_parrot)
        mocker.patch(
            "openai.ChatCompletion.create", side_effect=_side_effect_chat_parrot
        )
        mocker.patch("openai.Completion.acreate", side_effect=_side_effect_compl_parrot)
        mocker.patch(
            "openai.ChatCompletion.acreate", side_effect=_side_effect_chat_parrot
        )


class MockResponse(dict):
    def __init__(self, **entries):
        super().__init__(**entries)
        self.__dict__ = self


def _side_effect_compl_parrot(prompt, **kwargs):
    return MockResponse(choices=[MockResponse(text=f"completion:{prompt}")])


def _side_effect_chat_parrot(messages, **kwargs):
    return MockResponse(
        choices=[MockResponse(message=MockResponse(content=messages[0]["content"]))]
    )


async def _aside_effect_compl_parrot(prompt, **kwargs):
    return MockResponse(choices=[MockResponse(text=f"completion:{prompt}")])


async def _aside_effect_chat_parrot(messages, **kwargs):
    return MockResponse(
        choices=[MockResponse(message=MockResponse(content=messages[0]["content"]))]
    )
