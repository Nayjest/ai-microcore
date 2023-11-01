from types import SimpleNamespace

import pytest
import microcore as mc


@pytest.fixture()
def setup(request, mocker):
    mock_openai_chat(mocker)
    mc.configure(PROMPT_TEMPLATES_PATH='tests/basic/tpl')
    yield


def mock_openai_chat(mocker):
    mocker.patch('openai.Completion.create', side_effect=_side_effect_completion_parrot)
    mocker.patch('openai.ChatCompletion.create', side_effect=_side_effect_chat_parrot)
    mocker.patch('openai.Completion.acreate', side_effect=_side_effect_completion_parrot)
    mocker.patch('openai.ChatCompletion.acreate', side_effect=_side_effect_chat_parrot)


class MockResponse(dict):
    def __init__(self, **entries):
        super().__init__(**entries)
        self.__dict__ = self


def _side_effect_completion_parrot(prompt, **kwargs):
    return MockResponse(choices=[MockResponse(text=f"completion:{prompt}")])


def _side_effect_chat_parrot(messages, **kwargs):
    return MockResponse(choices=[MockResponse(message=MockResponse(content=messages[0]['content']))])