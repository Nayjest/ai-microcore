import json
from microcore._prepare_llm_args import prompt_to_message_dicts
from microcore.message_types import UserMsg, SysMsg, AssistantMsg


def test_prepare_chat_messages(monkeypatch):
    sample_src = [
        {"role": "system", "content": "from sys"},
        {"role": "user", "content": "from user"},
    ]
    sample = json.dumps(sample_src)
    assert json.dumps((prompt_to_message_dicts(
        [
            SysMsg("from sys"),
            UserMsg("from user"),
        ])
    )) == sample
    assert json.dumps(prompt_to_message_dicts(sample_src)) == sample
    assert prompt_to_message_dicts("test") == [{"role": "user", "content": "test"}]
    assert prompt_to_message_dicts(["1", "2"]) == [
        {"role": "user", "content": "1"},
        {"role": "user", "content": "2"}
    ]
    assert prompt_to_message_dicts(["1", AssistantMsg()]) == [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": ""}
    ]
    assert prompt_to_message_dicts(AssistantMsg()) == [{"role": "assistant", "content": ""}]
