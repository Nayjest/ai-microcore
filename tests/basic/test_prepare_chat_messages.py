import json
from microcore._prepare_llm_args import prepare_chat_messages
from microcore.message_types import UserMsg, SysMsg, AssistantMsg


def test_prepare_chat_messages(monkeypatch):
    sample_src = [
        {"role": "system", "content": "from sys"},
        {"role": "user", "content": "from user"},
    ]
    sample = json.dumps(sample_src)
    assert json.dumps((prepare_chat_messages(
        [
            SysMsg("from sys"),
            UserMsg("from user"),
        ])
    )) == sample
    assert json.dumps(prepare_chat_messages(sample_src)) == sample
    assert prepare_chat_messages("test") == [{"role": "user", "content": "test"}]
    assert prepare_chat_messages(["1", "2"]) == [
        {"role": "user", "content": "1"},
        {"role": "user", "content": "2"}
    ]
    assert prepare_chat_messages(["1", AssistantMsg()]) == [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": ""}
    ]
    assert prepare_chat_messages(AssistantMsg()) == [{"role": "assistant", "content": ""}]
