import microcore as mc
import pytest
from microcore.configuration import Config, LLMConfigError


def test_no_model():
    # should raise an error
    with pytest.raises(LLMConfigError):
        # No model name
        mc.configure(LLM_API_TYPE=mc.ApiType.TRANSFORMERS, CHAT_MODE=True)


def test():
    import os
    os.environ['NVIDIA_VISIBLE_DEVICES'] = 'all'
    defaults = dict(
        LLM_API_TYPE=mc.ApiType.TRANSFORMERS,
        LLM_DEFAULT_ARGS={
            "max_new_tokens": 30,
        },
        INIT_PARAMS={
            'quantize_4bit': True,
        }
    )
    configs = [
        Config(
            MODEL='microsoft/phi-1_5',
            CHAT_MODE=False,
            **defaults
        ),
        Config(
            MODEL='deepseek-ai/deepseek-coder-1.3b-instruct',
            CHAT_MODE=False,
            **defaults
        ),
        Config(
            MODEL='google/gemma-2b-it',
            CHAT_MODE=True,
            **defaults
        ),
    ]
    for config in configs:
        mc.configure(**dict(config))
        mc.use_logging()
        assert '3' in mc.llm('Count from 1 to 3: 1..., 2...')