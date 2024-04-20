import gc
import logging

import torch

import microcore as mc
import pytest

from microcore import ui
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
            LLM_API_TYPE=mc.ApiType.TRANSFORMERS,
            MODEL='Qwen/Qwen1.5-1.8B-Chat',
            CHAT_MODE=True,
        ),
        Config(
            MODEL='google/gemma-2b-it',
            CHAT_MODE=True,
            **{**defaults,
               "INIT_PARAMS": {'quantize_4bit': False, 'use_pipeline': True, 'gradient_checkpointing': True}}
        ),
        Config(
            MODEL='microsoft/phi-1_5',
            CHAT_MODE=False,
            **{**defaults, "INIT_PARAMS": {'quantize_4bit': True, 'use_pipeline': True}}
        ),
        Config(
            MODEL='deepseek-ai/deepseek-coder-1.3b-instruct',
            CHAT_MODE=False,
            **defaults
        ),
    ]
    for config in configs:
        logging.info(ui.magenta(f"Testing model {config.MODEL}..."))
        mc.configure(**dict(config))
        mc.use_logging()
        assert '3' in mc.llm('Count from 1 to 3: 1..., 2...')