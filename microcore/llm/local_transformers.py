import logging
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from .local_llm import make_llm_functions as make_local_llm_functions
from ..configuration import Config
from ..types import LLMFunctionType, LLMAsyncFunctionType


def make_llm_functions(config: Config, env=None) -> tuple[LLMFunctionType, LLMAsyncFunctionType]:
    logging.info(f"Loading local Transformers model {config.MODEL}...")
    params = config.INIT_PARAMS
    mc_param_names = ['model', 'tokenizer', 'device']
    model_init_params = {k: v for k, v in params.items() if k not in mc_param_names}
    mc_params = {k: v for k, v in params.items() if k in mc_param_names}
    if 'model' not in mc_params or 'tokenizer' not in mc_params:
        gc.collect()
        torch.cuda.empty_cache()
    if 'tokenizer' in mc_params:
        tokenizer = mc_params['tokenizer']
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL, trust_remote_code=True)
    if 'model' in mc_params:
        model = mc_params['model']
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL,
            **dict(
                **dict(
                    trust_remote_code=True,
                    torch_dtype="auto",
                    device_map="auto",
                    offload_folder=config.STORAGE_PATH,
                ),
                **model_init_params
            )
        )
        if 'device' in mc_params:
            model.to(mc_params['device'])
    transformers_model_args = {
        "max_new_tokens": 2048,
    }
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id:
        transformers_model_args['eos_token_id'] = tokenizer.eos_token_id

    def inference(prompt: dict | str, **kwargs):
        if config.CHAT_MODE:
            prompt = tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=False
            )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, **{**transformers_model_args, **kwargs})
        out = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return out

    if env:
        setattr(env, 'tokenizer', tokenizer)
        setattr(env, 'model', model)

    logging.debug(f"Local Transformers model loaded: {config.MODEL}")

    return make_local_llm_functions(config, inference)
