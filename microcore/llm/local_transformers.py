import logging
import gc

import transformers
import torch

from .local_llm import make_llm_functions as make_local_llm_functions
from ..configuration import Config
from ..types import LLMFunctionType, LLMAsyncFunctionType


def inference(prompt: str, model, tokenizer, **kwargs):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **kwargs)
    out = tokenizer.decode(outputs[0][len(inputs[0]) :], skip_special_tokens=True)
    return out


def pipeline_inference(prompt: str, pipeline, **kwargs):
    raw_output = pipeline(prompt, **{"return_full_text": False, **kwargs})
    return raw_output[0]["generated_text"]


def make_llm_functions(
    config: Config, env
) -> tuple[LLMFunctionType, LLMAsyncFunctionType]:
    logging.info(f"Loading local Transformers model {config.MODEL}...")
    params = config.INIT_PARAMS

    tokenizer = params.get("tokenizer") or transformers.AutoTokenizer.from_pretrained(
        config.MODEL, trust_remote_code=True
    )

    if not (model := params.get("model")):
        gc.collect()
        torch.cuda.empty_cache()
        model_config = transformers.AutoConfig.from_pretrained(
            config.MODEL,
            trust_remote_code=True,
        )
        if params.get("gradient_checkpointing"):
            model_config.gradient_checkpointing = True
        mc_param_names = [
            "model",  # custom transformers model instance
            "tokenizer",  # custom tokenizer instance
            "device",  # device if model.to(device) is needed
            "quantize_4bit",  # bool, default quantization config will be used
            "inference",  # callable, custom inference function.
            # this is different from Config.INFERENCE_FUNC,
            # because accepts model and tokenizer as arguments
            "always_clear_mem",  # gc.collect() & torch.cuda.empty_cache() before inference
            "gradient_checkpointing",  # bool, enable gradient checkpointing
            "use_pipeline",  # bool
            "pipeline_task",  # str, pipeline task name
        ]
        model_init_params = {
            **dict(
                trust_remote_code=True,
                torch_dtype="auto",
                config=model_config,
                device_map="auto",
                offload_folder=config.STORAGE_PATH,
            ),
            **{k: v for k, v in params.items() if k not in mc_param_names},
        }
        if (
            params.get("quantize_4bit")
            and "quantization_config" not in model_init_params
        ):
            model_init_params["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.MODEL, **model_init_params
        )
        if "device" in params:
            model.to(params["device"])

    transformers_model_args = {
        "max_new_tokens": 2048,
    }
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id:
        transformers_model_args["eos_token_id"] = tokenizer.eos_token_id

    inference_fn = params.get("inference") or inference
    if isinstance(inference_fn, str):
        inference_fn = globals()[inference_fn]
    use_pipeline = params.get("use_pipeline", False)
    if use_pipeline:
        pipeline = transformers.pipeline(
            params.get("pipeline_task", "text-generation"),
            model=model,
            tokenizer=tokenizer,
            torch_dtype=params.get("torch_dtype", "auto"),
            device_map=params.get("device_map", "auto"),
        )
        setattr(env, "pipeline", pipeline)

    setattr(env, "tokenizer", tokenizer)
    setattr(env, "model", model)
    setattr(env, "inference", inference_fn)
    always_clear_mem = params.get("always_clear_mem", False)

    def wrapped_inference(prompt: dict | str, **kwargs):
        if config.CHAT_MODE:
            prompt = tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=False
            )
        args = {**transformers_model_args, **kwargs}
        if always_clear_mem:
            gc.collect()
            torch.cuda.empty_cache()
        if use_pipeline:
            return pipeline_inference(prompt, pipeline, **args)

        return env.inference(
            prompt, model=env.model, tokenizer=env.tokenizer, **args
        )

    logging.debug(f"Local Transformers model loaded: {config.MODEL}")
    return make_local_llm_functions(config, wrapped_inference)
