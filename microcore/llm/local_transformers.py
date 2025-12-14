import logging
import gc

import transformers
import torch

from .local_llm import make_llm_functions as make_local_llm_functions
from ..message_types import PartialMsg
from ..wrappers.llm_response_wrapper import LLMResponse
from ..configuration import Config, LLMConfigError
from ..types import LLMFunctionType, LLMAsyncFunctionType


def inference(prompt: str, model, tokenizer, **kwargs):
    skip_special_tokens = kwargs.pop("skip_special_tokens", True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **kwargs)
    outputs = [
        tokenizer.decode(i[len(inputs[0]):], skip_special_tokens=skip_special_tokens)
        for i in outputs
    ]
    return LLMResponse(outputs[0], dict(all=outputs))


def pipeline_inference(prompt: str, pipeline, **kwargs):
    raw_output = pipeline(prompt, **{"return_full_text": False, **kwargs})
    return LLMResponse(
        raw_output[0]["generated_text"],
        dict(all=list(map(lambda x: x["generated_text"], raw_output))),
    )


def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()


def try_make_model_config(config: Config):
    try:
        model_config = transformers.AutoConfig.from_pretrained(
            config.MODEL,
            trust_remote_code=True,
        )
    except OSError:
        model_config = None
    if config.INIT_PARAMS.get("gradient_checkpointing"):
        if model_config:
            model_config.gradient_checkpointing = True
        else:
            raise LLMConfigError(
                "Can't apply gradient checkpointing without model config."
            )
    return model_config


def get_last_message_ending(tokenizer) -> str:
    try:
        sample_content = "sample"
        sample_prompt = tokenizer.apply_chat_template(
            [
                dict(role="user", content="hi"),
                dict(role="assistant", content=sample_content),
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        pos = sample_prompt.rfind(sample_content) + len(sample_content)
        return sample_prompt[pos:]
    except:  # pylint: disable=bare-except # noqa
        logging.warning(
            "Can't determine the ending of the last message in the tokenizer's chat template."
        )
        return ""


def resolve_tokenizer(config: Config):
    return config.INIT_PARAMS.get(
        "tokenizer"
    ) or transformers.AutoTokenizer.from_pretrained(
        config.MODEL, trust_remote_code=True
    )


def resolve_model(config: Config):
    params = config.INIT_PARAMS
    if not (model := params.get("model")):
        clear_mem()
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
            "stops_from_chat_template",  # bool, use chat template to determine stopping criteria
        ]

        model_init_params = {
            **dict(
                trust_remote_code=True,
                dtype="auto",
                device_map="auto",
                offload_folder=config.STORAGE_PATH,
            ),
            **{k: v for k, v in params.items() if k not in mc_param_names},
        }
        if "config" not in model_init_params:
            model_init_params["config"] = try_make_model_config(config)

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
        try:
            model.generation_config = transformers.GenerationConfig.from_pretrained(
                config.MODEL
            )
        except:  # pylint: disable=bare-except  # noqa
            logging.warning("Can't create generation config")

        if "device" in params:
            model.to(params["device"])

    return model


def make_llm_functions(
    config: Config, env
) -> tuple[LLMFunctionType, LLMAsyncFunctionType]:
    logging.info(f"Loading local Transformers model {config.MODEL}...")
    params = config.INIT_PARAMS

    tokenizer = resolve_tokenizer(config)
    model = resolve_model(config)

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
            dtype=params.get("torch_dtype") or params.get("dtype") or "auto",
            device_map=params.get("device_map", "auto"),
        )
        setattr(env, "pipeline", pipeline)

    def make_stopping_criteria(seq: str | list[str]) -> list[callable]:
        if isinstance(seq, list):
            funcs = []
            for s in seq:
                funcs += make_stopping_criteria(s)
            return funcs

        stop_seq = tokenizer.encode(
            seq, add_special_tokens=False, return_tensors="pt"
        ).to(model.device)[0]
        # Some tokenizers like Phi-3 (SentencePiece tokenizer) add an extra token at the beginning
        # independently of add_special_tokens=False
        if stop_seq[0] == 29871:  # SENTENCEPIECE_UNDERLINE
            stop_seq = stop_seq[1:]
        offset = -len(stop_seq)

        def stopping_criteria(ids, *args):  # pylint: disable=unused-argument
            return torch.all(torch.eq(ids[0][offset:], stop_seq))

        return [stopping_criteria]

    setattr(env, "tokenizer", tokenizer)
    setattr(env, "model", model)
    setattr(env, "inference", inference_fn)
    setattr(env, "make_stopping_criteria", make_stopping_criteria)
    always_clear_mem = params.get("always_clear_mem", False)
    ending = get_last_message_ending(tokenizer) if config.CHAT_MODE else ""

    stops_from_chat_template: list[callable] = []
    if params.get("stops_from_chat_template", True) and ending:
        stops_from_chat_template: list[callable] = make_stopping_criteria(
            ending.strip()
        )

    def wrapped_inference(prompt: list[dict] | str, **kwargs):
        if (n := kwargs.pop("n", None)) is not None:  # open_ai style
            kwargs["num_return_sequences"] = n

        if (seed := kwargs.pop("seed", None)) is not None:  # open_ai style
            transformers.set_seed(seed)

        if (stop := kwargs.pop("stop", None)) is not None:  # open_ai style
            kwargs["stopping_criteria"] = make_stopping_criteria(stop)

        partial_msg_used = False
        if config.CHAT_MODE:
            sc = kwargs.get("stopping_criteria", [])
            if stops_from_chat_template:
                sc += stops_from_chat_template
            if isinstance(prompt, list) and len(prompt) > 0:
                last = prompt[-1]
                if isinstance(last, dict) and hasattr(last, "is_partial"):
                    kwargs["continue_last_message"] = True
                    prefix, suffixes = PartialMsg.split_prefix_and_suffixes(
                        last["content"]
                    )
                    last["content"] = prefix
                    if suffixes:
                        sc += make_stopping_criteria(suffixes)
                    partial_msg_used = True
            if sc:
                kwargs["stopping_criteria"] = sc
            prompt = tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=False
            )
            if kwargs.pop("continue_last_message", False) and ending:
                prompt = prompt[: -len(ending)]
        if seed := kwargs.pop("seed", None):
            torch.manual_seed(seed)
        args = {**transformers_model_args, **kwargs}
        always_clear_mem and clear_mem()
        if use_pipeline:
            out = pipeline_inference(prompt, pipeline, **args)
        else:
            out = env.inference(
                prompt, model=env.model, tokenizer=env.tokenizer, **args
            )
        if not isinstance(out, LLMResponse):
            out = LLMResponse(out)
        if partial_msg_used:
            # original generation doesn't include the prefix, but includes the suffix
            actual_suffix = ""
            for s in suffixes:
                if out.endswith(s):
                    actual_suffix = s
                    break
            setattr(
                out,
                "inner",
                LLMResponse(out[: -len(actual_suffix)]) if actual_suffix else out,
            )
            setattr(out, "outer", LLMResponse(prefix + out))
            setattr(out, "prefixed", LLMResponse(prefix + out.inner))
        return out

    logging.info(f"Local Transformers model loaded: {config.MODEL}")
    return make_local_llm_functions(config, wrapped_inference)
