from ..config import Config, ApiType
from ..extended_string import ExtendedString
from ..prepare_llm_args import prepare_chat_messages, prepare_prompt
from ..types import LLMFunctionType
from ..utils import is_chat_model
import openai
import openai.util


def _process_streamed_response(response, callbacks: list[callable], mode_chat_model: bool):
    response_text: str = ''
    for chunk in response:
        # Azure API gives first chunk with empty choices
        choice = chunk.choices[0] if len(chunk.choices) else {}

        if mode_chat_model:
            text_chunk = choice.get('delta', {}).get('content', '')
        else:
            text_chunk = choice.get('text', '')

        # avoiding callbacks if empty
        if text_chunk == '':
            continue
        response_text += text_chunk
        for cb in callbacks:
            cb(text_chunk)
    return ExtendedString(response_text, {})


def make_llm_function(config: Config) -> LLMFunctionType:
    try:
        api_type = config.LLM_API_TYPE
        openai.util.ApiType.from_str(api_type)
    except openai.error.InvalidAPIType:
        api_type = ApiType.OPEN_AI

    openai.api_type = api_type
    openai.api_key = config.LLM_API_KEY
    openai.api_base = config.LLM_API_BASE
    openai.api_version = config.LLM_API_VERSION

    def llm(prompt, **kwargs):

        args = {**config.LLM_DEFAULT_ARGS, **kwargs}

        args['model'] = args.get('model', config.MODEL)

        if config.LLM_API_TYPE == ApiType.AZURE:
            args['deployment_id'] = args.get('deployment_id', config.LLM_DEPLOYMENT_ID)

        callbacks: list[callable] = args.pop('callbacks', [])
        if 'callback' in args:
            callbacks.append(args.pop('callback'))
        args['stream'] = bool(callbacks)

        if is_chat_model(args['model']):
            response = openai.ChatCompletion.create(messages=prepare_chat_messages(prompt), **args)
            if args['stream']:
                return _process_streamed_response(response, callbacks, mode_chat_model=True)
            else:
                for cb in callbacks:
                    cb(response.choices[0].message.content)
                return ExtendedString(response.choices[0].message.content, response)
        else:
            response = openai.Completion.create(prompt=prepare_prompt(prompt), **args)
            if args['stream']:
                return _process_streamed_response(response, callbacks, mode_chat_model=False)
            else:
                return ExtendedString(response.choices[0].text, response)

    return llm
