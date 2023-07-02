import microcore
import openai
from microcore.prepare_llm_args import *
from microcore.extended_string import ExtendedString


def llm(prompt, **kwargs):
    args = {**microcore.llm_default_args, **kwargs}
    if 'gpt' in args['model']:
        response = openai.ChatCompletion.create(messages=prepare_chat_messages(prompt), **args)
        return ExtendedString(response.choices[0].message.content, response)
    else:
        response = openai.Completion.create(prompt=prepare_prompt(prompt), **args)
        return ExtendedString(response.choices[0].choices[0].text, response)


microcore.llm = llm
if not microcore.llm_default_args.get('model'):
    microcore.llm_default_args['model'] = 'gpt-3.5-turbo'
