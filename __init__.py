import dotenv
from os import PathLike
from jinja2 import Environment, FileSystemLoader, ChoiceLoader
dotenv.load_dotenv()
import openai


def tpl(file: PathLike[str] | str, **kwargs): return j2env.get_template(file).render(**kwargs)


def llm(prompt: str | list[str], **kwargs):
    if isinstance(prompt, list): prompt = '\n'.join(prompt)
    args = {**llm_default_args, **kwargs, 'messages': [{"role": "user", "content": prompt}]}
    return openai.ChatCompletion.create(**args).choices[0].message.content


j2env = Environment(loader=ChoiceLoader([FileSystemLoader('tpl')]))
llm_default_args = {
    'model': 'gpt-4',
}
