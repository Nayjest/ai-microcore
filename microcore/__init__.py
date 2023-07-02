"""Minimalistic core for large language model applications"""
import dotenv
import jinja2
import os


def llm(prompt, **kwargs):
    """Request LLM"""
    import microcore.openai
    return microcore.openai.llm(prompt, **kwargs)


def tpl(file: os.PathLike[str] | str, **kwargs): return j2env.get_template(file).render(**kwargs)


dotenv.load_dotenv()
llm_default_args = {}
j2env = jinja2.Environment(loader=jinja2.ChoiceLoader([jinja2.FileSystemLoader(os.getenv('APP_TPL_PATH', 'tpl'))]))





