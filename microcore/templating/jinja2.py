import os
import jinja2
from ..types import TplFunctionType


def make_jinja2_env(env) -> jinja2.Environment:
    return jinja2.Environment(
        autoescape=env.config.JINJA2_AUTO_ESCAPE,
        loader=jinja2.ChoiceLoader(
            [jinja2.FileSystemLoader(env.config.PROMPT_TEMPLATES_PATH)]
        ),
    )


def make_tpl_function(env) -> TplFunctionType:
    def tpl(file: os.PathLike[str] | str, **kwargs) -> str:
        return env.jinja_env.get_template(file).render(**kwargs)

    return tpl
