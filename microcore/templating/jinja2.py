import os
from pathlib import Path

import jinja2
from ..types import TplFunctionType


def make_jinja2_env(env) -> jinja2.Environment:
    j2 = jinja2.Environment(
        autoescape=env.config.JINJA2_AUTO_ESCAPE,
        loader=jinja2.ChoiceLoader([
            jinja2.FileSystemLoader(env.config.PROMPT_TEMPLATES_PATH),
            jinja2.FileSystemLoader(Path(__file__).parent.parent / "ai_func"),
        ]),
    )
    j2.globals.update(
        env=env,
        config=env.config,
        **env.config.JINJA2_GLOBALS,
    )
    return j2


def make_tpl_function(env) -> TplFunctionType:
    def tpl(file: os.PathLike[str] | str, **kwargs) -> str:
        return env.jinja_env.get_template(file).render(**kwargs)

    return tpl
