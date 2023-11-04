import builtins
import yaml
from jinja2 import PackageLoader
from ._env import env

_original_import = builtins.__import__


def custom_import(name, global_vars=None, local_vars=None, fromlist=(), level=0):
    module = _original_import(name, global_vars, local_vars, fromlist, level)
    if module.__doc__ and "ai_module" in module.__doc__:
        data = yaml.safe_load(module.__doc__)
        tpl_path = data.get("tpl_path", "")
        env().jinja_env.loader.loaders.append(PackageLoader(module.__name__, tpl_path))
    return module


builtins.__import__ = custom_import
