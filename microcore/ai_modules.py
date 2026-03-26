from typing import Optional
from types import ModuleType
import builtins
import yaml

from jinja2 import PackageLoader, PrefixLoader
from pydantic import BaseModel, Field

from ._env import env


class AIModuleConfig(BaseModel):
    """
    Configuration for an AI module, read from the module's docstring.
    """
    ai_module: str = Field(description="Programmatic name of the AI module")
    tpl_path: str = Field(
        default="",
        description="Path to the templates directory within the module, "
                    "relative to the module's root. "
                    "If not specified, defaults to the module's root directory."
    )
    tpl_prefix: Optional[str] = Field(
        default="",
        description="Optional prefix to use for template names from this module. "
                    "If not specified, ai_module name will be used as prefix."
    )
    use_tpl_prefix: Optional[bool] = Field(
        default=True,
        description="Whether to use the tpl_prefix for template names. "
                    "If false, templates will be loaded without any prefix."
    )
    use_templates: bool = Field(
        default=True,
        description="Whether to load templates from this module. "
                    "Set to false to skip loading templates even if tpl_path is specified."
    )
    package_name: str = Field(
        description="The name of python package which is AI module, "
                    "automatically set when reading configuration."
    )

    @staticmethod
    def read(module: ModuleType) -> Optional["AIModuleConfig"]:
        """
        Reads AI module configuration from the module's docstring
        if it contains 'ai_module' key.
        """
        if module.__doc__ and "ai_module" in module.__doc__:
            data = yaml.safe_load(module.__doc__)
            return AIModuleConfig(**data, package_name=module.__name__)
        return None


def _custom_import(name, global_vars=None, local_vars=None, fromlist=(), level=0):
    module = _original_import(name, global_vars, local_vars, fromlist, level)
    if config := AIModuleConfig.read(module):
        if config.package_name not in _ai_modules:
            _ai_modules[config.package_name] = config
            _register_module_tpl_loader(config)
    return module


def _register_module_tpl_loader(config: AIModuleConfig) -> None:
    """Registers a Jinja template loader for the given AI module config."""
    if config.use_templates:
        pkg_loader = PackageLoader(config.package_name, config.tpl_path)
        if config.use_tpl_prefix:
            pkg_loader = PrefixLoader({config.tpl_prefix or config.ai_module: pkg_loader})
        env().jinja_env.loader.loaders.append(pkg_loader)


def register_module_tpl_loaders() -> None:
    """
    When env is recreated (during configure() call, etc.),
    all previously registered Jinja loaders are cleared and need to be re-registered
    using this function.
    """
    for module_config in _ai_modules.values():
        _register_module_tpl_loader(module_config)


_original_import = builtins.__import__
builtins.__import__ = _custom_import
_ai_modules: dict[str, AIModuleConfig] = {}
