from dataclasses import dataclass, field
from importlib.util import find_spec

from .config import Config
from .embedding_db.base import EmbeddingDB
from .types import TplFunctionType
from .templating.jinja2 import make_jinja2_env, make_tpl_function
from .llm.openai_llm import make_llm_function
import jinja2
from .logging import use_logging


@dataclass
class Env:
    config: Config
    jinjaEnvironment: jinja2.Environment = None
    tpl_function: TplFunctionType = None
    llm_function: TplFunctionType = None
    llm_before_handlers: list[callable] = field(default_factory=list)
    llm_after_handlers: list[callable] = field(default_factory=list)
    texts: EmbeddingDB = None

    def __post_init__(self):
        global _env
        _env = self
        self.init_templating()
        self.init_llm()
        if self.config.USE_LOGGING:
            use_logging()
        self.init_similarity_search()

    def init_templating(self):
        self.jinjaEnvironment = make_jinja2_env(self)
        self.tpl_function = make_tpl_function(self)

    def init_llm(self):
        self.llm_function = make_llm_function(self.config)

    def init_similarity_search(self):
        if find_spec("chromadb") is not None:
            from .embedding_db.chromadb import ChromaEmbeddingDB

            self.texts = ChromaEmbeddingDB(self.config)


_env: Env | None = None


def env() -> Env:
    global _env
    _env or Env(Config())
    return _env


@dataclass
class _Configure(Config):
    def __post_init__(self):
        super().__post_init__()
        Env(self)


configure: callable = _Configure
