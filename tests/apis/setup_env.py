import glob
import os
import pytest
from colorama import Fore as c
import logging
import microcore

envs = glob.glob(".env.test.*")
# envs = ['.env.test.open_ai']
# envs = ['.env.test.azure']
# envs = ['.env.test.anyscale']
# envs = ['.env.test.open_ai-instruct']


@pytest.fixture(params=envs)
def setup_env(request):
    logging.info(f"\n{c.MAGENTA}===  [ SETUP ENV {request.param} ] ===")
    original_env = dict(os.environ)
    os.environ.clear()
    microcore.configure(
        DOT_ENV_FILE=request.param, LLM_DEFAULT_ARGS=dict(temperature=0.01)
    )
    yield
    os.environ.clear()
    os.environ.update(original_env)
