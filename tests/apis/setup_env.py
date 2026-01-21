import glob
import os
import logging
from colorama import Fore as c
import pytest
import microcore

envs = glob.glob(".env.test.*")
# envs = ['.env.test.openai.low-end']
# envs = ['.env.test.azure']
# envs = ['.env.test.anthropic']


@pytest.fixture(params=envs)
def setup_env(request):
    logging.info(f"\n{c.MAGENTA}===  [ SETUP ENV {request.param} ] ===")
    original_env = dict(os.environ)
    os.environ.clear()
    microcore.configure(
        USE_DOT_ENV=True,
        DOT_ENV_FILE=request.param,
    )
    yield
    os.environ.clear()
    os.environ.update(original_env)
