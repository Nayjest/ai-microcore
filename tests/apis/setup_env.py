import os
import glob
import logging

from colorama import Fore as c
import pytest
import microcore as mc

envs = glob.glob(".env.test.*")
# envs = ['.env.test.openai.low-end']
# envs = ['.env.test.azure']
# envs = ['.env.test.anthropic']


@pytest.fixture(params=envs)
def setup_env(request):
    logging.info(f"\n{c.MAGENTA}===  [ SETUP ENV {request.param} ] ===")
    original_env = dict(os.environ)
    critical_vars = {
        k: v for k, v in os.environ.items()
        if k.upper() in (
           'SYSTEMROOT', 'PATH', 'COMSPEC', 'TEMP', 'TMP', 'SYSTEMDRIVE', 'USERPROFILE'
        )
    }
    os.environ.clear()
    os.environ.update(critical_vars)
    mc.configure(
        USE_DOT_ENV=True,
        DOT_ENV_FILE=request.param,
        HTTP_HEADERS={
            "X-Client": "AI MicroCore",
            "X-Client-Version": mc.__version__,
        },
    )
    yield
    os.environ.clear()
    os.environ.update(original_env)
