import sys
import asyncio
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
    critical_vars = {
        k: v for k, v in os.environ.items()
        if k.upper() in (
           'SYSTEMROOT', 'PATH', 'COMSPEC', 'TEMP', 'TMP', 'SYSTEMDRIVE', 'USERPROFILE'
        )
    }
    os.environ.clear()
    os.environ.update(critical_vars)
    microcore.configure(
        USE_DOT_ENV=True,
        DOT_ENV_FILE=request.param,
    )
    yield
    os.environ.clear()
    os.environ.update(original_env)
