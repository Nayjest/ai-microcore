import os
from . import * # noqa


def test_valid(): # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    mc.configure(USE_DOT_ENV=False, LLM_API_KEY='123')
    mc.validate_config()
    os.environ.update(original_env)

def test_no_key(): # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    mc.configure(USE_DOT_ENV=False)
    with pytest.raises(mc.LLMConfigError):
        mc.validate_config()
    os.environ.update(original_env)

def test_azure_no_deployment(): # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    mc.configure(
        USE_DOT_ENV=False,
        LLM_API_TYPE=mc.ApiType.AZURE,
        LLM_API_KEY='123',
        LLM_API_VERSION='123',
        LLM_API_BASE='https://example.com'
    )
    with pytest.raises(mc.LLMConfigError):
        mc.validate_config()
    os.environ.update(original_env)

def test_azure_ok(): # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    mc.configure(
        USE_DOT_ENV=False,
        LLM_API_TYPE=mc.ApiType.AZURE,
        LLM_API_KEY='123',
        LLM_DEPLOYMENT_ID='123',
        LLM_API_VERSION='123',
        LLM_API_BASE='https://example.com'
    )

    mc.validate_config()
    os.environ.update(original_env)

def test_azure_no_version(): # noqa
    original_env = dict(os.environ)
    os.environ.clear()
    mc.configure(
        USE_DOT_ENV=False,
        LLM_API_TYPE=mc.ApiType.AZURE,
        LLM_API_KEY='123',
        LLM_DEPLOYMENT_ID='123',
        LLM_API_BASE='https://example.com'
    )
    with pytest.raises(mc.LLMConfigError):
        mc.validate_config()
    os.environ.update(original_env)
