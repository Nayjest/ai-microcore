from microcore._llm_functions import convert_exception
from microcore import LLMContextLengthExceededError
from httpx import Request, Response
from openai import BadRequestError as OpenAIBadRequestError
from anthropic import BadRequestError as AnthropicBadRequestError
import google.genai.errors


def test_convert_unsupported_exception():
    class CustomException(Exception):
        pass

    e = CustomException("This is a custom exception message.")
    ce = convert_exception(e)
    assert ce is None


def test_anthropic_bad_request_context_length_exceeded():
    error = AnthropicBadRequestError(
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'prompt is too long: 200037 tokens > 200001 maximum'}, 'request_id': 'req_011CW6wXL4zz2k4enwfeVmTi'}",
        response=Response(
            status_code=400,
            request=Request('POST', 'https://api.anthropic.com/v1/messages')
        ),
        body={
            'type': 'error',
            'error': {
                'type': 'invalid_request_error',
                'message': 'prompt is too long: 200037 tokens > 200001 maximum'
            },
            'request_id': 'req_011CW6wXL4zz2k4enwfeVmTi'
        }
    )
    ce = convert_exception(error, 'my_model')
    assert isinstance(ce, LLMContextLengthExceededError)
    assert ce.actual_tokens == 200037
    assert ce.max_tokens == 200001
    assert ce.model == 'my_model'


def test_openai_bad_request_context_length_exceeded():
    error = OpenAIBadRequestError(
        "Error code: 400 - {'error': {'message': \"This model's maximum context length is 16385 tokens. However, your messages resulted in 600007 tokens. Please reduce the length of the messages.\", 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}",  # noqa: E501
        response=Response(
            status_code=400,
            request=Request('POST', 'https://api.openai.com/v1/chat/completions')
        ),
        body={
            'message': "This model's maximum context length is 16385 tokens. However, your messages resulted in 600007 tokens. Please reduce the length of the messages.",  # noqa: E501
            'type': 'invalid_request_error',
            'param': 'messages',
            'code': 'context_length_exceeded'
        }
    )

    ce = convert_exception(error, 'my_model')
    assert isinstance(ce, LLMContextLengthExceededError)
    assert ce.actual_tokens == 600007
    assert ce.max_tokens == 16385
    assert ce.model == 'my_model'


def test_google_ai_studio_bad_request_context_length_exceeded():
    error = google.genai.errors.ClientError(
        code=400,
        response_json={
            'error': {
                'code': 400,
                'message': 'The input token count exceeds the maximum number of tokens allowed 1048577.',  # noqa: E501
                'status': 'INVALID_ARGUMENT'
            }
        }
    )
    ce = convert_exception(error, 'my_model')
    assert isinstance(ce, LLMContextLengthExceededError)
    assert ce.actual_tokens is None
    assert ce.max_tokens == 1048577
    assert ce.model == 'my_model'


def test_google_vertex_bad_request_context_length_exceeded():
    error = google.genai.errors.ClientError(
        code=400,
        response_json={
            'error': {
                'code': 400,
                'message': 'The input token count (1600009) exceeds the maximum number of tokens allowed (1048576).',  # noqa: E501
                'status': 'INVALID_ARGUMENT'
            }
        }
    )
    ce = convert_exception(error, 'my_model')
    assert isinstance(ce, LLMContextLengthExceededError)
    assert ce.actual_tokens == 1600009
    assert ce.max_tokens == 1048576
    assert ce.model == 'my_model'
