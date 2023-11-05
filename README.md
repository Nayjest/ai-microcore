<p align="right">
    <a href="https://github.com/Nayjest/ai-microcore/releases" target="_blank"><img src="https://img.shields.io/github/release/ai-microcore/microcore" alt="Release Notes"></a>
    <a href="https://app.codacy.com/gh/Nayjest/ai-microcore/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade" target="_blank"><img src="https://app.codacy.com/project/badge/Grade/441d03416bc048828c649129530dcbc3" alt="Code Quality"></a>
    <a href="https://github.com/Nayjest/ai-microcore/actions/workflows/pylint.yml" target="_blank"><img src="https://github.com/Nayjest/ai-microcore/actions/workflows/pylint.yml/badge.svg" alt="Pylint"></a>
    <a href="https://github.com/Nayjest/ai-microcore/actions/workflows/tests.yml" target="_blank"><img src="https://github.com/Nayjest/ai-microcore/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
    <a href="https://github.com/Nayjest/ai-microcore/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/static/v1?label=license&message=MIT&color=d08aff" alt="License"></a>
</p>


# AI MicroCore: A Minimalistic Foundation for AI Applications

**microcore** is a collection of python adapters for Large Language Models
and Semantic Search APIs allowing to 
communicate with these services convenient way, make it easily switchable 
and separate business logic from implementation details.

It defines interfaces for features typically used in AI applications,
that allows you to keep your application as simple as possible and try various models & services
without need to change your application code.

You even can switch between text completion and chat completion models only using configuration.

The basic example of usage is as follows:

```python
from microcore import llm

while user_msg := input('Enter message: '):
    print('AI: ' + llm(user_msg))
```

## Links

 -   [API Reference](https://ai-microcore.github.io/api-reference/)
 -   [PyPi Package](https://pypi.org/project/ai-microcore/)
 -   [GitHub Repository](https://github.com/Nayjest/ai-microcore)
## Installation

Install as PyPi package:
```
pip install ai-microcore
```

Alternatively may just copy `microcore` folder to your project sources root.
```bash
git clone git@github.com:Nayjest/ai-microcore.git && mv ai-microcore/microcore ./ && rm -rf ai-microcore
```
## Requirements

Python 3.10+

## Configuring

### Minimal Configuration

Having `OPENAI_API_KEY` in OS environment variables is enough for basic usage.

Similarity search features will work out of the box if you have the `chromadb` pip package installed.

### Configuration Methods

There are a few options available for configuring microcore:

-   Use `microcore.configure()`
    <br>💡 <small>All configuration options should be available in IDE autocompletion tooltips</small>
-   Create a `.env` file in your project root ([example](https://github.com/Nayjest/ai-microcore/blob/main/.env.example))
-   Use a custom configuration file: `mc.configure(DOT_ENV_FILE='dev-config.ini')`
-   Define OS environment variables

For the full list of available configuration options, you may also check [`microcore/config.py`](https://github.com/Nayjest/ai-microcore/blob/main/microcore/config.py).

### Priority of Configuration Sources

1.  Configuration options passed as arguments to `microcore.configure()` have the highest priority.
2.  The priority of configuration file options (`.env` by default or the value of `DOT_ENV_FILE`) is higher than OS environment variables.
    <br>💡 <small>Setting `USE_DOT_ENV` to `false` disables reading configuration files.</small>
3.  OS environment variables has the lowest priority.


## Core Functions

### llm(prompt: str, \*\*kwargs) → str

Performs a request to a large language model (LLM)

```python
from microcore import *

# Will print all requests and responses to console
use_logging()

# Basic usage
ai_response = llm('What is your model name?')

# You also may pass a list of strings as prompt
# - For chat completion models elements are treated as separate messages
# - For completion LLMs elements are treated as text lines
llm(['1+2', '='])
llm('1+2=', model='gpt-4')

# To specify a message role, you can use dictionary or classes
llm(dict(role='system', content='1+2='))
# equivalent
llm(SysMsg('1+2='))

# The returned value is a string
assert '7' == llm([
 SysMsg('You are a calculator'),
 UserMsg('1+2='),
 AssistantMsg('3'),
 UserMsg('3+4=')]
).strip()

# But it contains all fields of the LLM response in additional attributes
for i in llm('1+2=?', n=3, temperature=2).choices:
    print('RESPONSE:', i.message.content)

# To use response streaming you may specify the callback function:
llm('Hi there', callback=lambda x: print(x, end=''))

# Or multiple callbacks:
output = []
llm('Hi there', callbacks=[
    lambda x: print(x, end=''),
    lambda x: output.append(x),
])
```

### tpl(file_path, \*\*params) → str
Renders prompt template with params.

Full-featured Jinja2 templates are used by default.

Related configuration options:

```python
from microcore import configure
configure(
    # 'tpl' folder in current working directory by default
    PROMPT_TEMPLATES_PATH = 'my_templates_folder'
)
```

### texts.search(collection: str, query: str | list, n_results: int = 5, where: dict = None, **kwargs) → list[str]
Similarity search

### texts.find_one(self, collection: str, query: str | list) → str | None
Find most similar text

### texts.get_all(self, collection: str) -> list[str]
Return collection of texts

### texts.save(collection: str, text: str, metadata: dict = None))
Store text and related metadata in embeddings database

### texts.save_many(collection: str, items: list[tuple[str, dict] | str])
Store mutiple texts and related metadata in embeddings database

### texts.clear(collection: str):
Clear collection

## API providers and models support

LLM Microcore supports all models & API providers having OpenAI API.

### List of API providers and models tested with LLM Microcore:

| API Provider                                                                             |                                                                                                                                      Models |
|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------:|
| [OpenAI](openai.com)                                                                     |                                    All GPT-4 and GTP-3.5-Turbo models<br/>all text completion models (davinci, gpt-3.5-turbo-instruct, etc) |
| [Microsoft Azure](https://azure.microsoft.com/en-us/products/ai-services/openai-service) |                                                                                                                           All OpenAI models |
| [deepinfra.com](deepinfra.com)                                                           | deepinfra/airoboros-70b<br/>jondurbin/airoboros-l2-70b-gpt4-1.4.1<br/>meta-llama/Llama-2-70b-chat-hf<br/>and other models having OpenAI API |
 | [Anyscale](anyscale.com)                                                                 |                                           meta-llama/Llama-2-70b-chat-hf<br/>meta-llama/Llama-2-13b-chat-hf<br/>meta-llama/Llama-7b-chat-hf |                                    meta-llama/Llama-2-70b-chat-hf | |


## Examples



#### [code-review-tool example](https://github.com/llm-microcore/microcore/blob/main/examples/code-review-tool)
Performs code review by LLM for changes in git .patch files in any programming languages.

#### [Other examples](https://github.com/llm-microcore/microcore/tree/main/examples)

## Python functions as AI tools

@TODO

## AI Modules
**This is experimental feature.**

Tweaks the Python import system to provide automatic setup of MicroCore environment
based on metadata in module docstrings.
### Usage:
```python
import microcore.ai_modules
```
### Features:

*   Automatically registers template folders of AI modules in Jinja2 environment
 

## License

Licensed under the [MIT License](https://github.com/Nayjest/ai-microcore) © 2023 [Vitalii Stepanenko](mailto:mail@vitalii.in)
