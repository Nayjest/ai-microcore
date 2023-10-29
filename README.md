# MicroCore: A Minimalistic Foundation for Large Language Model Applications

This package is rooted in a design philosophy that promotes following guiding principles:
* Keeping it simple
* Keep implementation details away from business logic
* Emphasis on working conveniently with complex prompt templates
* Laconic code

By adhering to these principles, MicroCore aims to deliver a range of benefits 
that address the specific needs of the rapidly evolving field of AI application development:

* Rapid prototyping / RnD / experimentation
* Smooth sub-systems interchangeability without need of massive codebase changes
* Code designed to be well accepted by AI agents with introspective code analysis and generation
* Clean and easy-to-follow examples to aid learning and development

## Architectural approaches
* Python functions as tools for AI agents
* Python packages as AI Modules
* Templating with inheritance and blocks for prompt composition 
* Business-logic layers with shared interfaces and smoothly interchangeable implementations:
  * LLM (OpenAI, BLOOM, Koala, etc)
  * Embedding databases (Chroma, Pinecone, Milvus, etc)
  * Communication (HTTP, WebSockets, Console, etc)

## Core Functions

### *llm(prompt: str, \*\*kwargs) →* str

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

# To specify a message role, you can use dictionary
llm(dict(role='user', content='1+2='))

# Or use specific classes
from microcore.openai_chat import UserMsg, SysMsg

llm(SysMsg('1+2='))

# The returned value is a string
out = llm([SysMsg('1+2=3'), UserMsg('3+4=')]).upper()

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

### *tpl(file_path, \*\*params) →* str
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

### *store(collection: str | None, **kwargs)
Stores data in embeddings database of your choice

@TODO

### *search(collection: str | None, **kwargs)
Performs semantic / similarity search over embeddings database

@TODO


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
* Automatically registers template folders of AI modules in Jinja2 environment
 

## License

© 2023&mdash;∞ Vitalii Stepanenko

Licensed under the MIT License. 