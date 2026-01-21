# AI MicroCore: A Minimalistic Foundation for AI Applications

<p align="center"> 
  <a href="https://pypi.org/project/ai-microcore/" target="_blank"><img src="https://img.shields.io/github/v/release/Nayjest/ai-microcore.svg" alt="Release Notes"></a>
  <a href="https://app.codacy.com/gh/Nayjest/ai-microcore/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade" target="_blank"><img src="https://app.codacy.com/project/badge/Grade/441d03416bc048828c649129530dcbc3" alt="Code Quality"></a>
  <a href="https://github.com/Nayjest/ai-microcore/actions/workflows/pylint.yml" target="_blank"><img src="https://github.com/Nayjest/ai-microcore/actions/workflows/pylint.yml/badge.svg" alt="Pylint"></a>
  <a href="https://github.com/Nayjest/ai-microcore/actions/workflows/tests.yml" target="_blank"><img src="https://github.com/Nayjest/ai-microcore/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <img src="https://raw.githubusercontent.com/Nayjest/ai-microcore/main/coverage.svg" alt="Code Coverage">
  <a href="https://github.com/vshymanskyy/StandWithUkraine/blob/main/README.md" target="_blank"><img src="https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/refs/heads/main/badges/StandWithUkraine.svg" alt="Stand With Ukraine"></a>
  <a href="https://github.com/Nayjest/ai-microcore/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/static/v1?label=license&message=MIT&color=d08aff" alt="License"></a>
</p>


**MicroCore** is a collection of python adapters for Large Language Models
and Vector Databases / Semantic Search APIs allowing to 
communicate with these services in a convenient way, make them easily switchable 
and separate business logic from the implementation details.

It defines interfaces for features typically used in AI applications,
which allows you to keep your application as simple as possible and try various models & services
without need to change your application code.

You can even switch between text completion and chat completion models only using configuration.

Thanks to LLM-agnostic MCP integration,
**MicroCore** connects MCP tools to any language models easily,
whether through API providers that do not support MCP, or through inference using pytorch or arbitrary python functions.

The basic example of usage is as follows:

```python
from microcore import llm

while user_msg := input('Enter message: '):
    print('AI: ' + llm(user_msg))
```

## 🔗 Links

 - [API Reference](https://ai-microcore.github.io/api-reference/)
 - [PyPi Package](https://pypi.org/project/ai-microcore/)
 - [GitHub Repository](https://github.com/Nayjest/ai-microcore)


## 💻 Installation

Install as PyPi package:
```
pip install ai-microcore
```

Alternatively, you may just copy `microcore` folder to your project sources root.
```bash
git clone git@github.com:Nayjest/ai-microcore.git && mv ai-microcore/microcore ./ && rm -rf ai-microcore
```


## 📋 Requirements

Python 3.10 / 3.11 / 3.12 / 3.13 / 3.14

## ⚙️ Configuring

### Minimal Configuration

Having `OPENAI_API_KEY` in OS environment variables is enough for basic usage.

Similarity search features will work out of the box if you have the `chromadb` pip package installed.

### Configuration Methods

There are a few options available for configuring microcore:

-   Use `microcore.configure(**params)`
    <br>💡 <small>All configuration options appear in IDE autocompletion tooltips</small>
-   Create a `.env` file in your project root; examples: [basic.env](https://github.com/Nayjest/ai-microcore/blob/main/.env.example), [Mistral Large.env](https://github.com/Nayjest/ai-microcore/blob/main/.env.mistral.example), [Anthropic Claude 3 Opus.env](https://github.com/Nayjest/ai-microcore/blob/main/.env.anthropic.example), [Gemini on Vertex AI.env](https://github.com/Nayjest/ai-microcore/blob/main/.env.google-vertex-gemini.example), [Gemini on AI Studio.env](https://github.com/Nayjest/ai-microcore/blob/main/.env.gemini.example)
-   Use a custom configuration file: `mc.configure(DOT_ENV_FILE='dev-config.ini')`
-   Define OS environment variables

For the full list of available configuration options, you may also check [`microcore/config.py`](https://github.com/Nayjest/ai-microcore/blob/main/microcore/configuration.py#L175).

### Installing vendor-specific packages
For models working not via OpenAI API, you may need to install additional packages:
#### Anthropic Claude
```bash
pip install anthropic
```
#### Google Gemini via AI Studio or Vertex AI
```bash
pip install google-genai
```

#### Local language models via Hugging Face Transformers

You will need to install transformers and a deep learning library of your choice
(PyTorch, TensorFlow, Flax, etc).

See [transformers installation](https://huggingface.co/docs/transformers/installation).

### Priority of Configuration Sources

1.  Configuration options passed as arguments to `microcore.configure()` have the highest priority.
2.  The priority of configuration file options (`.env` by default or the value of `DOT_ENV_FILE`) is higher than OS environment variables.
    <br>💡 <small>Setting `USE_DOT_ENV` to `false` disables reading configuration files.</small>
3.  OS environment variables have the lowest priority.

### Vector Databases

Vector database functions are available via `microcore.texts`.

#### ChromaDB
The default vector database is [Chroma](https://www.trychroma.com/).
In order to use vector database functions with ChromaDB, you need to install the `chromadb` package:
```bash
pip install chromadb
```
By default, MicroCore will use ChromaDB PersistentClient (if the corresponding package is installed).
Alternatively, you can run Chroma as a separate service and configure MicroCore to use HttpClient:

```python
from microcore import configure
configure(
    EMBEDDING_DB_HOST = 'localhost',
    EMBEDDING_DB_PORT = 8000,
)
```
#### Qdrant
In order to use vector database functions with Qdrant, you need to install the `qdrant-client` package:
```bash
pip install qdrant-client
```
Configuration example
```python
from microcore import configure, EmbeddingDbType
from sentence_transformers import SentenceTransformer

configure(
    EMBEDDING_DB_TYPE=EmbeddingDbType.QDRANT,
    EMBEDDING_DB_HOST="localhost",
    EMBEDDING_DB_PORT="6333",
    EMBEDDING_DB_SIZE=384,  # number of dimensions in the SentenceTransformer model
    EMBEDDING_DB_FUNCTION=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"),
)
```


## 🌟 Core Functions

### llm(prompt: str, \*\*kwargs) → str

Performs a request to a large language model (LLM).

Asynchronous variant: `allm(prompt: str, **kwargs)`

```python
from microcore import *

# Will print all requests and responses to console
use_logging()

# Basic usage
ai_response = llm('What is your model name?')

# You may also pass a list of strings as prompt
# - For chat completion models elements are treated as separate messages
# - For completion LLMs elements are treated as text lines
llm(['1+2', '='])
llm('1+2=', model='gpt-5.2')

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
Store multiple texts and related metadata in the embeddings database

### texts.clear(collection: str):
Clear collection

## API providers and models support

LLM Microcore supports all models & API providers having OpenAI API.

### List of API providers and models tested with LLM Microcore:

| API Provider                                                                             |                                                                                                                                      Models |
|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------:|
| [OpenAI](https://openai.com)                                                             |                                    All GPT-4 and GTP-3.5-Turbo models<br/>all text completion models (davinci, gpt-3.5-turbo-instruct, etc) |
| [Microsoft Azure](https://azure.microsoft.com/en-us/products/ai-services/openai-service) |                                                                                                            All OpenAI models, Mistral Large |
| [Anthropic](https://anthropic.com)                                                       |                                                                                                                             Claude 3 models |
| [MistralAI](https://mistral.ai)                                                          |                                                                                                                          All Mistral models |
| [Google AI Studio](https://aistudio.google.com/)                             |                                                                                                                        Google Gemini models |
| [Google Vertex AI](https://cloud.google.com/vertex-ai?hl=en)                             |                                                   Gemini Pro & [other models](https://cloud.google.com/vertex-ai/docs/start/explore-models) |
| [Deep Infra](https://deepinfra.com)                                                      | deepinfra/airoboros-70b<br/>jondurbin/airoboros-l2-70b-gpt4-1.4.1<br/>meta-llama/Llama-2-70b-chat-hf<br/>and other models having OpenAI API |
| [Anyscale](https://anyscale.com)                                                         |                                           meta-llama/Llama-2-70b-chat-hf<br/>meta-llama/Llama-2-13b-chat-hf<br/>meta-llama/Llama-7b-chat-hf |
| [Groq](https://groq.com/)                                                         |                                           LLaMA2 70b<br>Mixtral 8x7b<br>Gemma 7b |
| [Fireworks](fireworks.ai)                                                         |                                           [Over 50 open-source language models](https://fireworks.ai/models?show=All) |

## Supported local language model APIs:
- HuggingFace [Transformers](https://huggingface.co/docs/transformers/index) (see configuration examples [here](https://github.com/Nayjest/ai-microcore/blob/main/tests/local/test_transformers.py)).
- Custom local models by providing own function for chat / text completion, sync / async inference.

## 🖼️ Examples

#### [Code review tool](https://github.com/llm-microcore/microcore/blob/main/examples/code-review-tool)
Performs a code review by LLM for changes in git .patch files in any programming languages.

#### [Image analysis](https://colab.research.google.com/drive/1qTJ51wxCv3VlyqLt3M8OZ7183YXPFpic) (Google Colab)
Determine the number of petals and the color of the flower from a photo (gpt-4-turbo)

#### [Benchmark LLMs on math problems](https://www.kaggle.com/code/nayjest/gigabenchmark-llm-accuracy-math-problems) (Kaggle Notebook)
Benchmark accuracy of 20+ state of the art models on solving olympiad math problems. Inferencing local language models via HuggingFace Transformers, parallel inference.

#### [Generate meme image](https://github.com/Nayjest/ai-microcore/blob/main/examples/generate_meme_image.py)
Simple example demonstrating image generation using [OpenAI GPT Image](https://platform.openai.com/docs/guides/image-generation?image-generation-model=gpt-image-1) model.


#### [Local inference with PyTorch / Transformers](https://github.com/Nayjest/ai-microcore/blob/main/examples/pytorch_transformers.py)
Text generation using HF/Transformers model locally (example with Qwen 3 0.6B).
 
#### [Other examples](https://github.com/llm-microcore/microcore/tree/main/examples)

## Python functions as AI tools
*Usage Example*:
```python
from microcore.ai_func import ai_func

@ai_func
def search_products(
    query: str,
    category: str = "all",
    max_results: int = 10,
    in_stock_only: bool = False
):
    """
    Search for products in the catalog.

    Args:
        query: Search terms to find matching products
        category: Product category to filter by (e.g., "electronics", "clothing")
        max_results: Maximum number of results to return
        in_stock_only: If True, only return products currently in stock

    Returns:
        List of matching products with name, price, and availability
    """
    # Implementation would go here
    pass
```
*Output*:
```
# Search for products in the catalog.

Args:
    query: Search terms to find matching products
    category: Product category to filter by (e.g., "electronics", "clothing")
    max_results: Maximum number of results to return
    in_stock_only: If True, only return products currently in stock

Returns:
    List of matching products with name, price, and availability
{
  "call": "search_products",
  "query": <str>,
  "category": <str> (default = "all"),
  "max_results": <int> (default = 10),
  "in_stock_only": <bool> (default = False)
}

```

## 🤖 AI Modules
**This is an experimental feature.**

Tweaks the Python import system to provide automatic setup of MicroCore environment
based on metadata in module docstrings.
### Usage:
```python
import microcore.ai_modules
```
### Features:

*   Automatically registers template folders of AI modules in Jinja2 environment

## 🛠️ Contributing

Please see [CONTRIBUTING](https://github.com/Nayjest/ai-microcore/blob/main/CONTRIBUTING.md) for details.


## 📝 License

Licensed under the [MIT License](https://github.com/Nayjest/ai-microcore/blob/main/LICENSE)
© 2023–2026 [Vitalii Stepanenko](mailto:mail@vitaliy.in)
