# MicroCore: A Minimalistic Foundation for Large Language Model Applications

This package is rooted in a design philosophy that promotes following guiding principles:
* Maximum simplicity: Keeping It Simple (KIS) is elevated to its peak.
* Separation of business logic: This encourages the decoupling of business logic from implementation specifics.
* Emphasis on working conveniently with complex prompt templates.
* Laconic code.

By adhering to this principles, MicroCore aims to deliver a range of benefits 
that address the specific needs of the rapidly evolving field of AI application development:

* Rapid prototyping / RnD / experimentation.
* Smooth sub-systems interchangeability without need of massive codebase changes.
* Code designed be well accepted by AI agents with introspective code generation.
* Clean and easy-to-follow examples to aid learning and development.

## Architectural approaches
* Python functions as tools for AI agents
* Python packages as AI Modules
* Templating with inheritance and blocks for prompt composition 
* Business-logic layers with shared interfaces and smoothly interchangeable implementations:
* * LLM (OpenAI, BLOOM, Koala, etc)
* * Embedding databases (Chroma, Pinecone, Milvus, etc)
* * Communication (HTTP, WebSockets, Console, etc)

## Core Functions

### *llm(prompt, \*\*kwargs) →* str
Performs a request to a large language model (LLM).
```python
# Examples
import microcore.logging # Automatically adds debug logging of LLM requests when imported
from microcore import *

# Basic usage
ai_response = llm('What is your model name?')
# Lists are allowed. 
# For chat LLMs elements are treated as separate messages
# For completion LLMs elements are treated as text lines
llm(['1+2', '='])
llm('1+2=', model='gpt-4')

# To specify a message role, you can use dictionary
llm(dict(role='user', content='1+2='))

# Or use specific classes
from microcore.openai_chat import UserMsg, SysMsg
llm(SysMsg('1+2='))

# The returned value is a string
string = llm([SysMsg('1+2=3'), UserMsg('3+4=')]).upper()

# But it contains all fields of the LLM response in additional attributes
for i in llm('1+2=?', n=3, temperature=2).choices:
    print('RESPONSE:', i.message.content)
```

### *tpl(file_path, \*\*vars) →* str
Renders a Jinja2 template.

The default templates path is *\<your app working dir>/tpl*

It can be changed in APP_TPL_PATH environment variable.

Alternatively, you may monkey patch Jinja2 loader in microcore.jinja2_env.

This loader is dynamically extended when importing MicroCore AI modules.

So, you can override templates of 3rd-party modules by placing templates with same names into your tpl folder.

### *store(collection: str | None, **kwargs)
Stores data in embeddings database of your choice

@TODO

### *search(collection: str | None, **kwargs)
Performs semantic / similarity search over embeddings database

@TODO

## MicroCore AI Modules
By importing microcore.ai_modules, you enable auto-registering additional Jinja2 template roots
for modules containing 'ai_module' string in module doc.string

## Python functions as AI tools

@TODO

## License

© 2023&mdash;∞ Vitalii Stepanenko

Licensed under the MIT License. 