# Custom HTTP Headers

MicroCore provides flexible control over HTTP headers sent to remote LLM providers.

Target applications includes request tracing, traffic tagging, authentication behind proxies, 
passing custom meta-data, integrating with LLM observability platforms, etc.

## Global Configuration

To apply headers to **every** LLM request, define `HTTP_HEADERS` configuration option: 

You can configure headers programmatically via `mc.configure()` or using environment variables, .env files.

```python
import microcore as mc

mc.configure(
    # ... other settings ...
    HTTP_HEADERS={
        "X-Client-ID": "MyApp",
        "X-Client-Version": "3.0.0",
        "X-Request-Source": "production-pipeline",
    }
)
```

Or via environment variable (JSON format):

```bash
HTTP_HEADERS='{"X-Client": "MyApp", "X-Trace-ID": "abc123"}'
```
> **Note:** ⚠️ HTTP headers are ignored when using local models.
> If you configure `HTTP_HEADERS` with `ApiType.FUNCTION` or `ApiType.TRANSFORMERS`, you'll get a warning.

## Per-Request Headers

Headers can be injected dynamically for specific requests using the `extra_headers` parameter in the `llm` function.
These headers merge with and override global headers if there are key conflicts.
```python
from microcore import llm
res = llm("Hello there!", extra_headers={"Helicone-Session-Id": "123"})
```

> **Note**: Headers provided via `extra_headers` will override any conflicting keys defined in the global `HTTP_HEADERS` configuration for that specific request.

> **Note**: When using ApiType.GOOGLE, custom headers also may be set using `http_options` parameter. In case if both are provided, MicroCore merges them, with `extra_headers` taking precedence on key conflicts.
## Supported Backends

Custom headers are supported for all major remote API backends:

- mc.ApiType.OPENAI
- mc.ApiType.ANTHROPIC
- mc.ApiType.GOOGLE

## Merging Behavior

Headers merge with (not replace) any existing headers you've set via provider-specific `INIT_PARAMS`. Your `HTTP_HEADERS` values win on collision:

```python
import microcore as mc

mc.configure(
    INIT_PARAMS={"default_headers": {"Authorization": "Bearer xyz"}},
    HTTP_HEADERS={"X-Custom": "value"}  # Both headers will be sent
)
```
