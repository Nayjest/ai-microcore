"""
Usage:
uv run https://raw.githubusercontent.com/Nayjest/ai-microcore/refs/heads/main/examples/pytorch_transformers.py
"""
# /// script
# dependencies = ["ai-microcore>=4.4.0", "transformers", "torch"]
# ///
import microcore as mc


print('Initializing...')
mc.configure(
    LLM_API_TYPE=mc.ApiType.TRANSFORMERS,
    MODEL='Qwen/Qwen3-0.6B',
    CHAT_MODE=True,
    USE_LOGGING=mc.PRINT_STREAM,
)
print('Starting inference...')
result = mc.llm("/no_think Count from 1 to 20")
print("\nGeneration duration:", result.gen_duration, "seconds")
