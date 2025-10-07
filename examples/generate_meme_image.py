"""Usage: uv run https://raw.githubusercontent.com/Nayjest/ai-microcore/refs/heads/main/examples/generate_meme_image.py"""
# /// script
# dependencies = ["ai-microcore>=4.4.0"]
# ///
from microcore import configure, llm, presets
configure(presets.MIN_SETUP)
llm("Meme that makes no sense.", model="gpt-image-1")
