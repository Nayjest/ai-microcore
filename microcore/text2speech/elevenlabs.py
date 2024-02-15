import os

import aiohttp

from .._env import env
from datetime import datetime


async def text_to_speech(
    text: str,
    out_file: str = None,
    voice: str = "D38z5RcWu1voky8WS1ja",
    stability=0.29,
    similarity_boost=0.5,
    style=0.0,
    chunk_size=1024,
) -> str:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
    if not out_file:
        base_name = datetime.now().strftime("%Y-%m-%d_%H_%M_%S__%f")
        os.makedirs(env().config.TEXT_TO_SPEECH_PATH, exist_ok=True)
        out_file = f"{env().config.TEXT_TO_SPEECH_PATH}/{base_name}.mp3"
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
        },
    }
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": env().config.ELEVENLABS_API_KEY,
    }
    async with aiohttp.request("POST", url, json=data, headers=headers) as response:
        if response.status != 200:
            r = await response.json()
            raise Exception(f"Bad response status: {response.status}: {r}")
        with open(out_file, "wb") as file:
            async for chunk in response.content.iter_chunked(chunk_size):
                if chunk:
                    file.write(chunk)
        return out_file
