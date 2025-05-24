import os
from dataclasses import dataclass, asdict
from datetime import datetime
import aiohttp
from .._env import env


@dataclass
class TTSArgs:
    text: str
    out_file: str = None
    voice: str = "D38z5RcWu1voky8WS1ja"
    stability: float = 0.29
    similarity_boost: float = 0.5
    style: float = 0.0
    chunk_size: int = 1024
    speed: float = 1.0
    use_speaker_boost: bool = False
    previous_text: str = None
    next_text: str = None

    def to_dict(self) -> dict:
        return asdict(self)


async def text_to_speech(  # pylint: disable=R0914
    text: str,
    out_file: str = None,
    voice: str = "D38z5RcWu1voky8WS1ja",
    stability=0.29,
    similarity_boost=0.5,
    style=0.0,
    chunk_size=1024,
    speed=1.0,
    use_speaker_boost: bool = False,
    previous_text: str = None,
    next_text: str = None,
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
            "speed": speed,
        },
    }
    if use_speaker_boost:
        data["voice_settings"]["use_speaker_boost"] = use_speaker_boost
    if previous_text:
        data["previous_text"] = previous_text
    if next_text:
        data["next_text"] = next_text
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": env().config.ELEVENLABS_API_KEY,
    }
    async with aiohttp.request("POST", url, json=data, headers=headers) as response:
        if response.status != 200:
            r = await response.json()
            raise RuntimeError(f"Bad response status: {response.status}: {r}")
        with open(out_file, "wb") as file:
            async for chunk in response.content.iter_chunked(chunk_size):
                if chunk:
                    file.write(chunk)
        return out_file
