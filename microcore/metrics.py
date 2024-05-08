import time

from .wrappers.llm_response_wrapper import LLMResponse
from ._env import env


class Metrics:
    def __init__(self):
        self._start: float = 0
        self.exec_duration: float = 0
        self.total_gen_duration: float = 0
        self.requests_count: int = 0
        self.succ_requests_count: int = 0
        self.gen_chars_count: int = 0
        self.avg_gen_duration: float = 0
        self.gen_chars_speed: float = 0

    def __enter__(self):
        self._start = time.time()

        env().llm_before_handlers.append(self._before_llm)
        env().llm_after_handlers.append(self._after_llm)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exec_duration = time.time() - self._start
        env().llm_before_handlers.remove(self._before_llm)
        env().llm_after_handlers.remove(self._after_llm)

    def _before_llm(self, prompt, **kwargs):  # pylint: disable=unused-argument
        self.requests_count += 1

    def _after_llm(self, response: str | LLMResponse):
        self.succ_requests_count += 1
        self.gen_chars_count += len(response) if isinstance(response, str) else 0
        self.total_gen_duration += (
            response.gen_duration if isinstance(response, LLMResponse) else 0
        )
        self.avg_gen_duration = self.total_gen_duration / self.succ_requests_count
        self.gen_chars_speed = (self.gen_chars_count or 1) / (
            self.total_gen_duration or 1
        )
