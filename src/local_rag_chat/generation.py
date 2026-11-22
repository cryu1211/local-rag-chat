"""Runtime-independent text generation contract; no retrieval or HTTP imports."""

from typing import Protocol


class LLMError(RuntimeError):
    """The inference runtime could not complete a generation request."""


class LLMClient(Protocol):
    def generate(self, prompt: str, max_tokens: int) -> str:
        """Generate text with an explicit maximum number of output tokens.

        Raise ValueError for invalid inputs and LLMError for runtime failures.
        Complete input-prompt token budgeting belongs to the future prompt builder.
        """
        ...
