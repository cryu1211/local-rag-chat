"""Local prompt serialization and counting for the supported Qwen2 text model."""

import json
import unicodedata
from pathlib import Path
from typing import Protocol


class PromptTokenizer(Protocol):
    model: str

    def serialize(self, instructions: str, user_text: str) -> str:
        """Render the complete model input, including role and generation markers."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count with the configured local model tokenizer."""
        ...

    def token_upper_bound(self, serialized: str) -> int:
        """Return a conservative bound for this tokenizer and serialization policy."""
        ...


class QwenPromptTokenizer:
    """Qwen2 ChatML plus an offline tokenizer.json; no model weights are loaded.

    The local tokenizer count is informative. Budget enforcement uses the larger
    of that count and a UTF-8 byte bound: Qwen2 byte-BPE tokens each consume at
    least one input byte. Two additional slots cover runtime BOS/EOS handling.
    This bound is deliberately conservative and is not valid for arbitrary models.
    """

    RUNTIME_SPECIAL_TOKEN_ALLOWANCE = 2

    def __init__(self, directory: Path, model: str = "qwen2.5:1.5b") -> None:
        if not model.strip():
            raise ValueError("Tokenizer model name must not be blank.")
        directory = directory.expanduser().resolve()
        config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
        data = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
        if (
            config.get("tokenizer_class") != "Qwen2Tokenizer"
            or config.get("add_bos_token") is not False
            or data.get("model", {}).get("type") != "BPE"
            or data.get("normalizer") not in (None, {"type": "NFC"})
        ):
            raise ValueError("Prompt budgeting currently supports the Qwen2 byte-BPE tokenizer only.")

        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        # These must remain model control tokens, not sequences of ordinary text.
        for marker in ("<|im_start|>", "<|im_end|>"):
            if len(self._tokenizer.encode(marker, add_special_tokens=False).ids) != 1:
                raise ValueError(f"Qwen tokenizer is missing the control token {marker}.")
        self.model = model.strip()

    @staticmethod
    def _content(text: str) -> str:
        # Prevent document/question text from introducing new ChatML roles.
        return unicodedata.normalize("NFC", text).replace("<|", "< |")

    def serialize(self, instructions: str, user_text: str) -> str:
        return (
            f"<|im_start|>system\n{self._content(instructions)}<|im_end|>\n"
            f"<|im_start|>user\n{self._content(user_text)}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def token_upper_bound(self, serialized: str) -> int:
        return max(
            self.count_tokens(serialized),
            len(serialized.encode("utf-8")) + self.RUNTIME_SPECIAL_TOKEN_ALLOWANCE,
        )
