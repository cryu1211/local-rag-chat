"""Local prompt serialization and counting for the supported Qwen2 text model."""

import json
import unicodedata
from collections.abc import Callable
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

    def validate_model_info(self, info: dict[str, object]) -> None:
        """Reject runtime metadata that does not support this tokenizer's bound."""
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
        try:
            config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
            data = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid tokenizer JSON in {directory}. Re-provision the tokenizer assets.") from error
        if (
            not isinstance(config, dict) or not isinstance(data, dict)
            or not isinstance(data.get("model"), dict)
        ):
            raise ValueError(f"Invalid tokenizer asset schema in {directory}. Re-provision the tokenizer assets.")
        if (
            config.get("tokenizer_class") != "Qwen2Tokenizer"
            or config.get("add_bos_token") is not False
            or data.get("model", {}).get("type") != "BPE"
            or data.get("normalizer") not in (None, {"type": "NFC"})
        ):
            raise ValueError("Prompt budgeting currently supports the Qwen2 byte-BPE tokenizer only.")

        from tokenizers import Tokenizer

        try:
            self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        except Exception as error:
            # The native loader reports malformed BPE assets as a generic Exception.
            # Translate only this external call, not unrelated application errors.
            raise ValueError(
                f"Cannot load tokenizer assets from {directory}: {error}. Re-provision the tokenizer assets."
            ) from error
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

    def validate_model_info(self, info: dict[str, object]) -> None:
        metadata = info.get("model_info")
        details = info.get("details")
        if (
            not isinstance(details, dict) or details.get("format") != "gguf"
            or not isinstance(metadata, dict)
            or metadata.get("general.architecture") != "qwen2"
            or metadata.get("tokenizer.ggml.model") != "gpt2"
            or metadata.get("tokenizer.ggml.pre") != "qwen2"
            or metadata.get("tokenizer.ggml.add_bos_token") is not False
            or metadata.get("tokenizer.ggml.add_eos_token") not in (None, False)
        ):
            raise ValueError("Bounded generation requires the supported Qwen2 GGUF byte-BPE model.")


# Factories, not shared instances: tokenizer assets are loaded only on demand.
TOKENIZER_FACTORIES: dict[str, Callable[[Path, str], PromptTokenizer]] = {
    "qwen2.5:1.5b": QwenPromptTokenizer,
}


def load_prompt_tokenizer(model: str, directory: Path) -> PromptTokenizer:
    model = model.strip()
    factory = TOKENIZER_FACTORIES.get(model)
    if factory is None:
        raise ValueError(
            f"No prompt tokenizer is registered for '{model}'. "
            f"Supported models: {', '.join(TOKENIZER_FACTORIES)}."
        )
    return factory(directory, model)
