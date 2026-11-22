import json

import pytest
from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers

from local_rag_chat.tokenization import QwenPromptTokenizer


@pytest.fixture
def qwen_tokenizer(tmp_path):
    """A real byte-BPE tokenizer with Qwen framing, built without model downloads.

    No merges are trained: tests stress the budget logic rather than model quality.
    The production Qwen vocabulary is exercised separately by the opt-in test.
    """
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE({character: i for i, character in enumerate(alphabet)}, []))
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.add_special_tokens([
        AddedToken("<|im_start|>", special=True), AddedToken("<|im_end|>", special=True),
    ])
    directory = tmp_path / "tokenizer"
    directory.mkdir()
    backend.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "Qwen2Tokenizer", "add_bos_token": False,
    }), encoding="utf-8")
    return QwenPromptTokenizer(directory)
