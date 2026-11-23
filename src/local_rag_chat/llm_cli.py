"""Standalone local inference smoke test; imports no retrieval components."""

import argparse
from pathlib import Path

from .generation import LLMClient, LLMError
from .ollama import OllamaClient
from .tokenization import load_prompt_tokenizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Send a prompt to a local Ollama model.")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Installed local model name")
    parser.add_argument(
        "--tokenizer-path", type=Path, default=Path("models/qwen2.5-1.5b-tokenizer"),
        help="Local tokenizer directory for the registered model profile",
    )
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Loopback Ollama URL")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum output tokens")
    parser.add_argument("--timeout", type=float, default=120, help="Per-request timeout in seconds")
    parser.add_argument("--prompt", default="Explain what RAG is in one sentence.")
    args = parser.parse_args(argv)

    try:
        if not args.prompt.strip():
            raise ValueError("Prompt must not be blank.")
        tokenizer = load_prompt_tokenizer(args.model, args.tokenizer_path)
        prompt = tokenizer.serialize("You are a helpful assistant.", args.prompt)
        client: LLMClient = OllamaClient(
            model=args.model, host=args.host, timeout=args.timeout, prompt_tokenizer=tokenizer,
        )
        answer = client.generate(prompt, max_tokens=args.max_tokens)
    except (OSError, ValueError, LLMError) as error:
        parser.exit(1, f"error: {error}\n")
    print(answer)


if __name__ == "__main__":
    main()
