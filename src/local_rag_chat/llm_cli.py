"""Standalone local inference smoke test; imports no retrieval components."""

import argparse

from .generation import LLMClient, LLMError
from .ollama import OllamaClient


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Send a prompt to a local Ollama model.")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Installed local model name")
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Loopback Ollama URL")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum output tokens")
    parser.add_argument("--timeout", type=float, default=120, help="Per-request timeout in seconds")
    parser.add_argument("--prompt", default="Explain what RAG is in one sentence.")
    args = parser.parse_args(argv)

    try:
        client: LLMClient = OllamaClient(model=args.model, host=args.host, timeout=args.timeout)
        answer = client.generate(args.prompt, max_tokens=args.max_tokens)
    except (ValueError, LLMError) as error:
        parser.exit(1, f"error: {error}\n")
    print(answer)


if __name__ == "__main__":
    main()
