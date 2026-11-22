"""Ollama's local HTTP API behind the runtime-independent LLMClient contract."""

import json
import math
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .generation import LLMError


CONTEXT_WINDOW = 6_000


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OllamaClient:
    """Synchronous, non-streaming inference using an already-installed local model."""

    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        host: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ) -> None:
        if not model.strip():
            raise ValueError("Model name must not be blank.")
        address = urlsplit(host)
        if (
            address.scheme != "http"
            or address.hostname not in {"127.0.0.1", "localhost", "::1"}
            or address.username is not None
            or address.password is not None
            or address.path not in {"", "/"}
            or address.query
            or address.fragment
        ):
            raise ValueError("Ollama host must be a loopback HTTP URL without a path.")
        if address.port is not None and not 1 <= address.port <= 65535:
            raise ValueError("Ollama host port must be between 1 and 65535.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Request timeout must be a positive, finite number.")

        self.model = model.strip()
        self.host = host.rstrip("/")
        self.timeout = timeout
        # Disable HTTP proxies to prevent routing prompts through another server
        # Reject redirects as well, so requests cannot silently leave the configured local endpoint.
        self._http = build_opener(ProxyHandler({}), _NoRedirects())

    def generate(self, prompt: str, max_tokens: int) -> str:
        if not prompt.strip():
            raise ValueError("Prompt must not be blank.")
        if not isinstance(max_tokens, int) or not 0 < max_tokens < CONTEXT_WINDOW:
            raise ValueError(f"Maximum output tokens must be an integer from 1 to {CONTEXT_WINDOW-1}.")

        # A local Ollama server can also proxy cloud models. Check the model before
        # sending any prompt;
        info = self._post_json("/api/show", {"model": self.model})
        if info.get("remote_host") or info.get("remote_model"):
            raise LLMError(f"Model '{self.model}' uses remote inference; choose a local model.")
        if not isinstance(info.get("model_info"), dict) or not info["model_info"]:
            raise LLMError(f"Cannot verify local model metadata for '{self.model}'.")

        result = self._post_json("/api/generate", {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": CONTEXT_WINDOW,
                "temperature": 0,
            },
        })
        text = result.get("response")
        if result.get("done") is not True or not isinstance(text, str) or not text.strip():
            raise LLMError("Ollama returned an incomplete or empty text response.")
        return text.strip()

    def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        request = Request(
            self.host + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._http.open(request, timeout=self.timeout) as response:
                body = response.read()
        except HTTPError as error:
            with error:
                raw_error = error.read()
            try:
                detail = json.loads(raw_error).get("error", error.reason)
            except (ValueError, AttributeError):
                detail = error.reason
            if error.code == 404:
                raise LLMError(
                    f"Ollama could not find model '{self.model}'. Install it with "
                    f"'ollama pull {self.model}' and check the configured host."
                ) from error
            raise LLMError(f"Ollama request failed (HTTP {error.code}): {detail}") from error
        except TimeoutError as error:
            raise LLMError(
                f"Ollama request timed out after {self.timeout:g} seconds. "
                "Model loading may need a longer timeout."
            ) from error
        except URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise LLMError(f"Ollama request timed out after {self.timeout:g} seconds.") from error
            raise LLMError(
                f"Cannot reach Ollama at {self.host}. Start it with 'ollama serve' "
                "or open the Ollama app, and check the host and port."
            ) from error
        except (OSError, HTTPException) as error:
            raise LLMError(f"Ollama connection failed: {error}") from error

        try:
            result = json.loads(body)
        except ValueError as error:
            raise LLMError("Ollama returned invalid JSON.") from error
        if not isinstance(result, dict):
            raise LLMError("Ollama returned a JSON value instead of an object.")
        if result.get("error"):
            raise LLMError(f"Ollama request failed: {result['error']}")
        return result
