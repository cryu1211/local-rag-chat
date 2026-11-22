import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from local_rag_chat import llm_cli
from local_rag_chat.generation import LLMError
from local_rag_chat.ollama import OllamaClient
from local_rag_chat.prompting import build_prompt


LOCAL_MODEL_INFO = {"model_info": {"general.architecture": "qwen2"}}
COMPLETE_RESPONSE = {"response": "RAG retrieves context before generating an answer.", "done": True}


@pytest.fixture
def fake_ollama():
    """Exercise real HTTP serialization with a deterministic loopback server."""
    state = SimpleNamespace(
        requests=[], replies=[(200, LOCAL_MODEL_INFO), (200, COMPLETE_RESPONSE)],
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append((self.path, payload))
            status, body = state.replies.pop(0)
            body = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if status == 302:
                self.send_header("Location", "http://127.0.0.1:1/redirected")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.host = f"http://127.0.0.1:{server.server_port}"
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_local_metadata_check_then_non_streaming_generation(fake_ollama, monkeypatch):
    # Even an explicitly configured HTTP proxy must not receive the prompt.
    for name in ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"]:
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    client = OllamaClient(model="test-model", host=fake_ollama.host)

    assert client.generate("Explain RAG.\nUse café as an example.", 32) == COMPLETE_RESPONSE["response"]
    assert fake_ollama.requests == [
        ("/api/show", {"model": "test-model"}),
        ("/api/generate", {
            "model": "test-model", "prompt": "Explain RAG.\nUse café as an example.",
            "stream": False,
            "options": {"num_predict": 32, "num_ctx": 6000, "temperature": 0},
        }),
    ]


@pytest.mark.parametrize("info", [
    {"remote_host": "https://ollama.com", **LOCAL_MODEL_INFO},
    {"remote_model": "remote-model", **LOCAL_MODEL_INFO},
    {}, {"model_info": {}}, {"model_info": "invalid"},
])
def test_unverified_or_remote_models_rejected_before_prompt_is_sent(fake_ollama, info):
    fake_ollama.replies = [(200, info)]
    with pytest.raises(LLMError, match="remote inference|Cannot verify"):
        OllamaClient(host=fake_ollama.host).generate("Private prompt", 32)
    assert len(fake_ollama.requests) == 1
    assert "prompt" not in fake_ollama.requests[0][1]


def test_missing_model_has_manual_provisioning_instruction(fake_ollama):
    fake_ollama.replies = [(404, {"error": "model 'missing' not found"})]
    with pytest.raises(LLMError, match="ollama pull missing"):
        OllamaClient(model="missing", host=fake_ollama.host).generate("Hello", 32)
    assert len(fake_ollama.requests) == 1  # No automatic download or inference.


@pytest.mark.parametrize("status,body,message", [
    (500, {"error": "not enough memory"}, "HTTP 500.*not enough memory"),
    (503, b"server busy", "HTTP 503"),
    (200, {"error": "inference failed"}, "inference failed"),
    (200, b"not json", "invalid JSON"),
    (200, [], "instead of an object"),
    (200, {"response": "partial", "done": False}, "incomplete or empty"),
    (200, {"response": " \n", "done": True}, "incomplete or empty"),
    (200, {"response": 42, "done": True}, "incomplete or empty"),
])
def test_inference_failures_are_clear(fake_ollama, status, body, message):
    fake_ollama.replies[1] = (status, body)
    with pytest.raises(LLMError, match=message):
        OllamaClient(host=fake_ollama.host).generate("Explain RAG", 32)
    assert len(fake_ollama.requests) == 2  # Failed requests are not retried.


def test_redirects_are_not_followed(fake_ollama):
    fake_ollama.replies = [(302, {})]
    with pytest.raises(LLMError, match="HTTP 302"):
        OllamaClient(host=fake_ollama.host).generate("Hello", 32)
    assert len(fake_ollama.requests) == 1


def test_stopped_server_has_actionable_error(monkeypatch):
    client = OllamaClient()

    def connection_refused(request, timeout):
        raise URLError(ConnectionRefusedError("Connection refused"))

    monkeypatch.setattr(client._http, "open", connection_refused)
    with pytest.raises(LLMError, match="Cannot reach Ollama.*ollama serve"):
        client.generate("Hello", 32)


def test_request_timeout_is_bounded_and_reported(monkeypatch):
    client = OllamaClient(timeout=0.1)

    def timed_out(request, timeout):
        assert timeout == 0.1
        raise TimeoutError("timed out")

    monkeypatch.setattr(client._http, "open", timed_out)
    with pytest.raises(LLMError, match="timed out after 0.1 seconds"):
        client.generate("Hello", 32)


@pytest.mark.parametrize("host", [
    "https://ollama.com", "http://192.168.1.2:11434", "http://0.0.0.0:11434",
    "http://localhost/api", "http://user:password@localhost:11434",
    "http://localhost?endpoint=cloud", "http://localhost#fragment", "http://localhost:0",
])
def test_non_local_or_invalid_hosts_are_rejected(host):
    with pytest.raises(ValueError):
        OllamaClient(host=host)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeouts_rejected(timeout):
    with pytest.raises(ValueError, match="timeout"):
        OllamaClient(timeout=timeout)


@pytest.mark.parametrize("prompt,max_tokens", [(" ", 32), ("Hello", 0), ("Hello", -1),
                                               ("Hello", 6000), ("Hello", 2.5), ("Hello", True)])
def test_invalid_generation_input_fails_before_network_request(fake_ollama, prompt, max_tokens):
    with pytest.raises(ValueError):
        OllamaClient(host=fake_ollama.host).generate(prompt, max_tokens)
    assert fake_ollama.requests == []


def test_cli_configuration_and_output(fake_ollama, capsys):
    llm_cli.main(["--model", "test-model", "--host", fake_ollama.host,
                  "--max-tokens", "16", "--timeout", "3", "--prompt", "Explain RAG"])
    output = capsys.readouterr()
    assert output.out == COMPLETE_RESPONSE["response"] + "\n"
    assert output.err == ""
    assert fake_ollama.requests[1][1]["options"]["num_predict"] == 16
    assert fake_ollama.requests[1][1]["model"] == "test-model"


def test_cli_failure_prints_error_without_traceback(fake_ollama, capsys):
    fake_ollama.replies = [(500, {"error": "not enough memory"})]
    with pytest.raises(SystemExit) as error:
        llm_cli.main(["--host", fake_ollama.host])
    assert error.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "error: Ollama request failed (HTTP 500): not enough memory" in output.err
    assert "Traceback" not in output.err


def test_llm_entry_point_does_not_import_retrieval_native_libraries():
    completed = subprocess.run(
        [sys.executable, "-c", "import sys; import local_rag_chat.llm_cli; "
         "assert not {'faiss', 'torch', 'sentence_transformers', "
         "'local_rag_chat.retrieval'} & sys.modules.keys()"],
        capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr


def test_real_ollama_smoke():
    if os.environ.get("RUN_OLLAMA_SMOKE") != "1":
        pytest.skip("Set RUN_OLLAMA_SMOKE=1 with Ollama and a local model already provisioned.")
    client = OllamaClient(
        model=os.environ.get("OLLAMA_TEST_MODEL", "qwen2.5:1.5b"),
        host=os.environ.get("OLLAMA_TEST_HOST", "http://127.0.0.1:11434"),
    )
    assert client.generate("Explain what RAG is in one sentence.", 128).strip()


QWEN_MODEL_INFO = {
    "details": {"format": "gguf"},
    "model_info": {
        "general.architecture": "qwen2", "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2", "tokenizer.ggml.add_bos_token": False,
    },
}


def test_bounded_generation_uses_exact_wire_prompt_and_output_reserve(fake_ollama, qwen_tokenizer):
    prompt = build_prompt("What is known?", [], qwen_tokenizer, reserved_output_tokens=32)
    fake_ollama.replies = [
        (200, QWEN_MODEL_INFO),
        (200, {**COMPLETE_RESPONSE, "prompt_eval_count": prompt.usage.total, "eval_count": 12}),
    ]
    client = OllamaClient(host=fake_ollama.host, prompt_tokenizer=qwen_tokenizer)

    assert client.generate(prompt.text, prompt.reserved_output_tokens) == COMPLETE_RESPONSE["response"]
    request = fake_ollama.requests[1][1]
    assert request["prompt"] == prompt.text
    assert request["raw"] is True
    assert request["truncate"] is False
    assert request["shift"] is False
    assert request["options"]["num_predict"] == prompt.reserved_output_tokens
    assert request["options"]["num_ctx"] == 6000


def test_bounded_client_rechecks_modified_prompt_before_network_request(fake_ollama, qwen_tokenizer):
    client = OllamaClient(host=fake_ollama.host, prompt_tokenizer=qwen_tokenizer)
    with pytest.raises(ValueError, match="exceeds the 6000-token"):
        client.generate("x" * 6000, 32)
    assert fake_ollama.requests == []
    with pytest.raises(ValueError, match="same model"):
        OllamaClient(model="different-model", prompt_tokenizer=qwen_tokenizer)


@pytest.mark.parametrize("key,value", [
    ("general.architecture", "llama"), ("tokenizer.ggml.model", "llama"),
    ("tokenizer.ggml.pre", "unknown"), ("tokenizer.ggml.add_bos_token", True),
])
def test_bounded_client_rejects_unsupported_runtime_tokenizers(fake_ollama, qwen_tokenizer, key, value):
    info = {**QWEN_MODEL_INFO, "model_info": {**QWEN_MODEL_INFO["model_info"], key: value}}
    fake_ollama.replies = [(200, info)]
    prompt = build_prompt("Question?", [], qwen_tokenizer)
    with pytest.raises(LLMError, match="supported Qwen2 GGUF byte-BPE"):
        OllamaClient(host=fake_ollama.host, prompt_tokenizer=qwen_tokenizer).generate(prompt.text, 32)
    assert len(fake_ollama.requests) == 1


@pytest.mark.parametrize("counts", [{}, {"prompt_eval_count": 6001, "eval_count": 1},
                                     {"prompt_eval_count": 100, "eval_count": 33}])
def test_bounded_client_checks_runtime_reported_counts(fake_ollama, qwen_tokenizer, counts):
    fake_ollama.replies = [(200, QWEN_MODEL_INFO), (200, {**COMPLETE_RESPONSE, **counts})]
    prompt = build_prompt("Question?", [], qwen_tokenizer)
    with pytest.raises(LLMError, match="do not satisfy"):
        OllamaClient(host=fake_ollama.host, prompt_tokenizer=qwen_tokenizer).generate(prompt.text, 32)


def test_real_ollama_bounded_prompt():
    tokenizer_path = os.environ.get("LOCAL_QWEN_TOKENIZER_PATH")
    if os.environ.get("RUN_OLLAMA_SMOKE") != "1" or not tokenizer_path:
        pytest.skip("Set RUN_OLLAMA_SMOKE=1 and LOCAL_QWEN_TOKENIZER_PATH for bounded inference.")
    from pathlib import Path
    from local_rag_chat.tokenization import QwenPromptTokenizer

    model = os.environ.get("OLLAMA_TEST_MODEL", "qwen2.5:1.5b")
    tokenizer = QwenPromptTokenizer(Path(tokenizer_path), model=model)
    prompt = build_prompt("What information is available?", [], tokenizer, reserved_output_tokens=64)
    client = OllamaClient(
        model=model, host=os.environ.get("OLLAMA_TEST_HOST", "http://127.0.0.1:11434"),
        prompt_tokenizer=tokenizer,
    )
    assert client.generate(prompt.text, prompt.reserved_output_tokens).strip()
