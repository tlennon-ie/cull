"""Tests for ``vision_model_catalog`` — best-effort model discovery for cloud
and local vision providers, plus the cloud workers' autodetect-on-empty-model
behaviour.

The catalog NEVER raises and returns ``[]`` on any error. Each provider parses
its own REST/SDK response shape into a flat ``list[str]`` of model ids:

  * openai / openrouter / lmstudio / llamacpp / openai-compat
        GET ``{base}/v1/models`` (Bearer key) -> ``data[].id``
  * ollama
        GET ``{base}/api/tags`` -> ``models[].name``
  * anthropic
        GET ``https://api.anthropic.com/v1/models`` (x-api-key +
        anthropic-version) -> ``data[].id``
  * gemini
        ``google-genai`` ``client.models.list()`` if available, else
        GET ``.../v1beta/models?key=`` -> ``models[].name`` filtered to ids
        that advertise ``generateContent``.

No real network is ever hit: ``requests`` and the genai SDK are monkeypatched.

Runnable:
    pytest tests/test_vision_model_catalog.py -q
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))

import vision_model_catalog as catalog  # noqa: E402


# ── A fake requests.get that returns a canned response ───────────────────────

class _FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: Any = None,
                 text: str = "", raise_exc: Exception | None = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text
        self._raise_exc = raise_exc

    def json(self) -> Any:
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._payload


class _GetRecorder:
    """Captures the args of the last ``requests.get`` call and replays a canned
    response (or raises a canned exception)."""

    def __init__(self, response: _FakeResponse | None = None,
                 exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if self.exc is not None:
            raise self.exc
        assert self.response is not None
        return self.response


def _patch_get(monkeypatch: pytest.MonkeyPatch, recorder: _GetRecorder) -> None:
    monkeypatch.setattr(catalog.requests, "get", recorder)


# ── openai / openrouter / lmstudio / llamacpp / openai-compat ────────────────

@pytest.mark.parametrize(
    "provider",
    ["openai", "openrouter", "lmstudio", "llamacpp", "openai-compat", "openai_compat"],
)
def test_openai_family_parses_data_id(monkeypatch: pytest.MonkeyPatch,
                                      provider: str) -> None:
    payload = {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}, {"no_id": 1}]}
    rec = _GetRecorder(_FakeResponse(payload=payload))
    _patch_get(monkeypatch, rec)

    models = catalog.list_models(provider, base_url="https://example.test/v1",
                                 api_key="sk-test")
    assert models == ["gpt-4o", "gpt-4o-mini"]
    # Bearer auth header sent; redirects disabled.
    call = rec.calls[0]
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["allow_redirects"] is False
    assert call["url"].endswith("/v1/models")


def test_openai_default_base_url_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _GetRecorder(_FakeResponse(payload={"data": [{"id": "gpt-4o"}]}))
    _patch_get(monkeypatch, rec)
    models = catalog.list_models("openai", api_key="sk-test")
    assert models == ["gpt-4o"]
    assert rec.calls[0]["url"] == "https://api.openai.com/v1/models"


def test_openrouter_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _GetRecorder(_FakeResponse(payload={"data": [{"id": "openai/gpt-4o"}]}))
    _patch_get(monkeypatch, rec)
    models = catalog.list_models("openrouter")
    assert models == ["openai/gpt-4o"]
    assert rec.calls[0]["url"] == "https://openrouter.ai/api/v1/models"


def test_base_url_with_trailing_v1_is_not_doubled(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _GetRecorder(_FakeResponse(payload={"data": [{"id": "m"}]}))
    _patch_get(monkeypatch, rec)
    catalog.list_models("lmstudio", base_url="http://127.0.0.1:1234/v1/")
    # Trailing /v1 (and slash) collapsed to a single /v1/models.
    assert rec.calls[0]["url"] == "http://127.0.0.1:1234/v1/models"


def test_no_api_key_sends_no_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _GetRecorder(_FakeResponse(payload={"data": []}))
    _patch_get(monkeypatch, rec)
    catalog.list_models("lmstudio", base_url="http://127.0.0.1:1234")
    assert "Authorization" not in rec.calls[0].get("headers", {})


# ── ollama ───────────────────────────────────────────────────────────────────

def test_ollama_parses_models_name(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"models": [{"name": "llava:13b"}, {"name": "qwen2-vl"}, {"x": 1}]}
    rec = _GetRecorder(_FakeResponse(payload=payload))
    _patch_get(monkeypatch, rec)
    models = catalog.list_models("ollama", base_url="http://127.0.0.1:11434")
    assert models == ["llava:13b", "qwen2-vl"]
    assert rec.calls[0]["url"].endswith("/api/tags")
    assert rec.calls[0]["allow_redirects"] is False


# ── anthropic ────────────────────────────────────────────────────────────────

def test_anthropic_parses_data_id_with_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"data": [{"id": "claude-3-5-sonnet-latest"},
                        {"id": "claude-3-opus-latest"}]}
    rec = _GetRecorder(_FakeResponse(payload=payload))
    _patch_get(monkeypatch, rec)
    models = catalog.list_models("anthropic", api_key="sk-ant-test")
    assert models == ["claude-3-5-sonnet-latest", "claude-3-opus-latest"]
    call = rec.calls[0]
    assert call["url"] == "https://api.anthropic.com/v1/models"
    assert call["headers"]["x-api-key"] == "sk-ant-test"
    assert "anthropic-version" in call["headers"]


def test_anthropic_without_key_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _GetRecorder(_FakeResponse(payload={"data": [{"id": "x"}]}))
    _patch_get(monkeypatch, rec)
    # No key -> the catalog should not even attempt the call (anthropic requires a key).
    assert catalog.list_models("anthropic") == []
    assert rec.calls == []


# ── gemini (REST fallback) ───────────────────────────────────────────────────

def test_gemini_rest_filters_generatecontent(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "models": [
            {"name": "models/gemini-2.0-flash",
             "supportedGenerationMethods": ["generateContent", "countTokens"]},
            {"name": "models/text-embedding-004",
             "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/gemini-1.5-pro",
             "supportedGenerationMethods": ["generateContent"]},
        ]
    }
    rec = _GetRecorder(_FakeResponse(payload=payload))
    _patch_get(monkeypatch, rec)
    # Ensure the genai SDK path is not taken for this test.
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    models = catalog.list_models("gemini", api_key="gm-test")
    # Only generateContent-capable models, with the "models/" prefix stripped.
    assert models == ["gemini-2.0-flash", "gemini-1.5-pro"]
    # The key travels in the x-goog-api-key header, NOT the URL query string —
    # _safe_get logs the URL at DEBUG on failure, so a ?key= param would leak it.
    assert "key=gm-test" not in rec.calls[0]["url"]
    assert "key=" not in rec.calls[0]["url"]
    assert rec.calls[0]["headers"]["x-goog-api-key"] == "gm-test"


def test_gemini_without_key_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _GetRecorder(_FakeResponse(payload={"models": []}))
    _patch_get(monkeypatch, rec)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    assert catalog.list_models("gemini") == []
    assert rec.calls == []


def test_gemini_prefers_genai_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``google-genai`` is importable, the catalog uses client.models.list()
    and does not fall back to REST."""
    captured: dict[str, Any] = {}

    class _Model:
        def __init__(self, name: str, methods: list[str]) -> None:
            self.name = name
            self.supported_actions = methods

    class _Models:
        def list(self) -> list[Any]:
            return [
                _Model("models/gemini-2.0-flash", ["generateContent"]),
                _Model("models/embed-only", ["embedContent"]),
            ]

    class Client:  # noqa: N801 - mirror SDK class name
        def __init__(self, **kwargs: Any) -> None:
            captured["init"] = kwargs
            self.models = _Models()

    genai = types.ModuleType("google.genai")
    genai.Client = Client  # type: ignore[attr-defined]
    google_pkg = types.ModuleType("google")
    google_pkg.genai = genai  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    # If the SDK path is taken, requests.get must NOT be called.
    rec = _GetRecorder(exc=AssertionError("REST fallback should not run"))
    _patch_get(monkeypatch, rec)

    models = catalog.list_models("gemini", api_key="gm-test")
    assert models == ["gemini-2.0-flash"]
    assert captured["init"].get("api_key") == "gm-test"


# ── error handling: never raise, always [] ───────────────────────────────────

def test_unknown_provider_returns_empty() -> None:
    assert catalog.list_models("does-not-exist") == []


def test_request_exception_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests as _rq
    rec = _GetRecorder(exc=_rq.RequestException("boom"))
    _patch_get(monkeypatch, rec)
    assert catalog.list_models("openai", api_key="sk-test") == []


def test_non_200_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _GetRecorder(_FakeResponse(status_code=401, text="nope"))
    _patch_get(monkeypatch, rec)
    assert catalog.list_models("openai", api_key="sk-test") == []


def test_non_json_body_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _GetRecorder(_FakeResponse(status_code=200, raise_exc=ValueError("not json")))
    _patch_get(monkeypatch, rec)
    assert catalog.list_models("openai", api_key="sk-test") == []


def test_non_http_base_url_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _GetRecorder(exc=AssertionError("must not probe a non-http scheme"))
    _patch_get(monkeypatch, rec)
    # file:// and other schemes are rejected before any network call.
    assert catalog.list_models("openai-compat", base_url="file:///etc/passwd") == []
    assert rec.calls == []


# ── pick_vision_model helper ─────────────────────────────────────────────────

def test_pick_vision_model_prefers_vision_hint() -> None:
    ids = ["text-only-model", "some-vl-model", "another"]
    assert catalog.pick_vision_model(ids) == "some-vl-model"


def test_pick_vision_model_falls_back_to_first() -> None:
    ids = ["alpha", "beta"]
    assert catalog.pick_vision_model(ids) == "alpha"


def test_pick_vision_model_empty_is_none() -> None:
    assert catalog.pick_vision_model([]) is None


# ── worker autodetect uses the catalog when the model env is unset ───────────

def _reimport(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop the workers' import-time ``load_dotenv`` from leaking a real .env."""
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for key in ("OPENAI_API_KEY", "OPENAI_VISION_MODEL", "OPENAI_BASE_URL",
                "OPENROUTER_API_KEY", "OPENROUTER_VISION_MODEL", "OPENROUTER_BASE_URL",
                "ANTHROPIC_API_KEY", "ANTHROPIC_VISION_MODEL", "ANTHROPIC_BASE_URL",
                "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_VISION_MODEL"):
        monkeypatch.delenv(key, raising=False)


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("openai")

    class OpenAI:  # noqa: N801
        def __init__(self, **kwargs: Any) -> None:
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **k: None)
            )

    mod.OpenAI = OpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mod)


def test_openai_worker_autodetects_model_via_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # No OPENAI_VISION_MODEL set -> must consult the catalog.
    mod = _reimport("vision_worker_openai")

    called: dict[str, Any] = {}

    def fake_list(provider: str, **kwargs: Any) -> list[str]:
        called["provider"] = provider
        called["kwargs"] = kwargs
        return ["gpt-4o-vision-preview", "gpt-3.5-turbo"]

    monkeypatch.setattr(mod.catalog, "list_models", fake_list)

    worker = object.__new__(mod.OpenAIVisionWorker)
    mod.OpenAIVisionWorker.__init__(worker)
    # The catalog was consulted for the openai provider and a vision id chosen.
    assert called["provider"] == "openai"
    assert worker.model_id == "gpt-4o-vision-preview"


def test_openai_worker_falls_back_to_default_when_catalog_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mod = _reimport("vision_worker_openai")
    monkeypatch.setattr(mod.catalog, "list_models", lambda *a, **k: [])

    worker = object.__new__(mod.OpenAIVisionWorker)
    mod.OpenAIVisionWorker.__init__(worker)
    assert worker.model_id == mod.DEFAULT_OPENAI_MODEL


def test_openai_worker_explicit_model_skips_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_VISION_MODEL", "gpt-explicit")
    mod = _reimport("vision_worker_openai")

    def boom(*a: Any, **k: Any) -> list[str]:
        raise AssertionError("catalog must not be consulted when model is set")

    monkeypatch.setattr(mod.catalog, "list_models", boom)
    worker = object.__new__(mod.OpenAIVisionWorker)
    mod.OpenAIVisionWorker.__init__(worker)
    assert worker.model_id == "gpt-explicit"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
