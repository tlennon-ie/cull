"""Tests for the four cloud vision workers (OpenAI / OpenRouter / Anthropic /
Gemini), registered alongside Groq via the ``vision_workers`` registry.

These workers each subclass ``BaseVisionWorker`` and send the strict vision
JSON schema (``vision_prompt.build_response_format``) using their provider's
structured-output mechanism:

  * OpenAI / OpenRouter  -> ``response_format={"type": "json_schema", ...}``
  * Anthropic            -> forced tool-use whose ``input_schema`` is the schema
  * Gemini               -> ``response_mime_type`` + ``response_schema``

The provider SDKs are NOT installed in CI, so every test injects a fake SDK
module into ``sys.modules`` before importing the worker. No network calls are
ever made.

Runnable:
    pytest tests/test_cloud_vision_workers.py -q
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise ``dotenv.load_dotenv`` so a developer's real ``.env`` can't
    leak provider keys into these hermetic tests (the workers call it at import
    time). Each test controls the environment purely via ``monkeypatch``."""
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    # Strip any provider keys already present in the process environment so a
    # delenv-then-reimport cycle starts from a clean slate.
    for key in (
        "OPENAI_API_KEY", "OPENAI_VISION_MODEL", "OPENAI_BASE_URL",
        "OPENROUTER_API_KEY", "OPENROUTER_VISION_MODEL", "OPENROUTER_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_VISION_MODEL", "ANTHROPIC_BASE_URL",
        "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_VISION_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


# ── A schema-valid model answer reused across providers ──────────────────────

def _valid_classification() -> dict[str, Any]:
    """A dict that satisfies vision_prompt's schema's required keys."""
    return {
        "description": "A literal description of the whole image for testing.",
        "primary_subject": "a test subject",
        "is_screenshot": False,
        "is_composite_grid": False,
        "contains_text_overlay": False,
        "is_human_photograph": True,
        "art_medium": "photograph",
        "photorealistic_style": True,
        "has_ai_flaws": False,
        "woman_present": False,
        "nsfw": False,
        "OVR_Quality_Score": 71,
        "REL_Quality_Score": 64,
        "quality_score": 7,
        "category": "DISCARD",
        "reason": "Just a deterministic test answer.",
        "caption": "",
    }


# ── Fake OpenAI / OpenRouter SDK ─────────────────────────────────────────────

class _FakeOpenAICapture:
    """Captures the kwargs passed to ``client.chat.completions.create``."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None
        self.payload: dict[str, Any] = _valid_classification()
        self.init_kwargs: dict[str, Any] | None = None


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> _FakeOpenAICapture:
    capture = _FakeOpenAICapture()
    mod = types.ModuleType("openai")

    class _Completions:
        def create(self, **kwargs: Any) -> Any:
            capture.last_kwargs = kwargs
            content = json.dumps(capture.payload)
            message = types.SimpleNamespace(content=content, model_dump=lambda: {"content": content})
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

    class _Chat:
        def __init__(self) -> None:
            self.completions = _Completions()

    class OpenAI:  # noqa: N801 - mirror the SDK's class name
        def __init__(self, **kwargs: Any) -> None:
            capture.init_kwargs = kwargs
            self.chat = _Chat()

    mod.OpenAI = OpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mod)
    return capture


# ── Fake Anthropic SDK ───────────────────────────────────────────────────────

class _FakeAnthropicCapture:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None
        self.payload: dict[str, Any] = _valid_classification()
        self.init_kwargs: dict[str, Any] | None = None


def _install_fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> _FakeAnthropicCapture:
    capture = _FakeAnthropicCapture()
    mod = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kwargs: Any) -> Any:
            capture.last_kwargs = kwargs
            # Anthropic returns content blocks; a forced tool_use block carries
            # the structured result in its `input` field.
            tool_block = types.SimpleNamespace(
                type="tool_use",
                name="classify_image",
                input=dict(capture.payload),
            )
            return types.SimpleNamespace(content=[tool_block], stop_reason="tool_use")

    class Anthropic:  # noqa: N801 - mirror the SDK's class name
        def __init__(self, **kwargs: Any) -> None:
            capture.init_kwargs = kwargs
            self.messages = _Messages()

    mod.Anthropic = Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return capture


# ── Fake google.genai SDK ────────────────────────────────────────────────────

class _FakeGeminiCapture:
    def __init__(self) -> None:
        self.model_calls: list[dict[str, Any]] = []
        self.payload: dict[str, Any] = _valid_classification()
        self.init_kwargs: dict[str, Any] | None = None


def _install_fake_gemini(monkeypatch: pytest.MonkeyPatch) -> _FakeGeminiCapture:
    capture = _FakeGeminiCapture()
    genai = types.ModuleType("google.genai")
    google_pkg = types.ModuleType("google")

    # types submodule: provide the helpers the worker uses for config/parts.
    genai_types = types.ModuleType("google.genai.types")

    class _Part:
        @staticmethod
        def from_bytes(*, data: bytes, mime_type: str) -> Any:
            return {"inline_data": {"data": data, "mime_type": mime_type}}

        @staticmethod
        def from_text(*, text: str) -> Any:
            return {"text": text}

    class _GenerateContentConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    genai_types.Part = _Part  # type: ignore[attr-defined]
    genai_types.GenerateContentConfig = _GenerateContentConfig  # type: ignore[attr-defined]

    class _Models:
        def generate_content(self, **kwargs: Any) -> Any:
            capture.model_calls.append(kwargs)
            return types.SimpleNamespace(text=json.dumps(capture.payload))

    class Client:  # noqa: N801 - mirror the SDK's class name
        def __init__(self, **kwargs: Any) -> None:
            capture.init_kwargs = kwargs
            self.models = _Models()

    genai.Client = Client  # type: ignore[attr-defined]
    genai.types = genai_types  # type: ignore[attr-defined]
    google_pkg.genai = genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types)
    return capture


# ── helpers ──────────────────────────────────────────────────────────────────

def _reimport(name: str):
    """Import (or re-import) a worker module fresh so the gated SDK import in
    its body picks up the fake module we just installed."""
    sys.modules.pop(name, None)
    return importlib.import_module(name)


B64 = "ZmFrZS1qcGVn"  # base64("fake-jpeg")
INSTRUCTION = "classify this test image"


# ── module import smoke ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "name",
    [
        "vision_worker_openai",
        "vision_worker_openrouter",
        "vision_worker_anthropic",
        "vision_worker_gemini",
    ],
)
def test_worker_module_imports_without_sdk(name: str) -> None:
    """Modules must import cleanly even when the heavy SDK is absent."""
    sys.modules.pop(name, None)
    mod = importlib.import_module(name)
    assert hasattr(mod, "run_vision_worker")


# ── registry wiring ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["openai", "anthropic", "gemini", "openrouter"])
def test_worker_registered(name: str) -> None:
    import vision_workers
    importlib.reload(vision_workers)
    assert name in vision_workers.WORKERS
    spec = vision_workers.WORKERS[name]
    assert spec.script == f"vision_worker_{name}.py"
    assert spec.description


def test_registry_keeps_legacy_entries() -> None:
    """Appending cloud workers must not drop the back-compat names."""
    import vision_workers
    importlib.reload(vision_workers)
    for legacy in ("balanced-groq", "balanced-lm", "balanced-lm-secondary",
                   "lm-autodetect", "balanced-openai-compat", "balanced-ollama",
                   "local"):
        assert legacy in vision_workers.WORKERS


# ── OpenAI worker ────────────────────────────────────────────────────────────

def test_openai_sends_schema_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_VISION_MODEL", "gpt-test-vision")

    mod = _reimport("vision_worker_openai")
    worker = object.__new__(mod.OpenAIVisionWorker)
    mod.OpenAIVisionWorker.__init__(worker)

    result = worker.classify_image_bytes(B64, INSTRUCTION)
    assert result is not None
    assert result["category"] == "DISCARD"
    assert result["OVR_Quality_Score"] == 71

    kwargs = capture.last_kwargs
    assert kwargs is not None
    # The strict vision schema must be present in response_format.
    rf = kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "VisionClassification"
    assert "category" in rf["json_schema"]["schema"]["properties"]
    # The image must be sent as a data URL and the model name must be honoured.
    # Post-perf-pass the OpenAI worker splits into system (long rubric, cached)
    # + user (image + short reference) so we scan every message for the image
    # block rather than assuming messages[0] carries it.
    assert kwargs["model"] == "gpt-test-vision"
    image_block = None
    for msg in kwargs["messages"]:
        content = msg.get("content")
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "image_url":
                    image_block = b
                    break
        if image_block:
            break
    assert image_block is not None
    assert image_block["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_openai_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mod = _reimport("vision_worker_openai")
    with pytest.raises(SystemExit):
        worker = object.__new__(mod.OpenAIVisionWorker)
        mod.OpenAIVisionWorker.__init__(worker)


# ── OpenRouter worker ────────────────────────────────────────────────────────

def test_openrouter_uses_router_base_url_and_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_VISION_MODEL", "some/vision-model")

    mod = _reimport("vision_worker_openrouter")
    worker = object.__new__(mod.OpenRouterVisionWorker)
    mod.OpenRouterVisionWorker.__init__(worker)

    result = worker.classify_image_bytes(B64, INSTRUCTION)
    assert result is not None and result["category"] == "DISCARD"

    # Constructed against the OpenRouter base URL with the OpenRouter key.
    assert capture.init_kwargs is not None
    assert capture.init_kwargs.get("base_url") == "https://openrouter.ai/api/v1"
    assert capture.init_kwargs.get("api_key") == "sk-or-test"
    # Schema still travels in response_format.
    rf = capture.last_kwargs["response_format"]
    assert rf["json_schema"]["name"] == "VisionClassification"
    assert capture.last_kwargs["model"] == "some/vision-model"


# ── Anthropic worker ─────────────────────────────────────────────────────────

def test_anthropic_forces_tool_with_schema_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _install_fake_anthropic(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_VISION_MODEL", "claude-test-vision")

    mod = _reimport("vision_worker_anthropic")
    worker = object.__new__(mod.AnthropicVisionWorker)
    mod.AnthropicVisionWorker.__init__(worker)

    result = worker.classify_image_bytes(B64, INSTRUCTION)
    assert result is not None
    assert result["category"] == "DISCARD"
    assert result["primary_subject"] == "a test subject"

    kwargs = capture.last_kwargs
    assert kwargs is not None
    assert kwargs["model"] == "claude-test-vision"
    # A single tool whose input_schema is the vision schema.
    tools = kwargs["tools"]
    assert len(tools) == 1
    schema = tools[0]["input_schema"]
    assert "category" in schema["properties"]
    assert "description" in schema["properties"]
    # tool_choice must FORCE that exact tool.
    assert kwargs["tool_choice"]["type"] == "tool"
    assert kwargs["tool_choice"]["name"] == tools[0]["name"]
    # Image sent as a base64 image block.
    content = kwargs["messages"][0]["content"]
    image_block = next(b for b in content if b.get("type") == "image")
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/jpeg"
    assert image_block["source"]["data"] == B64


def test_anthropic_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_anthropic(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    mod = _reimport("vision_worker_anthropic")
    with pytest.raises(SystemExit):
        worker = object.__new__(mod.AnthropicVisionWorker)
        mod.AnthropicVisionWorker.__init__(worker)


# ── Gemini worker ────────────────────────────────────────────────────────────

def test_gemini_sends_response_schema_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _install_fake_gemini(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    monkeypatch.setenv("GEMINI_VISION_MODEL", "gemini-test-vision")

    mod = _reimport("vision_worker_gemini")
    worker = object.__new__(mod.GeminiVisionWorker)
    mod.GeminiVisionWorker.__init__(worker)

    result = worker.classify_image_bytes(B64, INSTRUCTION)
    assert result is not None
    assert result["category"] == "DISCARD"

    assert capture.model_calls, "generate_content was never called"
    call = capture.model_calls[0]
    assert call["model"] == "gemini-test-vision"
    cfg = call["config"]
    # Structured output: JSON mime type + a response_schema derived from ours.
    assert cfg.kwargs["response_mime_type"] == "application/json"
    schema = cfg.kwargs["response_schema"]
    assert "category" in schema["properties"]
    assert "description" in schema["properties"]


def test_gemini_falls_back_to_google_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _install_fake_gemini(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-fallback")

    mod = _reimport("vision_worker_gemini")
    worker = object.__new__(mod.GeminiVisionWorker)
    mod.GeminiVisionWorker.__init__(worker)
    assert capture.init_kwargs is not None
    assert capture.init_kwargs.get("api_key") == "google-fallback"


def test_gemini_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_gemini(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    mod = _reimport("vision_worker_gemini")
    with pytest.raises(SystemExit):
        worker = object.__new__(mod.GeminiVisionWorker)
        mod.GeminiVisionWorker.__init__(worker)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
