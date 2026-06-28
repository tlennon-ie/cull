"""tests/test_scraper_test.py — pytest suite for pipeline_code/scraper_test.py.

TDD RED phase: all tests written before implementation exists.
Run with:
    .venv/Scripts/python.exe -m pytest tests/test_scraper_test.py -q

All network calls are intercepted at _http_request — no real HTTP in tests.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

# Ensure pipeline_code is importable as a flat namespace (mirrors how the
# existing tests in tests/ do it).
PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


# ---------------------------------------------------------------------------
# Module import helper — lets us reload with a clean env each time.
# ---------------------------------------------------------------------------

def _import_scraper_test():
    """Import (or re-import) scraper_test from pipeline_code/."""
    import importlib
    mod_name = "scraper_test"
    # Remove cached copy so monkeypatching env works across tests.
    sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location(
        mod_name, PIPELINE_CODE / "scraper_test.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def st(monkeypatch):
    """Return the scraper_test module with a clean import."""
    # Isolate from real env
    monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
    monkeypatch.delenv("CIVITAI_API_RED_KEY", raising=False)
    monkeypatch.delenv("TWITTER_COOKIES", raising=False)
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.delenv("DISCORD_AUTH_MODE", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("REDDIT_USER_AGENT", raising=False)
    return _import_scraper_test()


def _patch_http(monkeypatch, mod, status: int, body: str = "ok"):
    """Monkeypatch _http_request to return (status, body) without latency."""
    calls = []

    def fake_http(method, url, *, headers=None, timeout=8.0):
        calls.append({"method": method, "url": url, "headers": headers or {}})
        return (status, body)

    monkeypatch.setattr(mod, "_http_request", fake_http)
    return calls


# ===========================================================================
# 1. SUPPORTED tuple shape
# ===========================================================================

class TestSupportedTuple:
    def test_supported_is_non_empty_tuple(self, st):
        assert isinstance(st.SUPPORTED, tuple)
        assert len(st.SUPPORTED) > 0

    def test_supported_contains_expected_names(self, st):
        expected = {"X.com", "Discord-1", "Civitai-Com", "Civitai-Red",
                    "Reddit", "Web", "Gallery-DL", "Local"}
        for name in expected:
            assert name in st.SUPPORTED, f"{name!r} missing from SUPPORTED"


# ===========================================================================
# 2. Return shape contract
# ===========================================================================

class TestReturnShape:
    def test_result_has_required_keys(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 200)
        monkeypatch.setenv("CIVITAI_API_KEY", "testkey123")
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "testkey123"})
        assert "ok" in result
        assert "message" in result
        assert "latency_ms" in result
        assert "detail" in result

    def test_ok_is_bool(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 200)
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "key"})
        assert isinstance(result["ok"], bool)

    def test_message_is_str(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 200)
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "key"})
        assert isinstance(result["message"], str)

    def test_latency_ms_is_int_or_none(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 200)
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "key"})
        assert result["latency_ms"] is None or isinstance(result["latency_ms"], int)

    def test_detail_is_str(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 200)
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "key"})
        assert isinstance(result["detail"], str)


# ===========================================================================
# 3. Unknown scraper name
# ===========================================================================

class TestUnknownScraper:
    def test_unknown_name_returns_ok_false(self, st):
        result = st.test_scraper("NonExistent-Scraper")
        assert result["ok"] is False

    def test_unknown_name_message_mentions_unsupported(self, st):
        result = st.test_scraper("NonExistent-Scraper")
        assert "unsupported" in result["message"].lower()

    def test_unknown_name_latency_ms_is_none(self, st):
        result = st.test_scraper("NonExistent-Scraper")
        assert result["latency_ms"] is None

    def test_empty_string_name_is_unsupported(self, st):
        result = st.test_scraper("")
        assert result["ok"] is False


# ===========================================================================
# 4. Civitai-Com
# ===========================================================================

class TestCivitaiCom:
    def test_missing_api_key_returns_ok_false(self, st):
        # No env override => no key set
        result = st.test_scraper("Civitai-Com", env={})
        assert result["ok"] is False

    def test_missing_api_key_message_mentions_key_name(self, st):
        result = st.test_scraper("Civitai-Com", env={})
        assert "CIVITAI_API_KEY" in result["message"]

    def test_missing_api_key_latency_ms_is_none(self, st):
        result = st.test_scraper("Civitai-Com", env={})
        assert result["latency_ms"] is None

    def test_missing_api_key_no_network_attempted(self, st, monkeypatch):
        """_http_request must NOT be called when the credential is absent."""
        calls = _patch_http(monkeypatch, st, 200)
        st.test_scraper("Civitai-Com", env={})
        assert calls == [], "No network call expected when credential is missing"

    def test_valid_key_200_returns_ok_true(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 200)
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "mykey"})
        assert result["ok"] is True

    def test_valid_key_200_latency_ms_is_int(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 200)
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "mykey"})
        assert isinstance(result["latency_ms"], int)
        assert result["latency_ms"] >= 0

    def test_valid_key_200_message_ok(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 200)
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "mykey"})
        assert "ok" in result["message"].lower() or "authenticated" in result["message"].lower()

    def test_valid_key_401_returns_ok_false(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 401)
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "badkey"})
        assert result["ok"] is False

    def test_valid_key_401_message_mentions_auth_rejected(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 401)
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "badkey"})
        assert "auth" in result["message"].lower() or "401" in result["message"]

    def test_valid_key_403_returns_ok_false(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 403)
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "badkey"})
        assert result["ok"] is False

    def test_valid_key_hits_civitai_com_url(self, st, monkeypatch):
        calls = _patch_http(monkeypatch, st, 200)
        st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "mykey"})
        assert len(calls) == 1
        assert "civitai.com" in calls[0]["url"]

    def test_timeout_returns_ok_false(self, st, monkeypatch):
        import requests

        def timeout_http(method, url, *, headers=None, timeout=8.0):
            raise requests.exceptions.Timeout("timed out")

        monkeypatch.setattr(st, "_http_request", timeout_http)
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "mykey"})
        assert result["ok"] is False

    def test_timeout_message_mentions_connect(self, st, monkeypatch):
        import requests

        def timeout_http(method, url, *, headers=None, timeout=8.0):
            raise requests.exceptions.Timeout("timed out")

        monkeypatch.setattr(st, "_http_request", timeout_http)
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "mykey"})
        assert "connect" in result["message"].lower() or "timeout" in result["message"].lower()

    def test_timeout_never_raises(self, st, monkeypatch):
        import requests

        def timeout_http(method, url, *, headers=None, timeout=8.0):
            raise requests.exceptions.Timeout("timed out")

        monkeypatch.setattr(st, "_http_request", timeout_http)
        # Must not propagate
        st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "mykey"})


# ===========================================================================
# 5. Civitai-Red reads a DIFFERENT env key
# ===========================================================================

class TestCivitaiRed:
    def test_missing_both_keys_returns_ok_false(self, st):
        result = st.test_scraper("Civitai-Red", env={})
        assert result["ok"] is False

    def test_missing_key_message_mentions_civitai_red_key(self, st):
        result = st.test_scraper("Civitai-Red", env={})
        # Must mention one of the red-specific key names
        msg = result["message"]
        assert "CIVITAI_API_RED_KEY" in msg or "CIVITAI_API_KEY" in msg

    def test_red_key_takes_priority_over_com_key(self, st, monkeypatch):
        """If CIVITAI_API_RED_KEY is set, Civitai-Red should succeed with 200."""
        calls = _patch_http(monkeypatch, st, 200)
        result = st.test_scraper(
            "Civitai-Red",
            env={"CIVITAI_API_RED_KEY": "redkey", "CIVITAI_API_KEY": "comkey"},
        )
        assert result["ok"] is True

    def test_red_key_hits_civitai_red_url(self, st, monkeypatch):
        calls = _patch_http(monkeypatch, st, 200)
        st.test_scraper("Civitai-Red", env={"CIVITAI_API_RED_KEY": "redkey"})
        assert len(calls) == 1
        assert "civitai.red" in calls[0]["url"]

    def test_com_key_fallback_used_when_red_key_absent(self, st, monkeypatch):
        """Civitai-Red should fall back to CIVITAI_API_KEY if RED_KEY is absent."""
        calls = _patch_http(monkeypatch, st, 200)
        result = st.test_scraper("Civitai-Red", env={"CIVITAI_API_KEY": "comkey"})
        assert result["ok"] is True

    def test_civitai_com_and_red_are_independent(self, st, monkeypatch):
        """Civitai-Com should NOT use CIVITAI_API_RED_KEY."""
        calls = _patch_http(monkeypatch, st, 200)
        # Only RED key set; Civitai-Com must fail (no com key)
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_RED_KEY": "redkey"})
        assert result["ok"] is False
        assert calls == [], "No network call when Civitai-Com has no API key"


# ===========================================================================
# 6. X.com / Twitter
# ===========================================================================

class TestXCom:
    def test_missing_cookies_returns_ok_false(self, st):
        result = st.test_scraper("X.com", env={})
        assert result["ok"] is False

    def test_missing_cookies_message_mentions_twitter_cookies(self, st):
        result = st.test_scraper("X.com", env={})
        assert "TWITTER_COOKIES" in result["message"]

    def test_missing_cookies_no_network(self, st, monkeypatch):
        calls = _patch_http(monkeypatch, st, 200)
        st.test_scraper("X.com", env={})
        assert calls == []

    def test_cookies_missing_auth_token_returns_ok_false(self, st, monkeypatch):
        """Cookie string without auth_token should fail offline check."""
        _patch_http(monkeypatch, st, 200)
        result = st.test_scraper("X.com", env={"TWITTER_COOKIES": "ct0=abc123"})
        assert result["ok"] is False

    def test_cookies_missing_ct0_returns_ok_false(self, st, monkeypatch):
        """Cookie string without ct0 should fail offline check."""
        _patch_http(monkeypatch, st, 200)
        result = st.test_scraper("X.com", env={"TWITTER_COOKIES": "auth_token=xyz"})
        assert result["ok"] is False

    def test_well_formed_cookies_ok(self, st, monkeypatch):
        """Both auth_token and ct0 present => passes offline check."""
        _patch_http(monkeypatch, st, 200)
        result = st.test_scraper(
            "X.com",
            env={"TWITTER_COOKIES": "auth_token=abc; ct0=def"},
        )
        # Might be ok=True (live check passed) or ok=False only if live call fails
        # but the offline structural check must not block it
        # With a mocked 200, expect ok=True
        assert result["ok"] is True

    def test_well_formed_cookies_latency_is_int(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 200)
        result = st.test_scraper(
            "X.com",
            env={"TWITTER_COOKIES": "auth_token=abc; ct0=def"},
        )
        assert isinstance(result["latency_ms"], int)


# ===========================================================================
# 7. Discord — bot vs user mode Authorization header
# ===========================================================================

class TestDiscord:
    def test_missing_token_returns_ok_false(self, st):
        result = st.test_scraper("Discord-1", env={})
        assert result["ok"] is False

    def test_missing_token_message_mentions_discord_bot_token(self, st):
        result = st.test_scraper("Discord-1", env={})
        assert "DISCORD_BOT_TOKEN" in result["message"]

    def test_missing_token_no_network(self, st, monkeypatch):
        calls = _patch_http(monkeypatch, st, 200)
        st.test_scraper("Discord-1", env={})
        assert calls == []

    def test_bot_mode_sends_bot_prefix(self, st, monkeypatch):
        """DISCORD_AUTH_MODE=bot => Authorization: Bot <token>."""
        calls = _patch_http(monkeypatch, st, 200)
        st.test_scraper(
            "Discord-1",
            env={"DISCORD_BOT_TOKEN": "mytoken", "DISCORD_AUTH_MODE": "bot"},
        )
        assert len(calls) == 1
        auth = calls[0]["headers"].get("Authorization", "")
        assert auth.startswith("Bot "), f"Expected 'Bot ...' got {auth!r}"

    def test_user_mode_sends_raw_token(self, st, monkeypatch):
        """DISCORD_AUTH_MODE=user => Authorization: <token> (no prefix)."""
        calls = _patch_http(monkeypatch, st, 200)
        st.test_scraper(
            "Discord-1",
            env={"DISCORD_BOT_TOKEN": "mytoken", "DISCORD_AUTH_MODE": "user"},
        )
        assert len(calls) == 1
        auth = calls[0]["headers"].get("Authorization", "")
        assert auth == "mytoken", f"Expected raw token, got {auth!r}"

    def test_auto_mode_defaults_to_bot_prefix(self, st, monkeypatch):
        """DISCORD_AUTH_MODE=auto (default) => start with Bot prefix."""
        calls = _patch_http(monkeypatch, st, 200)
        st.test_scraper(
            "Discord-1",
            env={"DISCORD_BOT_TOKEN": "mytoken"},  # no AUTH_MODE => auto
        )
        assert len(calls) >= 1
        auth = calls[0]["headers"].get("Authorization", "")
        assert auth.startswith("Bot "), f"Expected 'Bot ...' got {auth!r}"

    def test_discord_endpoint_is_users_me(self, st, monkeypatch):
        """Must probe https://discord.com/api/v10/users/@me."""
        calls = _patch_http(monkeypatch, st, 200)
        st.test_scraper(
            "Discord-1",
            env={"DISCORD_BOT_TOKEN": "mytoken", "DISCORD_AUTH_MODE": "bot"},
        )
        assert any("users/@me" in c["url"] for c in calls)

    def test_200_returns_ok_true(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 200)
        result = st.test_scraper(
            "Discord-1",
            env={"DISCORD_BOT_TOKEN": "mytoken", "DISCORD_AUTH_MODE": "bot"},
        )
        assert result["ok"] is True

    def test_401_returns_ok_false(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 401)
        result = st.test_scraper(
            "Discord-1",
            env={"DISCORD_BOT_TOKEN": "badtoken", "DISCORD_AUTH_MODE": "bot"},
        )
        assert result["ok"] is False

    def test_401_message_mentions_auth(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 401)
        result = st.test_scraper(
            "Discord-1",
            env={"DISCORD_BOT_TOKEN": "badtoken", "DISCORD_AUTH_MODE": "bot"},
        )
        assert "auth" in result["message"].lower() or "401" in result["message"]


# ===========================================================================
# 8. Reddit
# ===========================================================================

class TestReddit:
    def test_no_credentials_returns_ok_true_unauthenticated(self, st, monkeypatch):
        """Reddit public search works without credentials.
        If no client_id/secret are set, we either skip the OAuth step or
        do a simple public ping. Either way test_scraper must not crash."""
        _patch_http(monkeypatch, st, 200)
        result = st.test_scraper("Reddit", env={})
        # Without credentials, the scraper should gracefully indicate status.
        # ok=True (public ping) or ok=False (credential missing) both allowed —
        # but the function must never raise.
        assert isinstance(result["ok"], bool)

    def test_with_client_id_and_secret_200_ok(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 200)
        result = st.test_scraper(
            "Reddit",
            env={
                "REDDIT_CLIENT_ID": "cid",
                "REDDIT_CLIENT_SECRET": "csecret",
                "REDDIT_USER_AGENT": "cull/test",
            },
        )
        assert result["ok"] is True

    def test_with_credentials_401_returns_ok_false(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 401)
        result = st.test_scraper(
            "Reddit",
            env={
                "REDDIT_CLIENT_ID": "cid",
                "REDDIT_CLIENT_SECRET": "csecret",
                "REDDIT_USER_AGENT": "cull/test",
            },
        )
        assert result["ok"] is False

    def test_reddit_never_raises(self, st, monkeypatch):
        import requests

        def boom(*a, **kw):
            raise requests.exceptions.ConnectionError("no network")

        monkeypatch.setattr(st, "_http_request", boom)
        # Must not propagate
        st.test_scraper(
            "Reddit",
            env={
                "REDDIT_CLIENT_ID": "cid",
                "REDDIT_CLIENT_SECRET": "csecret",
            },
        )


# ===========================================================================
# 9. Web scraper
# ===========================================================================

class TestWeb:
    def test_no_config_returns_result_without_raise(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 200)
        result = st.test_scraper("Web", env={})
        assert isinstance(result["ok"], bool)

    def test_with_url_config_200_ok(self, st, monkeypatch):
        _patch_http(monkeypatch, st, 200)
        result = st.test_scraper(
            "Web",
            config={"target_url": "https://example.com"},
            env={},
        )
        assert result["ok"] is True

    def test_with_url_config_timeout_ok_false(self, st, monkeypatch):
        import requests

        def timeout_http(method, url, *, headers=None, timeout=8.0):
            raise requests.exceptions.Timeout()

        monkeypatch.setattr(st, "_http_request", timeout_http)
        result = st.test_scraper(
            "Web",
            config={"target_url": "https://example.com"},
            env={},
        )
        assert result["ok"] is False


# ===========================================================================
# 10. Gallery-DL
# ===========================================================================

class TestGalleryDL:
    def test_package_importable_returns_ok_true(self, st, monkeypatch):
        """gallery-dl is installed in .venv so this should pass offline."""
        result = st.test_scraper("Gallery-DL", env={})
        # ok=True when gallery_dl can be imported and no missing disk paths
        assert result["ok"] is True

    def test_cookies_file_missing_returns_ok_false(self, st, monkeypatch, tmp_path):
        fake_path = str(tmp_path / "nonexistent_cookies.txt")
        result = st.test_scraper(
            "Gallery-DL",
            config={"cookies_file": fake_path},
            env={},
        )
        assert result["ok"] is False

    def test_cookies_file_missing_message_mentions_cookies(self, st, monkeypatch, tmp_path):
        fake_path = str(tmp_path / "nonexistent_cookies.txt")
        result = st.test_scraper(
            "Gallery-DL",
            config={"cookies_file": fake_path},
            env={},
        )
        assert "cookies" in result["message"].lower() or "file" in result["message"].lower()

    def test_cookies_file_existing_ok(self, st, monkeypatch, tmp_path):
        cookies_file = tmp_path / "cookies.txt"
        cookies_file.write_text("# Netscape HTTP Cookie File\n")
        result = st.test_scraper(
            "Gallery-DL",
            config={"cookies_file": str(cookies_file)},
            env={},
        )
        assert result["ok"] is True

    def test_config_path_missing_returns_ok_false(self, st, monkeypatch, tmp_path):
        fake_path = str(tmp_path / "nonexistent_config.json")
        result = st.test_scraper(
            "Gallery-DL",
            config={"config_path": fake_path},
            env={},
        )
        assert result["ok"] is False

    def test_config_path_existing_ok(self, st, monkeypatch, tmp_path):
        cfg_file = tmp_path / "gdl_config.json"
        cfg_file.write_text("{}")
        result = st.test_scraper(
            "Gallery-DL",
            config={"config_path": str(cfg_file)},
            env={},
        )
        assert result["ok"] is True

    def test_latency_ms_is_none_for_offline_check(self, st, monkeypatch):
        """Gallery-DL is an offline check; latency_ms should be None."""
        result = st.test_scraper("Gallery-DL", env={})
        assert result["latency_ms"] is None


# ===========================================================================
# 11. Local folder import
# ===========================================================================

class TestLocal:
    def test_no_dir_config_returns_ok_false(self, st):
        result = st.test_scraper("Local", config={}, env={})
        assert result["ok"] is False

    def test_nonexistent_dir_returns_ok_false(self, st, tmp_path):
        bad_dir = str(tmp_path / "nonexistent_folder")
        result = st.test_scraper("Local", config={"dir": bad_dir}, env={})
        assert result["ok"] is False

    def test_nonexistent_dir_message_mentions_dir(self, st, tmp_path):
        bad_dir = str(tmp_path / "nonexistent_folder")
        result = st.test_scraper("Local", config={"dir": bad_dir}, env={})
        assert "dir" in result["message"].lower() or "exist" in result["message"].lower()

    def test_nonexistent_dir_latency_ms_is_none(self, st, tmp_path):
        bad_dir = str(tmp_path / "nonexistent_folder")
        result = st.test_scraper("Local", config={"dir": bad_dir}, env={})
        assert result["latency_ms"] is None

    def test_existing_readable_dir_returns_ok_true(self, st, tmp_path):
        result = st.test_scraper("Local", config={"dir": str(tmp_path)}, env={})
        assert result["ok"] is True

    def test_existing_dir_latency_ms_is_none(self, st, tmp_path):
        """Local is a pure filesystem check; no HTTP => no latency_ms."""
        result = st.test_scraper("Local", config={"dir": str(tmp_path)}, env={})
        assert result["latency_ms"] is None

    def test_existing_dir_message_positive(self, st, tmp_path):
        result = st.test_scraper("Local", config={"dir": str(tmp_path)}, env={})
        msg = result["message"].lower()
        assert "ok" in msg or "readable" in msg or "accessible" in msg


# ===========================================================================
# 12. test_scraper never raises — robustness / fuzz
# ===========================================================================

class TestNeverRaises:
    @pytest.mark.parametrize("config,env", [
        (None, None),
        ({}, {}),
        ({"dir": 123}, {"CIVITAI_API_KEY": None}),
        ({"cookies_file": []}, {}),
        ({"unknown_key": object()}, {"DISCORD_BOT_TOKEN": ""}),
    ])
    def test_never_raises_for_odd_inputs(self, st, config, env):
        """test_scraper must not raise regardless of input types."""
        for name in st.SUPPORTED:
            try:
                result = st.test_scraper(name, config=config, env=env)
                assert isinstance(result, dict)
            except Exception as exc:
                pytest.fail(
                    f"test_scraper({name!r}, config={config!r}, env={env!r}) "
                    f"raised {type(exc).__name__}: {exc}"
                )

    def test_never_raises_for_unknown_name_with_weird_config(self, st):
        result = st.test_scraper("??", config={"x": [1, 2, 3]}, env={"KEY": "val"})
        assert result["ok"] is False

    def test_exception_in_http_request_never_propagates(self, st, monkeypatch):
        """Even an unexpected exception in _http_request must be swallowed."""
        def explode(*a, **kw):
            raise RuntimeError("unexpected internal error")

        monkeypatch.setattr(st, "_http_request", explode)
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "key"})
        assert result["ok"] is False
        assert isinstance(result["message"], str)


# ===========================================================================
# 13. Connection error (not timeout) path
# ===========================================================================

class TestConnectionError:
    def test_connection_error_returns_ok_false(self, st, monkeypatch):
        import requests

        def conn_err(method, url, *, headers=None, timeout=8.0):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(st, "_http_request", conn_err)
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "key"})
        assert result["ok"] is False

    def test_connection_error_message_mentions_connect(self, st, monkeypatch):
        import requests

        def conn_err(method, url, *, headers=None, timeout=8.0):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(st, "_http_request", conn_err)
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "key"})
        msg = result["message"].lower()
        assert "connect" in msg or "error" in msg


# ===========================================================================
# 14. Module-level import test (confirms SUPPORTED is importable directly)
# ===========================================================================

class TestModuleImport:
    def test_import_and_supported_is_accessible(self):
        mod = _import_scraper_test()
        assert hasattr(mod, "SUPPORTED")
        assert hasattr(mod, "test_scraper")
        assert isinstance(mod.SUPPORTED, tuple)


# ===========================================================================
# 15. Additional coverage: uncommon HTTP status, Reddit exceptions, Local edge
# ===========================================================================

class TestAdditionalCoverage:
    def test_unexpected_http_500_returns_ok_false(self, st, monkeypatch):
        """HTTP 500 is neither 2xx nor 401/403 — hits the 'unexpected HTTP' branch."""
        _patch_http(monkeypatch, st, 500)
        result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "key"})
        assert result["ok"] is False
        assert "500" in result["message"]

    def test_reddit_oauth_timeout_returns_ok_false(self, st, monkeypatch):
        """Reddit with credentials but _http_request raises Timeout."""
        import requests

        def timeout_http(method, url, *, headers=None, timeout=8.0):
            raise requests.exceptions.Timeout("reddit timed out")

        monkeypatch.setattr(st, "_http_request", timeout_http)
        result = st.test_scraper(
            "Reddit",
            env={
                "REDDIT_CLIENT_ID": "cid",
                "REDDIT_CLIENT_SECRET": "sec",
            },
        )
        assert result["ok"] is False
        assert "timed out" in result["message"] or "connect" in result["message"]

    def test_reddit_oauth_connection_error_returns_ok_false(self, st, monkeypatch):
        """Reddit with credentials but _http_request raises ConnectionError."""
        import requests

        def conn_err(method, url, *, headers=None, timeout=8.0):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(st, "_http_request", conn_err)
        result = st.test_scraper(
            "Reddit",
            env={
                "REDDIT_CLIENT_ID": "cid",
                "REDDIT_CLIENT_SECRET": "sec",
            },
        )
        assert result["ok"] is False

    def test_reddit_oauth_unexpected_exception_returns_ok_false(self, st, monkeypatch):
        """Reddit with credentials but _http_request raises an unexpected exception."""
        def weird_error(method, url, *, headers=None, timeout=8.0):
            raise ValueError("surprise")

        monkeypatch.setattr(st, "_http_request", weird_error)
        result = st.test_scraper(
            "Reddit",
            env={
                "REDDIT_CLIENT_ID": "cid",
                "REDDIT_CLIENT_SECRET": "sec",
            },
        )
        assert result["ok"] is False

    def test_local_path_is_file_not_dir_returns_ok_false(self, st, tmp_path):
        """When dir points to a file (not a directory), return ok=False."""
        a_file = tmp_path / "not_a_dir.txt"
        a_file.write_text("hello")
        result = st.test_scraper("Local", config={"dir": str(a_file)}, env={})
        assert result["ok"] is False
        msg = result["message"].lower()
        assert "dir" in msg or "directory" in msg

    def test_checker_exception_caught_by_dispatcher(self, st, monkeypatch):
        """If a checker raises unexpectedly, test_scraper catches it."""
        def bomb(config, env):
            raise RuntimeError("checker exploded")

        monkeypatch.setattr(st, "_check_civitai_com", bomb, raising=False)
        # We need to also patch the dispatcher dict if it pre-bound the reference
        original_checkers = dict(st._CHECKERS)
        st._CHECKERS["Civitai-Com"] = bomb
        try:
            result = st.test_scraper("Civitai-Com", env={"CIVITAI_API_KEY": "key"})
            assert result["ok"] is False
            assert "internal error" in result["message"].lower() or isinstance(result["message"], str)
        finally:
            st._CHECKERS.update(original_checkers)
