"""Tests for configuration loading and env-placeholder expansion."""

import os
import textwrap

import pytest

from src.utils.config import Config, _expand_env_placeholders


class TestExpandEnvPlaceholders:
    def test_uses_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("CFG_TEST_HOST", raising=False)
        assert _expand_env_placeholders("${CFG_TEST_HOST:-0.0.0.0}") == "0.0.0.0"

    def test_uses_env_value_when_set(self, monkeypatch):
        monkeypatch.setenv("CFG_TEST_HOST", "127.0.0.1")
        assert _expand_env_placeholders("${CFG_TEST_HOST:-0.0.0.0}") == "127.0.0.1"

    def test_bare_placeholder_empty_when_unset(self, monkeypatch):
        monkeypatch.delenv("CFG_TEST_BARE", raising=False)
        assert _expand_env_placeholders("${CFG_TEST_BARE}") == ""

    def test_recurses_into_dicts_and_lists(self, monkeypatch):
        monkeypatch.setenv("CFG_TEST_PORT", "8001")
        tree = {"api": {"port": "${CFG_TEST_PORT:-8000}", "tags": ["${CFG_TEST_PORT}"]}}
        result = _expand_env_placeholders(tree)
        assert result == {"api": {"port": "8001", "tags": ["8001"]}}

    def test_non_string_scalars_passthrough(self):
        assert _expand_env_placeholders(42) == 42
        assert _expand_env_placeholders(True) is True


class TestConfigGet:
    def test_get_returns_expanded_value(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CFG_TEST_URL", raising=False)
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            textwrap.dedent(
                """
                data:
                  supabase:
                    url: ${CFG_TEST_URL:-https://example.test}
                api:
                  host: ${CFG_TEST_API_HOST:-0.0.0.0}
                """
            )
        )
        config = Config(config_path=str(cfg_file))
        # No literal "${...}" leaks through to Config.get anymore.
        assert config.get("data.supabase.url") == "https://example.test"
        assert config.get("api.host") == "0.0.0.0"

    def test_get_reflects_environment_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFG_TEST_URL", "https://real.supabase.co")
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("data:\n  supabase:\n    url: ${CFG_TEST_URL:-https://example.test}\n")
        config = Config(config_path=str(cfg_file))
        assert config.get("data.supabase.url") == "https://real.supabase.co"

    def test_get_returns_default_for_missing_key(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("api:\n  host: localhost\n")
        config = Config(config_path=str(cfg_file))
        assert config.get("does.not.exist", "fallback") == "fallback"
