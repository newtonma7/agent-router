import pytest

from adaptive_router.config import Settings, SettingsError


def test_settings_have_safe_mock_defaults_without_credentials():
    settings = Settings.from_env({})

    assert settings.mock_mode is True
    assert settings.openai_api_key is None
    assert settings.max_tool_calls == 3


def test_settings_load_dotenv_when_using_process_environment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "ADAPTIVE_ROUTER_MOCK_MODE=true\n"
        "ADAPTIVE_ROUTER_DIRECT_INPUT_COST_PER_1K_TOKENS=0.15\n"
        "ADAPTIVE_ROUTER_DIRECT_OUTPUT_COST_PER_1K_TOKENS=0.60\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    settings = Settings.from_env()

    assert settings.mock_mode is True
    assert settings.direct_input_cost_per_1k_tokens == 0.15
    assert settings.direct_output_cost_per_1k_tokens == 0.60


def test_process_environment_overrides_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("ADAPTIVE_ROUTER_MAX_TOOL_CALLS=3\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ADAPTIVE_ROUTER_MAX_TOOL_CALLS", "7")

    assert Settings.from_env().max_tool_calls == 7


def test_settings_parse_typed_environment_values():
    settings = Settings.from_env(
        {
            "ADAPTIVE_ROUTER_MOCK_MODE": "false",
            "OPENAI_API_KEY": "secret",
            "ADAPTIVE_ROUTER_DIRECT_MODEL": "cheap",
            "ADAPTIVE_ROUTER_STRONG_MODEL": "capable",
            "ADAPTIVE_ROUTER_MAX_TOOL_CALLS": "5",
            "ADAPTIVE_ROUTER_COST_PENALTY": "0.2",
            "ADAPTIVE_ROUTER_DIRECT_INPUT_COST_PER_1K_TOKENS": "0.15",
            "ADAPTIVE_ROUTER_DIRECT_OUTPUT_COST_PER_1K_TOKENS": "0.60",
        }
    )

    assert settings.mock_mode is False
    assert settings.openai_api_key == "secret"
    assert settings.direct_model == "cheap"
    assert settings.strong_model == "capable"
    assert settings.max_tool_calls == 5
    assert settings.cost_penalty == 0.2
    assert settings.direct_input_cost_per_1k_tokens == 0.15
    assert settings.direct_output_cost_per_1k_tokens == 0.60


def test_live_mode_requires_a_provider_key():
    with pytest.raises(SettingsError, match="OPENAI_API_KEY"):
        Settings.from_env({"ADAPTIVE_ROUTER_MOCK_MODE": "false"})


def test_invalid_values_are_typed_configuration_errors():
    with pytest.raises(SettingsError, match="MAX_TOOL_CALLS"):
        Settings.from_env({"ADAPTIVE_ROUTER_MAX_TOOL_CALLS": "zero"})
