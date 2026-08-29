import pytest

from adaptive_router.config import Settings, SettingsError


def test_settings_have_safe_mock_defaults_without_credentials():
    settings = Settings.from_env({})

    assert settings.mock_mode is True
    assert settings.openai_api_key is None
    assert settings.max_tool_calls == 3


def test_settings_parse_typed_environment_values():
    settings = Settings.from_env(
        {
            "ADAPTIVE_ROUTER_MOCK_MODE": "false",
            "OPENAI_API_KEY": "secret",
            "ADAPTIVE_ROUTER_DIRECT_MODEL": "cheap",
            "ADAPTIVE_ROUTER_STRONG_MODEL": "capable",
            "ADAPTIVE_ROUTER_MAX_TOOL_CALLS": "5",
            "ADAPTIVE_ROUTER_COST_PENALTY": "0.2",
        }
    )

    assert settings.mock_mode is False
    assert settings.openai_api_key == "secret"
    assert settings.direct_model == "cheap"
    assert settings.strong_model == "capable"
    assert settings.max_tool_calls == 5
    assert settings.cost_penalty == 0.2


def test_live_mode_requires_a_provider_key():
    with pytest.raises(SettingsError, match="OPENAI_API_KEY"):
        Settings.from_env({"ADAPTIVE_ROUTER_MOCK_MODE": "false"})


def test_invalid_values_are_typed_configuration_errors():
    with pytest.raises(SettingsError, match="MAX_TOOL_CALLS"):
        Settings.from_env({"ADAPTIVE_ROUTER_MAX_TOOL_CALLS": "zero"})
