from types import SimpleNamespace

from adaptive_router.features import extract_features


def test_features_are_pre_action_and_fixed_order():
    task = SimpleNamespace(category="arithmetic", prompt="Calculate 37 * 19 and return `answer`.")
    features = extract_features(task)
    assert features[:4] == (1.0, 0.0, 0.0, 0.0)
    assert features[4] > 0
    assert features[5] == 1.0
    assert features[6] > 0
    assert len(features) == 8


def test_feature_extraction_ignores_post_action_attributes():
    before = SimpleNamespace(category="reasoning", prompt="Who is first?")
    after = SimpleNamespace(**vars(before), answer="secret", quality=1.0, cost_usd=99, latency_seconds=99, tool_calls=4)
    assert extract_features(before) == extract_features(after)
