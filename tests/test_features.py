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


def test_seed_extraction_counts_requested_output_fields():
    from adaptive_router.models import load_seed_dataset

    tasks = {task.id: task for task in load_seed_dataset("data/seed_tasks.json").tasks}

    assert extract_features(tasks["S1"])[7] == 0.3
    assert extract_features(tasks["S2"])[7] == 0.4
    assert extract_features(tasks["S3"])[7] == 0.3


def test_feature_extraction_ignores_post_action_attributes():
    before = SimpleNamespace(category="reasoning", prompt="Who is first?")
    after = SimpleNamespace(**vars(before), answer="secret", quality=1.0, cost_usd=99, latency_seconds=99, tool_calls=4)
    assert extract_features(before) == extract_features(after)
