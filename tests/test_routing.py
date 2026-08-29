from adaptive_router.routing import CategoryPolicy, EpsilonGreedyPolicy, LinUCBPolicy, RandomPolicy, StaticPolicy, UCBPolicy
from adaptive_router.routing.reward import calculate_reward, hindsight_oracle


def test_seeded_random_policy_replays_same_actions():
    context = (1.0,) * 8
    left = RandomPolicy(seed=7)
    right = RandomPolicy(seed=7)
    assert [left.select(context) for _ in range(20)] == [right.select(context) for _ in range(20)]


def test_static_and_category_policies():
    assert StaticPolicy("strong").select((1, 0, 0, 0)) == "strong"
    policy = CategoryPolicy({"arithmetic": "tool", "reasoning": "strong"})
    assert policy.select((1, 0, 0, 0, 0, 0, 0, 0)) == "tool"
    assert policy.select((0, 1, 0, 0, 0, 0, 0, 0)) == "strong"


def test_epsilon_greedy_and_ucb_update_only_observed_action():
    epsilon = EpsilonGreedyPolicy(epsilon=0, seed=1)
    epsilon.update("strong", 1)
    assert epsilon.counts == {"direct": 0, "strong": 1, "tool": 0}
    assert epsilon.select() == "strong"
    ucb = UCBPolicy(exploration=0)
    assert [ucb.select() for _ in range(3)] == ["direct", "direct", "direct"]  # no update, no information
    ucb.update("direct", 0.5)
    assert ucb.select() == "strong"


def test_linucb_updates_selected_action():
    policy = LinUCBPolicy(alpha=0, n_features=2)
    assert policy.select((1, 0)) == "direct"
    policy.update("direct", 1.0)
    assert policy.counts == {"direct": 1, "strong": 0, "tool": 0}


def test_reward_keeps_components_and_oracle_prefers_passing_quality():
    reward = calculate_reward(0.9, 2, 3, reference_cost_usd=2, reference_latency_seconds=3, cost_penalty=.1, latency_penalty=.2)
    assert reward.quality == .9
    assert reward.normalized_cost == 1
    assert reward.normalized_latency == 1
    assert abs(reward.reward - .6) < 1e-9
    candidates = [{"passed": False, "quality": 1, "cost_usd": 0}, {"passed": True, "quality": .8, "cost_usd": 10}]
    assert hindsight_oracle(candidates) is candidates[1]
