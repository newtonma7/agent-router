"""Routing policies for the adaptive agent router."""

from .base import ACTIONS, Policy, RoutingPolicy
from .epsilon_greedy import EpsilonGreedy, EpsilonGreedyPolicy, EpsilonGreedyRouter
from .linucb import LinUCB, LinUCBPolicy, LinUCBRouter
from .random import RandomPolicy, RandomRouter, RandomRoutingPolicy
from .static import (
    DEFAULT_CATEGORY_MAPPING,
    AlwaysDirect,
    AlwaysDirectPolicy,
    AlwaysStrong,
    AlwaysStrongPolicy,
    AlwaysTool,
    AlwaysToolPolicy,
    CategoryPolicy,
    CategoryRoutingPolicy,
    StaticRoutingPolicy,
    StaticPolicy,
)
from .ucb import UCB, UCBPolicy, UCBRouter
from .reward import Reward, RewardConfig, calculate_reward, hindsight_oracle, normalize_cost, normalize_latency, regret

__all__ = [
    "ACTIONS", "Policy", "RoutingPolicy", "RandomPolicy", "RandomRouter", "RandomRoutingPolicy",
    "StaticPolicy", "CategoryPolicy", "StaticRoutingPolicy", "CategoryRoutingPolicy", "DEFAULT_CATEGORY_MAPPING", "AlwaysDirect", "AlwaysStrong", "AlwaysTool",
    "AlwaysDirectPolicy", "AlwaysStrongPolicy", "AlwaysToolPolicy",
    "EpsilonGreedy", "EpsilonGreedyPolicy", "EpsilonGreedyRouter", "UCB", "UCBPolicy", "UCBRouter", "LinUCB", "LinUCBPolicy", "LinUCBRouter",
    "Reward", "RewardConfig", "calculate_reward", "normalize_cost", "normalize_latency", "hindsight_oracle", "regret",
]
