from enum import Enum


class AgentStrategy(str, Enum):
    DIRECT = "direct"
    STRONG = "strong"
    TOOL = "tool"


class TaskCategory(str, Enum):
    ARITHMETIC = "arithmetic"
    REASONING = "reasoning"
    EXPLANATION = "explanation"
    EXTRACTION = "extraction"


class EvaluationType(str, Enum):
    NUMERIC = "numeric"
    EXACT = "exact"
    STRUCTURED = "structured"
    RUBRIC = "rubric"
