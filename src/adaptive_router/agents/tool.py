"""Bounded calculator-tool strategy."""

from __future__ import annotations

import ast
import json
import math
import operator
from collections.abc import Callable, Mapping
from typing import Any

from adaptive_router.models.agent_result import AgentStrategy
from adaptive_router.output import task_output_contract
from adaptive_router.providers.base import CompletionResponse, ToolCall

from .base import MeasuredStrategy, Pricing, _prompt


class ToolExecutionError(RuntimeError):
    """A tool call was malformed, unavailable, or exceeded the configured bound."""

    def __init__(self, message: str, *, calls: int) -> None:
        super().__init__(message)
        self.tool_calls = calls


_BINARY_OPERATORS: dict[
    type[ast.operator], Callable[[int | float, int | float], int | float]
] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[int | float], int | float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculate(expression: str) -> int | float:
    """Evaluate an allowlisted arithmetic or statistical expression."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("calculator expression must be non-empty text")
    tree = ast.parse(expression, mode="eval")

    def visit_values(node: ast.AST) -> list[float]:
        if not isinstance(node, (ast.List, ast.Tuple)):
            raise ValueError("statistical functions require a numeric list")
        values = [float(visit(item)) for item in node.elts]
        if not values:
            raise ValueError("statistical functions require at least one value")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("calculator values must be finite")
        return values

    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("calculator exponent is too large")
            value = _BINARY_OPERATORS[type(node.op)](left, right)
            if not math.isfinite(float(value)):
                raise ValueError("calculator result is not finite")
            return value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in {"mean", "population_stddev"}:
                raise ValueError("calculator function is not allowlisted")
            if len(node.args) != 1 or node.keywords:
                raise ValueError("statistical functions accept one numeric list")
            values = visit_values(node.args[0])
            average = math.fsum(values) / len(values)
            if node.func.id == "mean":
                return average
            variance = math.fsum((value - average) ** 2 for value in values) / len(values)
            return math.sqrt(variance)
        raise ValueError("calculator accepts arithmetic and allowlisted statistics only")

    return visit(tree)


class ToolStrategy(MeasuredStrategy):
    """Let a provider request calculator calls, with a hard call limit."""

    strategy = AgentStrategy.TOOL

    def __init__(
        self,
        provider: Any,
        model: str | None = None,
        *,
        max_tool_calls: int | None = None,
        pricing: Pricing | None = None,
        settings: Any | None = None,
    ) -> None:
        super().__init__(provider, model, pricing, settings=settings)
        if max_tool_calls is None:
            max_tool_calls = getattr(settings, "max_tool_calls", 3)
        if max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative")
        self.max_tool_calls = max_tool_calls

    @property
    def tools(self) -> tuple[Mapping[str, Any], ...]:
        return (
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": (
                        "Evaluate one arithmetic expression exactly, or use mean([numbers]) "
                        "and population_stddev([numbers]) for statistics."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                        "additionalProperties": False,
                    },
                },
            },
        )

    def _complete(self, task: Any) -> tuple[CompletionResponse, int]:
        prompt = _prompt(task)
        calls = 0
        results: list[Mapping[str, Any]] = []
        response = self._provider_call(task, calls, results)
        input_tokens = response.input_tokens
        output_tokens = response.output_tokens
        while response.tool_calls:
            if calls + len(response.tool_calls) > self.max_tool_calls:
                raise ToolExecutionError("maximum tool call count exceeded", calls=calls)
            for call in response.tool_calls:
                calls += 1
                result = self._run_tool(call, calls)
                results.append(
                    {
                        "tool_call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                        "content": str(result),
                    }
                )
            response = self._provider_call(task, calls, results)
            input_tokens = self._sum_tokens(input_tokens, response.input_tokens)
            output_tokens = self._sum_tokens(output_tokens, response.output_tokens)
        if response.text is None:
            raise ToolExecutionError("provider returned no final answer", calls=calls)
        return CompletionResponse(
            text=response.text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=response.finish_reason,
        ), calls

    @staticmethod
    def _sum_tokens(current: int | None, new: int | None) -> int | None:
        if current is None or new is None:
            return None
        return current + new

    def _provider_call(
        self,
        task: Any,
        calls: int,
        results: list[Mapping[str, Any]],
    ) -> CompletionResponse:
        prompt = _prompt(task)
        system_prompt, response_format = task_output_contract(task)
        category = getattr(getattr(task, "category", None), "value", getattr(task, "category", None))
        tools = self.tools if str(category).lower() == "arithmetic" else ()
        if tools:
            system_prompt += (
                " Use the calculator only for arithmetic or supported statistics; "
                "use mean([numbers]) or population_stddev([numbers]) for lists, "
                "and do not call it for other task types."
            )
        try:
            if results:
                return self.provider.complete(
                    prompt,
                    model=self.model,
                    tools=tools,
                    tool_results=results,
                    system_prompt=system_prompt,
                    response_format=response_format,
                    parallel_tool_calls=False if tools else None,
                )
            return self.provider.complete(
                prompt,
                model=self.model,
                tools=tools,
                system_prompt=system_prompt,
                response_format=response_format,
                parallel_tool_calls=False if tools else None,
            )
        except Exception as exc:
            raise ToolExecutionError(f"provider call failed: {exc}", calls=calls) from exc

    @staticmethod
    def _run_tool(call: ToolCall, calls: int) -> int | float:
        if not isinstance(call, ToolCall) or call.name != "calculator":
            raise ToolExecutionError(f"unavailable tool: {getattr(call, 'name', None)!r}", calls=calls)
        arguments = call.arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ToolExecutionError("malformed calculator arguments", calls=calls) from exc
        if not isinstance(arguments, Mapping) or set(arguments) != {"expression"}:
            raise ToolExecutionError("calculator arguments must contain only expression", calls=calls)
        try:
            return calculate(arguments["expression"])
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise ToolExecutionError(f"calculator failed: {exc}", calls=calls) from exc


ToolAgent = ToolStrategy

__all__ = ["ToolAgent", "ToolExecutionError", "ToolStrategy", "calculate"]
