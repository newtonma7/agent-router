# Domain glossary

## Task
A bounded input to which a strategy returns an answer and an evaluator may assign quality.

## Strategy
A selectable way to execute a task: `direct`, `strong`, or `tool`.

## Context
Information available before strategy selection. It may describe a task, but never includes execution results or other post-action information.

## Router
A policy that selects one strategy from a task's context.

## Evaluation
The act of converting a strategy result into quality, pass/fail, and grading evidence.

## Experiment
A reproducible sequence of routed task executions used to compare policies and strategies.

## Run record
The durable record of one task execution, including context, selected strategy, result, evaluation, measurements, reward, and run metadata.

## Reward
The scalar feedback used by a learning policy, derived from quality with configured cost and latency penalties.
