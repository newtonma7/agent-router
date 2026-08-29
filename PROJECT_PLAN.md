# Adaptive Agent Router

> [!summary] Project Goal
> Build a contextual-bandit-based AI router that learns which agent strategy to use for each incoming task while maintaining an acceptable quality level and minimizing unnecessary inference cost and latency.

---

# 1. Core Research Question

Can a learned contextual router outperform static routing policies by selecting the cheapest/fastest agent strategy that is likely to meet the required quality level for a task?

The router should learn relationships like:

```text
task context
    ↓
routing policy
    ↓
direct / strong / tool
    ↓
execution
    ↓
evaluation
    ↓
quality + cost + latency
    ↓
learning signal
```

---

# 2. Project Constraints

- [x] Target completion: **4–5 weeks**
- [x] Learning is the primary objective
- [x] Resume strength for AI Engineer / FDE roles is secondary
- [x] Do not fine-tune an LLM
- [x] Do not perform expensive large-model RL rollouts
- [x] Keep initial agent count to three
- [x] Keep the initial task benchmark small
- [x] Prioritize experiments over frontend work
- [x] Use deterministic evaluation wherever possible
- [x] Use LLM judges only where semantic evaluation is necessary
- [x] Keep routing and evaluation decoupled

## Current implementation status

The first implementation pass is complete on `main` (commit `03c2342`). The repository now contains:

- typed task, result, rubric, and evaluation models;
- the versioned 12-task seed dataset;
- numeric, exact, structured, and rubric evaluators;
- `direct`, `strong`, and bounded `tool` strategies with mock and OpenAI-compatible providers;
- random, static/category, epsilon-greedy, UCB, and LinUCB policies;
- pre-action feature extraction, seeded replay, reward/regret helpers, JSONL persistence, and reports;
- a shared FastAPI service with `/health` and `/infer`, Docker packaging, and a Docker smoke script;
- 52 automated tests and passing mypy checks.

The project is implementation-complete but not research-conclusion-complete. Remaining work is to run the full live/baseline comparisons, repeated stochastic trials, human judge calibration, ablations, distribution-shift experiments, visualizations, and final analysis. Docker smoke execution is still pending because Docker is unavailable in the development environment. The A2 and A3 expected values in this plan should also be verified against their prompts before treating benchmark results as evidence. The locked v1 feature vector uses category, normalized prompt length, number presence/count, and requested-field count; `contains code` remains a future feature rather than an implemented input.

---

# 3. What I Want to Learn

## Reinforcement Learning

- [ ] Multi-armed bandits
- [ ] Contextual bandits
- [ ] Exploration vs exploitation
- [ ] Reward design
- [ ] Regret
- [ ] Epsilon-greedy
- [ ] Upper Confidence Bound
- [ ] LinUCB
- [ ] Why routing can be modeled as a contextual bandit
- [ ] Difference between bandits and full RL
- [ ] Constrained optimization vs weighted reward
- [ ] Distribution shift and policy adaptation

## LLM Engineering

- [ ] Model routing
- [ ] Tool calling
- [ ] Structured outputs
- [ ] Token accounting
- [ ] Inference cost measurement
- [ ] Latency measurement
- [ ] Model quality/cost tradeoffs
- [ ] LLM-as-judge evaluation
- [ ] Evaluation calibration
- [ ] Handling stochastic model output

## Agent Engineering

- [ ] Common agent interfaces
- [ ] Tool-enabled execution loops
- [ ] Routing policies
- [ ] Agent traces
- [ ] Failure handling
- [ ] Strategy selection
- [ ] Agent evaluation

## AI Engineer / FDE Skills

- [ ] FastAPI inference service
- [ ] Experiment design
- [ ] Evaluation infrastructure
- [ ] Observability
- [ ] Structured logging
- [ ] Cost/latency optimization
- [ ] Reproducible benchmarks
- [ ] Docker
- [ ] Explaining AI system tradeoffs

---

# 4. System Formulation

## Context

Information available **before** the router chooses an action.

Initial features:

- task category
- prompt length
- contains numbers
- contains code
- number of numeric inputs
- number of requested output fields / inputs where applicable

Important:

The router cannot use information generated after agent execution.

---

## Actions

Three initial strategies:

### `direct`

Cheap model.

```text
task
 ↓
cheap model
 ↓
answer
```

Characteristics:

- low cost
- low latency
- sufficient for simple tasks

---

### `strong`

Higher-capability model or higher reasoning configuration.

```text
task
 ↓
strong model
 ↓
answer
```

Characteristics:

- higher expected reasoning quality
- more expensive
- slower

---

### `tool`

Cheap model with deterministic tool access.

Initial tool:

- calculator and/or restricted Python computation

```text
task
 ↓
cheap model
 ↓
tool call
 ↓
tool output
 ↓
final answer
```

Characteristics:

- inexpensive model
- deterministic computation
- useful for calculation-heavy tasks

---

# 5. Product Objective

Initial product policy:

> Maintain acceptable answer quality and avoid paying for additional model capability unless the cheaper strategy is expected to fall below the required quality level.

Long-term formulation:

```text
minimize:
cost + latency

subject to:
task quality requirement being satisfied
```

Important:

There will **not** initially be one universal quality threshold across all task types.

Each evaluation type defines what constitutes passing.

---

# 6. Initial Dataset

Create a frozen **12-task seed development set**.

Four categories:

- Arithmetic / computation
- Logic / reasoning
- Factual / explanatory
- Structured extraction

Three tasks per category.

---

# 7. Seed Tasks

## Arithmetic

### A1 — Simple multiplication

**Prompt**

What is 37 × 19? Return only the number.

**Expected**

`703`

**Evaluation**

Numeric exact match.

**Hypothesis**

Tool should provide strong quality cheaply, although direct may also succeed.

---

### A2 — Multi-step percentage

**Prompt**

A laptop originally costs $1,240. It is discounted by 18%, and then a 6% sales tax is applied to the discounted price. What is the final price? Round to the nearest cent and return only the amount.

**Expected**

`1077.81`

**Evaluation**

Numeric comparison with rounding tolerance.

**Hypothesis**

Tool should reduce arithmetic error.

---

### A3 — Statistics

**Prompt**

Given the values 14, 22, 9, 31, 18, 27, 11, 25, 19, and 34, calculate their mean and population standard deviation. Round both to two decimal places.

**Expected**

```text
mean = 21.00
population standard deviation ≈ 7.92
```

**Evaluation**

Numeric comparison with tolerance.

**Hypothesis**

Tool should have the strongest quality/cost profile.

---

# Logic

## R1 — Basic deduction

**Prompt**

All lorps are mavens. No mavens are tals. If Zed is a lorp, can Zed be a tal? Return only yes or no.

**Expected**

No.

**Evaluation**

Exact conclusion.

**Hypothesis**

Direct should be sufficient.

---

## R2 — Constraint reasoning

**Prompt**

Four people—Ava, Ben, Chloe, and Diego—stand in a line. Ava is somewhere before Ben. Chloe is immediately after Ben. Diego is before Ava. Who must be first?

**Expected**

Diego.

**Evaluation**

Exact answer.

**Hypothesis**

Strong may provide increased reliability.

This is an empirical hypothesis, not part of the grading criteria.

---

## R3 — Multi-constraint scheduling

**Prompt**

Five presentations—A, B, C, D, and E—are scheduled one per time slot from 1 through 5. A must occur before C. B must occur immediately after D. E cannot be first or last. C must occur after E. If A is scheduled third, which presentation must be first? Return only the answer.

**Expected**

D.

**Evaluation**

Exact answer.

**Hypothesis**

Strong should have an advantage on multi-step constraint reasoning.

---

# Explanation

## E1 — Hash table

**Prompt**

In two sentences or fewer, explain what a hash table is and why it is useful.

**Reference concepts**

- key-value association
- hashing
- efficient lookup/insertion

**Evaluation**

LLM rubric.

**Hypothesis**

Direct should provide sufficient quality.

---

## E2 — Interest rates and inflation

**Prompt**

Explain why increasing interest rates can reduce inflation. Your explanation should connect interest rates, borrowing, spending or investment, aggregate demand, and price pressure.

**Reference concepts**

- borrowing becomes more expensive
- consumption/investment tends to decline
- aggregate demand falls
- price pressure decreases

**Evaluation**

LLM rubric.

**Hypothesis**

Strong may provide better reasoning and completeness.

---

## E3 — Redis caching tradeoffs

**Prompt**

An API currently returns database results directly from PostgreSQL on every request. The engineering team is considering adding a Redis cache. Explain two situations where caching would substantially improve the system and two situations where caching could introduce problems. Briefly explain why in each case.

**Expected concepts**

Possible benefits:

- repeated queries
- read-heavy workloads
- expensive database operations
- relatively stable data
- reduced database load
- lower latency

Possible risks:

- stale data
- cache invalidation complexity
- rapidly changing data
- poor hit rate
- consistency issues
- infrastructure complexity

**Evaluation**

LLM rubric.

**Hypothesis**

Strong may provide more complete technical reasoning.

---

# Structured Extraction

## S1 — Simple extraction

Extract:

```text
name
plan
seats
```

Expected:

```json
{
  "name": "Maya Chen",
  "plan": "Enterprise",
  "seats": 48
}
```

Evaluation:

Structured field comparison.

---

## S2 — Extraction with distractors

Expected:

```json
{
  "order_id": "A83921",
  "email": "samira.khan@example.com",
  "action": "replacement",
  "item": "USB-C docking station"
}
```

Evaluation:

Structured field comparison.

---

## S3 — Multi-record extraction

Extract service status records into JSON.

Evaluation:

Structured record comparison.

---

# 8. Locked Task Schema

Use Pydantic rather than raw dictionaries so invalid benchmark tasks fail early.

Initial conceptual model:

```python
class Task(BaseModel):
    id: str
    prompt: str
    category: TaskCategory
    evaluation_type: EvaluationType

    expected_answer: Any | None = None
    rubric: EvaluationRubric | None = None

    metadata: dict[str, Any] = {}
```

---

## Task Categories

```python
class TaskCategory(str, Enum):
    ARITHMETIC = "arithmetic"
    REASONING = "reasoning"
    EXPLANATION = "explanation"
    EXTRACTION = "extraction"
```

---

## Evaluation Types

```python
class EvaluationType(str, Enum):
    NUMERIC = "numeric"
    EXACT = "exact"
    STRUCTURED = "structured"
    RUBRIC = "rubric"
```

---

# 9. Evaluation Philosophy

Use the simplest reliable evaluator for each task.

```text
Arithmetic
    ↓
NumericEvaluator

Logic
    ↓
ExactEvaluator

Extraction
    ↓
StructuredEvaluator

Explanation
    ↓
RubricLLMEvaluator
```

Every evaluator must produce the same `EvaluationResult`.

---

# 10. Locked EvaluationResult Schema

Conceptually:

```python
class EvaluationResult(BaseModel):
    quality: float
    passed: bool

    grader_type: str

    component_scores: dict[str, float] = {}
    feedback: str | None = None
```

Constraints:

```text
0 <= quality <= 1
```

The router should only depend on standardized outputs such as:

```text
quality
passed
```

It should not care how the evaluator produced them.

---

# 11. Numeric Evaluation

Used for:

- A1
- A2
- A3

Quality:

```text
correct → 1
incorrect → 0
```

Use numeric tolerance where appropriate.

Passing generally requires:

```text
quality = 1
```

---

# 12. Exact Evaluation

Used initially for logic tasks.

```text
correct conclusion → 1
incorrect conclusion → 0
```

Avoid grading reasoning style initially.

---

# 13. Structured Evaluation

Used for extraction tasks.

Use partial credit.

For `n` requested fields:

```text
quality =
correct_fields / total_fields
```

Example:

```text
4/4 = 1.00
3/4 = 0.75
2/4 = 0.50
```

Passing requirement for v1:

```text
all required fields correct
```

Therefore:

```text
passed = quality == 1.0
```

---

# 14. Rubric Evaluation

Used for explanatory tasks.

Each task may have its own task-specific rubric.

Common rubric dimensions:

| Dimension | Weight |
|---|---:|
| Technical correctness | 40% |
| Completeness | 25% |
| Reasoning / depth | 15% |
| Clarity | 10% |
| Relevance / concision | 10% |

Each dimension should use an anchored score such as:

```text
0
1
2
3
4
```

Example:

### Technical correctness

```text
4 = entirely technically correct
3 = correct overall with minor imprecision
2 = partially correct with meaningful issue
1 = largely incorrect
0 = fundamentally incorrect
```

Normalize component scores into `[0,1]`.

Then calculate weighted quality.

Important:

- Do not reward verbosity
- Do not reward fancy writing by itself
- Depth means useful reasoning
- Correctness dominates writing quality
- Judge should not know model identity
- Judge should not know inference cost
- Judge should not know latency
- Judge should not know routing strategy

---

# 15. Human Calibration

The LLM evaluator is not assumed to be ground truth.

For explanation tasks:

- [ ] Collect representative responses
- [ ] Manually score a subset
- [ ] Compare human judgments against LLM scores
- [ ] Inspect disagreements
- [ ] Refine rubrics
- [ ] Determine reasonable pass thresholds empirically

Do not manually grade every experiment.

Human evaluation is for **calibration and auditing**.

---

# 16. Quality Thresholds

Do not use one arbitrary global threshold such as:

```text
quality >= 0.90
```

for every task.

Instead:

### Arithmetic

```text
passed = correct
```

### Logic

```text
passed = correct
```

### Extraction

```text
passed = all required fields correct
```

### Explanation

```text
passed = rubric_score >= calibrated threshold
```

Determine the explanation threshold after human calibration.

---

# 17. Agent Result Schema

All agent strategies should eventually return a common object:

```python
class AgentResult(BaseModel):
    task_id: str
    strategy: AgentStrategy

    answer: Any

    latency_seconds: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float

    tool_calls: int = 0
    error: str | None = None
```

This allows every strategy to be evaluated identically.

---

# 18. Reward

Initially keep:

```text
quality
cost
latency
```

stored separately.

Never save only the combined reward.

Raw measurements:

```text
quality
cost_usd
latency_seconds
```

Normalized measurements:

```text
quality_normalized
cost_normalized
latency_normalized
```

Initial experimental reward:

```text
reward =
quality
- λ(cost_normalized)
- β(latency_normalized)
```

Quality should dominate initially.

---

# 19. Normalization

Quality is already:

```text
[0,1]
```

Cost and latency should use fixed reference values rather than dataset-dependent min/max normalization.

Conceptually:

```text
normalized_cost =
actual_cost / reference_cost

normalized_latency =
actual_latency / reference_latency
```

Reference values will be selected after collecting baseline measurements.

---

# 20. Longer-Term Objective

The more production-like version of the router is:

```text
Choose the cheapest / fastest strategy
that is sufficiently likely to pass
the task's quality requirement.
```

Conceptually:

```text
P(pass | context, action)
```

This becomes a later experiment after the basic bandit implementation works.

---

# 21. Repository Structure

Initial structure:

```text
adaptive-agent-router/
├── src/
│   └── adaptive_router/
│       ├── models/
│       │   ├── task.py
│       │   ├── agent_result.py
│       │   └── evaluation_result.py
│       │
│       ├── agents/
│       │   ├── base.py
│       │   ├── direct.py
│       │   ├── strong.py
│       │   └── tool.py
│       │
│       ├── evaluation/
│       │   ├── base.py
│       │   ├── numeric.py
│       │   ├── exact.py
│       │   ├── structured.py
│       │   └── rubric.py
│       │
│       ├── routing/
│       │   ├── base.py
│       │   ├── random.py
│       │   ├── static.py
│       │   ├── epsilon_greedy.py
│       │   ├── ucb.py
│       │   └── linucb.py
│       │
│       ├── features/
│       │   └── extractor.py
│       │
│       └── experiments/
│
├── data/
│   └── seed_tasks.json
│
├── tests/
│
├── scripts/
│
├── README.md
└── pyproject.toml
```

Do not create every file immediately.

Add structure only as needed.

---

# 22. Implementation Order

## Phase 1 — Data Models

- [x] Create project
- [x] Configure Python environment
- [x] Install Pydantic
- [x] Implement `TaskCategory`
- [x] Implement `EvaluationType`
- [x] Implement `Task`
- [x] Implement `EvaluationResult`
- [x] Write schema validation tests

### Definition of Done

Can create valid A1 and E3 `Task` objects.

Invalid combinations should eventually be rejected.

Examples:

```text
numeric task without expected_answer → invalid
rubric task without rubric → invalid
```

---

# 23. Phase 2 — Seed Dataset

- [x] Encode all 12 tasks
- [x] Load tasks from JSON/YAML
- [x] Validate every task through Pydantic
- [x] Freeze seed dataset
- [x] Store predicted strategy winners separately from router-visible data

Important:

Predicted winners are experimental hypotheses.

The router must never receive them.

---

# 24. Phase 3 — Evaluators

## NumericEvaluator

- [x] exact values
- [x] tolerance
- [x] numeric parsing
- [x] tests

## ExactEvaluator

- [x] answer normalization
- [x] case normalization
- [x] tests

## StructuredEvaluator

- [x] JSON parsing
- [x] field comparison
- [x] partial-credit scoring
- [x] tests

## RubricEvaluator

- [x] structured judge response
- [x] task-specific rubric
- [x] component scoring
- [x] weighted score
- [x] blind model identity
- [x] judge failure handling

---

# 25. Phase 4 — Human Calibration

- [ ] Generate several explanation responses
- [ ] Manually grade them
- [ ] Compare against judge
- [ ] Adjust rubric wording
- [ ] Select explanation pass threshold
- [ ] Document evaluator limitations

---

# 26. Phase 5 — Agent Interface

- [x] Define `AgentStrategy`
- [x] Define `AgentResult`
- [x] Define common `Agent` interface
- [x] Instrument latency
- [x] Instrument tokens
- [x] Instrument cost
- [x] Instrument tool calls

---

# 27. Phase 6 — Agents

## Direct

- [x] cheap model call
- [x] structured result
- [x] error handling

## Strong

- [x] stronger model/configuration
- [x] same interface
- [x] cost tracking

## Tool

- [x] cheap model
- [x] calculator/Python tool
- [x] maximum tool-call count
- [x] result instrumentation

---

# 28. Phase 7 — First Baseline Experiment

Before building any learned router:

Run:

```text
12 tasks
×
3 strategies
×
multiple trials where appropriate
```

Collect:

- quality
- pass/fail
- latency
- token usage
- cost
- errors
- tool usage

Goal:

> Determine whether meaningful routing opportunities actually exist.

Questions:

- [ ] Does tool win computation tasks?
- [ ] Is direct sufficient for easy tasks?
- [ ] Does strong actually improve difficult explanation/reasoning tasks?
- [ ] Are there tasks where the predicted winner was wrong?
- [ ] Are the performance differences large enough to justify routing?

Do not modify the dataset merely to make hypotheses correct.

---

# 29. Phase 8 — Experiment Storage

Store each run:

```text
experiment_id
trial_id
task_id
task_category
strategy
quality
passed
component_scores
cost_usd
latency
input_tokens
output_tokens
tool_calls
errors
timestamp
```

- [ ] SQLite
- [x] experiment configuration
- [x] random seeds
- [ ] CSV export

---

# 30. Phase 9 — Baseline Routers

Implement:

- [x] Random
- [x] Static rules

Possible static rules:

```text
arithmetic → tool
simple extraction → direct
complex explanation → strong
```

These become baselines for learned policies.

---

# 31. Phase 10 — Epsilon-Greedy

Before implementation understand:

- [ ] action values
- [ ] expected reward
- [ ] epsilon
- [ ] exploration
- [ ] exploitation
- [ ] incremental mean update

Then:

- [x] implement
- [x] test
- [ ] run experiments

---

# 32. Phase 11 — UCB

Understand:

```text
estimated reward
+
uncertainty / exploration bonus
```

- [x] implement UCB
- [x] test
- [ ] compare to epsilon-greedy
- [ ] measure convergence

---

# 33. Phase 12 — Feature Extraction

Initial explicit context:

```text
category
prompt length
contains numbers
contains code
number of numeric inputs
number of requested fields
```

- [x] Implement deterministic feature extractor
- [x] Prevent post-execution leakage
- [x] Encode categorical values
- [x] Normalize numerical features where required

---

# 34. Phase 13 — LinUCB

Understand:

- [ ] contextual bandits
- [ ] context vector
- [ ] action-specific reward estimates
- [ ] linear reward assumption
- [ ] confidence bounds

Implement:

- [x] state matrices
- [x] action selection
- [x] updates
- [x] numerical tests

Primary hypothesis:

> Contextual routing should outperform algorithms that only learn global average strategy performance.

---

# 35. Phase 14 — Main Experiment

Compare:

```text
Random
Static
Epsilon-Greedy
UCB
LinUCB
```

Metrics:

- [ ] task pass rate
- [ ] average quality
- [ ] average reward
- [ ] inference cost/task
- [ ] latency/task
- [ ] strategy usage
- [ ] cumulative reward
- [ ] regret

---

# 36. Phase 15 — Ablations

## Category Ablation

Compare:

```text
LinUCB + explicit category
```

against:

```text
LinUCB without category
```

Question:

> How much does routing rely on explicit task metadata?

---

## Reward Sensitivity

Compare:

- quality-first
- cost-sensitive
- latency-sensitive

Question:

> How does routing behavior change when product priorities change?

---

## Distribution Shift

Artificially degrade one strategy later in the experiment.

Question:

> Can the router adapt when historical performance becomes stale?

---

# 37. Stretch Experiment — Inferred Task Category

> [!important]
> Keep this as a later thought piece / extension. Do not block the core project on it.

Instead of providing:

```text
category = arithmetic
```

directly to the router:

```text
raw prompt
   ↓
cheap pre-routing classifier
   ↓
computation_score
reasoning_score
extraction_score
ambiguity_score
   ↓
contextual router
```

Compare:

```text
explicit categories
vs
no category
vs
inferred category
```

Research question:

> How much routing performance can be retained when manually supplied task metadata is replaced by automatically inferred semantic features?

---

# 38. Stretch Experiment — Quality-Constrained Routing

Compare scalar reward routing:

```text
maximize:
quality - cost - latency
```

against:

```text
meet task quality requirement
then minimize cost / latency
```

Potential learned target:

```text
P(pass | task context, strategy)
```

This more closely matches the desired product policy.

---

# 39. Production API

Only after experiments work.

FastAPI endpoints:

```text
POST /route
GET /health
```

Pipeline:

```text
request
 ↓
Task
 ↓
FeatureExtractor
 ↓
Router
 ↓
Agent
 ↓
Evaluator
 ↓
metrics
 ↓
router update
 ↓
response
```

---

# 40. Observability

Track:

- [ ] routing decision
- [ ] contextual features
- [ ] selected strategy
- [ ] agent answer
- [ ] quality
- [ ] pass/fail
- [ ] cost
- [ ] latency
- [ ] tokens
- [ ] tool calls
- [ ] errors

---

# 41. Deployment

- [x] Dockerfile
- [x] environment variables
- [x] reproducible local startup
- [x] pytest
- [ ] optional CI
- [ ] optional inexpensive hosted demo

---

# 42. Visualization

Only after core experiments.

Useful charts:

- [ ] cumulative reward
- [ ] pass rate by router
- [ ] cost/task by router
- [ ] latency/task
- [ ] strategy usage
- [ ] strategy usage by category
- [ ] quality/cost frontier
- [ ] category-ablation results
- [ ] routing under distribution shift

---

# 43. Final README

Include:

- [ ] problem
- [ ] motivation
- [ ] contextual-bandit formulation
- [ ] system architecture
- [ ] agents
- [ ] evaluation methodology
- [ ] reward design
- [ ] dataset
- [ ] baseline results
- [ ] routing algorithms
- [ ] main results
- [ ] ablations
- [ ] evaluator calibration
- [ ] limitations
- [ ] future work
- [ ] setup instructions

---

# 44. Interview Questions I Should Be Able to Answer

- [ ] What problem does the router solve?
- [ ] Why not always use the strongest model?
- [ ] Why is this a contextual bandit?
- [ ] Why not full reinforcement learning?
- [ ] What is exploration vs exploitation?
- [ ] How does epsilon-greedy work?
- [ ] How does UCB work?
- [ ] How does LinUCB work?
- [ ] Why does context matter?
- [ ] How did you evaluate heterogeneous tasks?
- [ ] Why use deterministic graders where possible?
- [ ] Why use an LLM judge?
- [ ] How did you calibrate the LLM judge?
- [ ] Why isn't one quality threshold appropriate for every evaluator?
- [ ] How did you prevent evaluation leakage?
- [ ] How did you balance quality against cost and latency?
- [ ] What happens under distribution shift?
- [ ] What did the category ablation show?
- [ ] What would inferred-category routing change?
- [ ] How would you deploy this at production scale?

---

# 45. 5-Week Timeline

## Week 1 — Benchmark + Evaluation

- [ ] repository setup
- [ ] Task schema
- [ ] EvaluationResult schema
- [ ] encode 12 tasks
- [ ] deterministic evaluators
- [ ] rubric evaluator
- [ ] human calibration

### Milestone

```text
Task → candidate answer → evaluator → quality/pass
```

works reliably.

---

## Week 2 — Agent Strategies + Baselines

- [ ] AgentResult
- [ ] direct agent
- [ ] strong agent
- [ ] tool agent
- [ ] instrumentation
- [ ] run all agents on all seed tasks
- [ ] analyze routing opportunity
- [ ] random router
- [ ] static router

### Milestone

Have an empirical performance matrix for:

```text
task × strategy
```

---

## Week 3 — Bandits

- [ ] experiment storage
- [ ] epsilon-greedy
- [ ] UCB
- [ ] feature extractor
- [ ] begin LinUCB

### Milestone

Router learns from repeated outcomes.

---

## Week 4 — Context + Experiments

- [ ] finish LinUCB
- [ ] expand benchmark if necessary
- [ ] main experiment
- [ ] category ablation
- [ ] reward sensitivity
- [ ] distribution shift
- [ ] graphs

### Milestone

Have defensible experimental conclusions.

---

## Week 5 — Production + Presentation

- [ ] FastAPI
- [ ] observability
- [ ] Docker
- [ ] README
- [ ] diagrams
- [ ] final graphs
- [ ] resume bullets
- [ ] interview walkthrough

Stretch only if time remains:

- [ ] inferred category
- [ ] quality-constrained router

---

# 46. Parking Lot

Do not add these until the core system works:

- [ ] inferred task categories
- [ ] embeddings as router context
- [ ] neural routing policy
- [ ] Thompson sampling
- [ ] model cascades
- [ ] verifier-based escalation
- [ ] fine-tuning
- [ ] RL over LLM weights
- [ ] multi-agent orchestration
- [ ] MCP integrations
- [ ] complex frontend
- [ ] Kubernetes
- [ ] distributed inference

---

# 47. Current Status

## Completed Design Decisions

- [x] Selected Adaptive Agent Router
- [x] Selected contextual-bandit framing
- [x] Defined three agent strategies
- [x] Defined four task categories
- [x] Designed 12 seed tasks
- [x] Defined deterministic vs semantic evaluation
- [x] Decided on task-specific explanation rubrics
- [x] Defined partial credit for extraction
- [x] Decided to human-calibrate LLM judge
- [x] Decided judge should be blind to model identity
- [x] Rejected universal arbitrary quality threshold
- [x] Defined initial quality/cost/latency objective
- [x] Identified quality-constrained routing as later experiment
- [x] Identified inferred-category routing as later experiment
- [x] Locked conceptual `Task` schema
- [x] Locked conceptual `EvaluationResult` schema

---

# 48. Current Next Steps

## Right Now

- [ ] Create repository
- [ ] Initialize Python project
- [ ] Install Pydantic + pytest
- [ ] Implement `TaskCategory`
- [ ] Implement `EvaluationType`
- [ ] Implement `Task`
- [ ] Implement `EvaluationResult`
- [ ] Write validation tests
- [ ] Instantiate A1
- [ ] Instantiate E3

## First Coding Checkpoint

We should be able to run something conceptually like:

```python
task = Task(...)
print(task)
```

for both:

```text
A1 → deterministic numeric evaluation
E3 → semantic rubric evaluation
```

and have Pydantic reject invalid task definitions.

Once that works, move to the evaluators.

---

# Session Log

## Session 1

### Decisions

- Selected Adaptive Agent Router
- Contextual bandit rather than full RL
- Three strategies: direct, strong, tool

### Learned

- context
- actions
- reward
- normal vs contextual bandits
- quality/cost/latency tradeoff

---

## Session 2

### Decisions

- Four task categories
- 12-task seed benchmark
- Hybrid evaluation
- Rubric-based LLM judge
- Human evaluator calibration
- Per-task quality requirements

### Learned

- evaluation design
- LLM-as-judge
- deterministic vs semantic grading
- partial credit
- evaluator leakage
- reward normalization

---

## Session 3

### Goal

Start implementation.

### Next

- Task schema
- EvaluationResult schema
- tests
