# Adaptive Agent Router

> [!summary] Project Goal
> Build a contextual-bandit-based AI router that learns which agent strategy to use for each incoming task while maintaining acceptable quality and minimizing unnecessary inference cost and latency.

---

## 1. Core Research Question

Can a learned contextual router outperform static routing policies by selecting the cheapest or fastest agent strategy that is likely to meet the required quality level for a task?

```text
task context → routing policy → direct / strong / tool → execution
    → evaluation → quality + cost + latency → learning signal
```

## 2. Project Constraints

- [x] Target completion: 4–5 weeks
- [x] Learning is the primary objective; resume strength is secondary
- [x] Do not fine-tune an LLM or perform expensive large-model RL rollouts
- [x] Start with three agent strategies and a small benchmark
- [x] Prioritize experiments over frontend work
- [x] Use deterministic evaluation wherever possible
- [x] Use LLM judges only where semantic evaluation is necessary
- [x] Keep routing and evaluation decoupled

## 3. What I Want to Learn

### Reinforcement Learning

- [ ] Multi-armed bandits and contextual bandits
- [ ] Exploration vs. exploitation
- [ ] Reward design and regret
- [ ] Epsilon-greedy, UCB, and LinUCB
- [ ] Why routing can be modeled as a contextual bandit
- [ ] Bandits vs. full RL
- [ ] Constrained optimization vs. weighted reward
- [ ] Distribution shift and policy adaptation

### LLM and Agent Engineering

- [ ] Model routing, tool calling, structured outputs
- [ ] Token accounting, cost, and latency measurement
- [ ] Model quality/cost tradeoffs
- [ ] LLM-as-judge evaluation and calibration
- [ ] Common agent interfaces and execution traces
- [ ] Failure handling and strategy selection

### AI Engineer / FDE Skills

- [ ] FastAPI inference service
- [ ] Experiment and evaluation infrastructure
- [ ] Observability, structured logging, and reproducible benchmarks
- [ ] Cost/latency optimization, Docker, and communicating tradeoffs

## 4. System Formulation

### Context

Only information available before action selection may be used by the router. Initial features may include:

- task category
- prompt length
- whether the prompt contains numbers or code
- number of numeric inputs
- number of requested output fields or inputs where applicable

The router cannot use information generated after agent execution.

### Actions

#### `direct`

Cheap model, low cost and latency, intended for simple tasks.

#### `strong`

Higher-capability model or higher-reasoning configuration, with greater expected quality but higher cost and latency.

#### `tool`

Cheap model with deterministic tool access, initially a calculator and/or restricted Python computation. The model can delegate exact computation to the tool and then produce the final answer.

## 5. Product Objective

Maintain acceptable answer quality and avoid paying for additional model capability unless the cheaper strategy is expected to fall below the task's quality requirement.

Long-term formulation:

```text
minimize cost + latency
subject to the task quality requirement being satisfied
```

There is not initially one universal quality threshold. Each evaluation type defines what constitutes passing.

## 6. Initial Dataset

Create a frozen 12-task seed development set:

- Arithmetic / computation: A1–A3
- Logic / reasoning: R1–R3
- Factual / explanatory: E1–E3
- Structured extraction: S1–S3

Run all three strategies on every task. Predicted winners are hypotheses, not grading criteria.

## 7. Seed Tasks

### Arithmetic / Computation

#### A1 — Simple multiplication

**Prompt:** What is 37 × 19? Return only the number.

**Expected:** `703`  
**Evaluation:** Numeric exact match  
**Hypothesis:** Tool is cheap and reliable, although direct should also succeed.

#### A2 — Multi-step percentage

**Prompt:** A laptop originally costs $1,240. It is discounted by 18%, and then a 6% sales tax is applied to the discounted price. What is the final price? Round to the nearest cent and return only the amount.

**Expected:** `1077.15`  
**Evaluation:** Numeric comparison with rounding tolerance  
**Hypothesis:** Tool should reduce arithmetic error.

#### A3 — Statistics

**Prompt:** Given the values 14, 22, 9, 31, 18, 27, 11, 25, 19, and 34, calculate their mean and population standard deviation. Round both to two decimal places.

**Expected:** mean `21.00`; population standard deviation approximately `7.76`  
**Evaluation:** Numeric comparison with tolerance  
**Hypothesis:** Tool should have the strongest quality/cost profile.

### Logic / Reasoning

#### R1 — Basic deduction

**Prompt:** All lorps are mavens. No mavens are tals. If Zed is a lorp, can Zed be a tal? Answer yes or no and briefly explain.

**Expected:** No  
**Evaluation:** Exact conclusion  
**Hypothesis:** Direct should be sufficient.

#### R2 — Constraint reasoning

**Prompt:** Four people—Ava, Ben, Chloe, and Diego—stand in a line. Ava is somewhere before Ben. Chloe is immediately after Ben. Diego is before Ava. Who must be first?

**Expected:** Diego  
**Evaluation:** Exact answer  
**Hypothesis:** Strong may provide increased reliability. This is an empirical hypothesis.

#### R3 — Multi-constraint scheduling

**Prompt:** Five presentations—A, B, C, D, and E—are scheduled one per time slot from 1 through 5. A must occur before C. B must occur immediately after D. E cannot be first or last. C must occur after E. If A is scheduled second, which presentation must be first?

**Expected:** D  
**Evaluation:** Exact answer  
**Hypothesis:** Strong may have an advantage on multi-step constraint reasoning.

### Factual / Explanatory

#### E1 — Hash table

**Prompt:** In two sentences or fewer, explain what a hash table is and why it is useful.

**Reference concepts:** key-value association; hashing; efficient lookup/insertion.  
**Evaluation:** Task-specific LLM rubric  
**Hypothesis:** Direct should provide sufficient quality.

#### E2 — Interest rates and inflation

**Prompt:** Explain why increasing interest rates can reduce inflation. Your explanation should connect interest rates, borrowing, spending or investment, aggregate demand, and price pressure.

**Reference concepts:** more expensive borrowing; reduced consumption/investment; lower aggregate demand; reduced price pressure.  
**Evaluation:** Task-specific LLM rubric  
**Hypothesis:** Strong may provide better reasoning and completeness.

#### E3 — Redis caching tradeoffs

**Prompt:** An API currently returns database results directly from PostgreSQL on every request. The engineering team is considering adding a Redis cache. Explain two situations where caching would substantially improve the system and two situations where caching could introduce problems. Briefly explain why in each case.

**Reference concepts:** repeated or read-heavy queries, expensive operations, stable data, reduced load and latency; stale data, invalidation complexity, rapidly changing data, poor hit rate, consistency, or infrastructure complexity.  
**Evaluation:** Task-specific LLM rubric  
**Hypothesis:** Strong may provide more complete technical reasoning.

### Structured Extraction

#### S1 — Basic field extraction

Extract `name`, `plan`, and `seats` from: “Hi, I'm Maya Chen from Acme Robotics. We'd like to upgrade our current account to the Enterprise plan for our 48 employees.” Return valid JSON with exactly those keys.

```json
{"name":"Maya Chen","plan":"Enterprise","seats":48}
```

#### S2 — Extraction with distractors

Extract `order_id`, `email`, `action`, and `item` from: “I placed order #A83921 last Tuesday using samira.khan@example.com. The headphones are fine, but the USB-C docking station arrived damaged. I'd like a replacement for the docking station rather than a refund.”

```json
{"order_id":"A83921","email":"samira.khan@example.com","action":"replacement","item":"USB-C docking station"}
```

#### S3 — Multi-record extraction

Convert this text into a JSON array whose objects contain `service`, `status`, and `region`: “Payments API is operational in us-east-1. Search is degraded in eu-west-1. Authentication is operational in us-west-2. Analytics is unavailable in eu-central-1.”

Expected records are Payments API/operational/us-east-1, Search/degraded/eu-west-1, Authentication/operational/us-west-2, and Analytics/unavailable/eu-central-1.

**Evaluation for S1–S3:** Parse JSON and compare fields or records; use partial credit for structured fields.

## 8. Locked Task Schema

Use Pydantic so invalid benchmark tasks fail early.

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

```python
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
```

Numeric, exact, and structured tasks require `expected_answer`; rubric tasks require a rubric. Invalid combinations should be rejected.

## 9. Evaluation Philosophy

Use the simplest reliable evaluator for each task:

```text
Arithmetic   → NumericEvaluator
Logic        → ExactEvaluator
Extraction   → StructuredEvaluator
Explanation  → RubricLLMEvaluator
```

Every evaluator returns the same `EvaluationResult`; routing depends only on standardized outputs, not evaluator implementation.

## 10. Locked Evaluation Result Schema

```python
class EvaluationResult(BaseModel):
    quality: float
    passed: bool
    grader_type: str
    component_scores: dict[str, float] = {}
    feedback: str | None = None
```

Enforce `0 <= quality <= 1`. Preserve raw component scores and feedback for analysis, while the router consumes `quality` and `passed`.

## 11. Evaluators and Passing Rules

- **Numeric:** correct is `1`, incorrect is `0`; use tolerance for rounded values.
- **Exact:** normalize the answer and grade the required conclusion only.
- **Structured:** `quality = correct_fields / total_fields`; v1 passes only when all required fields are correct.
- **Rubric:** score task-specific dimensions and normalize the weighted result into `[0, 1]`.

Suggested rubric dimensions:

| Dimension | Weight |
|---|---:|
| Technical correctness | 40% |
| Completeness | 25% |
| Reasoning / depth | 15% |
| Clarity | 10% |
| Relevance / concision | 10% |

Use anchored 0–4 component scores. Correctness dominates style; do not reward verbosity or fancy writing by itself. The judge must not know model identity, cost, latency, or routing strategy.

## 12. Human Calibration

The LLM evaluator is not ground truth. Collect representative explanation responses, manually score a subset, compare human and judge scores, inspect disagreements, refine rubrics, and determine explanation pass thresholds empirically. Human evaluation is for calibration and auditing, not for grading every experiment.

## 13. Agent Result Schema

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

All strategies must return this common object so execution and evaluation remain interchangeable.

## 14. Reward and Normalization

Store quality, cost, and latency separately; never save only a combined reward.

```text
reward = quality - λ(cost_normalized) - β(latency_normalized)
```

Quality should dominate initially. Quality is already `[0, 1]`; normalize cost and latency against fixed reference values rather than dataset-dependent min/max values:

```text
normalized_cost = actual_cost / reference_cost
normalized_latency = actual_latency / reference_latency
```

Choose reference values after collecting baseline measurements. A later production-like objective is to choose the cheapest/fastest strategy that is sufficiently likely to pass the task requirement, modeled as `P(pass | context, action)`.

## 15. Repository Structure

```text
adaptive-agent-router/
├── src/adaptive_router/
│   ├── models/{task.py,agent_result.py,evaluation_result.py}
│   ├── agents/{base.py,direct.py,strong.py,tool.py}
│   ├── evaluation/{base.py,numeric.py,exact.py,structured.py,rubric.py}
│   ├── routing/{base.py,random.py,static.py,epsilon_greedy.py,ucb.py,linucb.py}
│   ├── features/extractor.py
│   └── experiments/
├── data/seed_tasks.json
├── tests/
├── scripts/
├── README.md
└── pyproject.toml
```

Do not create every file immediately; add structure only as needed.

## 16. Implementation Order

### Phase 1 — Data Models

- [ ] Create the project and Python environment
- [ ] Install/configure Pydantic
- [ ] Implement enums, `Task`, `EvaluationRubric`, and `EvaluationResult`
- [ ] Add schema validation tests

**Definition of done:** valid A1 and E3 task objects can be created, while a numeric task without an expected answer and a rubric task without a rubric are rejected.

### Phase 2 — Seed Dataset

- [ ] Encode all 12 tasks
- [ ] Load and validate JSON/YAML through Pydantic
- [ ] Freeze the seed dataset
- [ ] Keep predicted winners separate from router-visible data

### Phase 3 — Evaluators

- [ ] Numeric evaluator: parsing, tolerance, tests
- [ ] Exact evaluator: normalization and tests
- [ ] Structured evaluator: JSON parsing, field comparison, partial credit, tests
- [ ] Rubric evaluator: structured judge response, task-specific rubric, weighted score, blind model identity, tests
- [ ] Human-calibrate a representative explanation subset

### Phase 4 — Agent Strategies

- [ ] Define a common agent interface
- [ ] Implement direct, strong, and tool strategies
- [ ] Capture latency, token counts, tool calls, errors, and estimated cost
- [ ] Add mocked tests before relying on live model calls

### Phase 5 — Baselines

- [ ] Run every task through every strategy
- [ ] Implement random routing
- [ ] Implement static policies such as always-direct, always-strong, and category-based routing
- [ ] Save raw results and metrics
- [ ] Compare quality, pass rate, cost, latency, and reward

### Phase 6 — Contextual Bandits

- [ ] Implement epsilon-greedy
- [ ] Implement UCB
- [ ] Implement LinUCB with explicit feature vectors
- [ ] Log selected action, context, reward, and outcome
- [ ] Track cumulative reward and regret against a hindsight oracle

### Phase 7 — Experiments

- [ ] Compare learned policies with baselines
- [ ] Run multiple trials because model outputs are stochastic
- [ ] Evaluate exploration rates and reward coefficients
- [ ] Analyze per-task and per-category behavior
- [ ] Test whether the router learns within-category difficulty differences
- [ ] Preserve a held-out set or later-generated validation set where feasible

### Phase 8 — Service and Observability

- [ ] Add a small FastAPI inference endpoint
- [ ] Add structured experiment logs and reproducible configuration
- [ ] Add Docker support if time permits
- [ ] Add a minimal results report; avoid building a substantial frontend

## 17. Evaluation and Experiment Design

The benchmark must run all three strategies on the same frozen tasks. Keep task definitions, router-visible features, hypotheses, and evaluation results separate to avoid leakage. Do not use the expected winning strategy as an input feature.

For each run record:

```text
task_id, strategy, context, answer, quality, passed,
cost_usd, latency_seconds, reward, error, timestamp, run_id
```

Report aggregate and per-task metrics. Because the seed set is small, treat results as development evidence rather than a claim of general production performance.

## 18. Success Criteria

The project succeeds if it demonstrates:

- deterministic and rubric-based evaluators behind one stable schema;
- three interchangeable agent strategies with measured quality, cost, and latency;
- baseline policies and at least one functioning contextual-bandit policy;
- reproducible experiments showing when learned routing helps or does not help;
- an honest analysis of judge calibration, stochasticity, small-data limitations, and distribution shift.

The goal is not to prove that the strong model always wins on difficult tasks. The goal is to measure the quality/cost/latency landscape and learn a policy from it.

## 19. Longer-Term Extensions

- Replace explicit category labels with a cheap task classifier producing computation, reasoning, extraction, and ambiguity scores.
- Learn `P(pass | context, action)` and enforce quality constraints directly.
- Add uncertainty estimates and safer fallback routing.
- Expand beyond the seed set with generated or production-like tasks.
- Study distribution shift and policy adaptation.

These are follow-up experiments, not prerequisites for the first working router.
