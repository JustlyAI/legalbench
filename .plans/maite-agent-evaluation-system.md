# Maite AI Agent Evaluation System - Product Requirements Document

## Feature Overview

### Purpose
Implement a rigorous, minimal evaluation framework to systematically measure the performance of the Maite AI Agent (built with Claude Agent SDK) on legal reasoning tasks from LegalBench. This system will enable:
- Quantifiable comparison of agent performance across runs
- Identification of agent strengths and weaknesses in legal reasoning
- Detection of regressions or improvements during agent development
- Data-driven decision making for agent optimization

### Context
**Current State:**
- LegalBench repository with 162 legal reasoning tasks available via HuggingFace
- Core LegalBench modules: `tasks.py`, `evaluation.py`, `utils.py`
- Canonical usage pattern documented in `UsingLegalBench.ipynb`
- Maite AI Agent exists in separate repository (`s_c_workbench`) built with Claude Agent SDK
- No systematic evaluation pipeline connecting Maite to LegalBench tasks
- Manual, ad-hoc testing is the current approach

**What We're Building:**
A clean bridge between Maite and LegalBench following canonical patterns:
1. Load tasks via `datasets.load_dataset("nguha/legalbench", task_name)`
2. Load prompt templates from `tasks/{task_name}/base_prompt.txt`
3. Generate prompts using `generate_prompts(template, data_df)` from utils.py
4. Execute Maite against each task sample
5. Capture execution traces (tool calls, chain steps, tokens, timing)
6. Evaluate outputs using `evaluate(task, generations, answers)` from evaluation.py
7. Persist results for comparison across runs
8. Generate comparison reports

**NOT Building (Out of Scope):**
- Full-featured agent monitoring dashboard (minimal CLI reports only)
- Manual evaluation UI for rule_qa tasks (mark as unsupported for now)
- Real-time streaming evaluation
- Multi-agent comparison (single agent focus)
- RAG-specific features (unless Maite already has retrieval)

### Value Proposition
- **Objectivity**: Replace subjective "it seems better" with hard numbers
- **Reproducibility**: Same tasks, same metrics, comparable results
- **Speed**: Run 5-10 representative tasks in <30 minutes vs manual testing
- **Insight**: Understand WHERE agent fails (issue spotting vs rule application vs interpretation)
- **Iteration velocity**: Quickly validate agent improvements

## Requirements Analysis

### Core Requirements

**Must Have (MVP):**
1. Load subset of LegalBench tasks (5-10 tasks spanning all categories)
2. Execute Maite agent against task samples with proper prompt formatting
3. Capture execution metadata: response, execution time, tool calls, tokens used
4. Evaluate outputs using existing `evaluation.py` metrics (balanced accuracy, F1, etc.)
5. Persist results to JSON with run metadata (timestamp, agent version, config)
6. CLI command to run evaluation: `python run_eval.py --tasks hearsay,contract_nli_*`
7. CLI command to compare runs: `python compare_runs.py run1.json run2.json`
8. Support existing LegalBench data format (TSV) and prompt templates

**Should Have (Nice to Have):**
9. Sample size control (e.g., run 10 samples per task vs full test set)
10. Category-based selection (e.g., `--category CONCLUSION_TASKS`)
11. Error handling and retry logic for agent failures
12. Basic statistics in comparison report (mean, std dev, improvement %)

**Won't Have (Future):**
- Web UI or dashboard
- Database backend (JSON files sufficient)
- Manual evaluation pipeline
- Cross-agent comparison
- Streaming or real-time updates

### Success Criteria

**Definition of Done:**
1. ✅ Can run `python run_eval.py --tasks hearsay,personal_jurisdiction --samples 20` successfully
2. ✅ Produces `results/run_YYYYMMDD_HHMMSS.json` with all required fields
3. ✅ Can run `python compare_runs.py results/run1.json results/run2.json` and get readable diff
4. ✅ Evaluation uses correct metrics per task (balanced accuracy, F1, stemming, tolerance)
5. ✅ Results are reproducible (same input → same metrics ±0.01 for stochastic agent)
6. ✅ Fails gracefully with informative errors
7. ✅ Documented in README with example commands

**Validation:**
- Sanity check: Run on `hearsay` task, verify accuracy metric matches manual count
- Regression test: Run twice on same tasks, compare results are within expected variance
- Integration test: Run on all 5 task categories, verify each uses correct evaluation metric

### Dependencies

**LegalBench Dependencies:**
- `datasets` library - HuggingFace dataset loading
- `evaluation.py` - existing `evaluate()` function (no changes needed)
- `tasks.py` - task lists and category mappings (ISSUE_TASKS, RULE_TASKS, etc.)
- `utils.py` - `generate_prompts()` function with `{{column}}` placeholders
- `tasks/*/base_prompt.txt` - prompt templates (local files)
- HuggingFace dataset: `nguha/legalbench` with train/test splits

**Maite Agent Dependencies:**
- Access to Maite agent code (likely in `../s_c_workbench/`)
- Agent can be imported as Python module OR called via CLI/API
- Agent configuration (model, temperature, tools available)
- Authentication/API keys for Claude API

**External Dependencies:**
- pandas (already in LegalBench)
- scikit-learn (already in LegalBench)
- nltk (already in LegalBench)
- pydantic (for data validation) - NEW
- Claude Agent SDK (from Maite) - EXTERNAL

### Constraints

**Technical:**
- Must work with existing LegalBench code (no breaking changes to evaluation.py, utils.py, tasks.py)
- Must use HuggingFace datasets library (canonical LegalBench loading method)
- Must use `generate_prompts()` from utils.py (don't reimplement)
- Must use `evaluate()` from evaluation.py (don't reimplement)
- Prompt templates use `{{column_name}}` syntax matching DataFrame columns
- Some tasks require special evaluation (stemming, tolerance, F1 vs accuracy) - handled by evaluate()
- Agent may be slow (Claude API latency + multi-step reasoning)
- Must respect train/test split: train for few-shot examples, test for evaluation

**Practical:**
- Budget: Minimize API costs (start with small sample sizes)
- Time: Each full run should complete in <30 minutes
- Simplicity: Prefer simple JSON files over database for MVP
- Maintenance: Code should be easy to understand and modify

**Data Integrity:**
- Never modify HuggingFace dataset (read-only)
- Never modify `evaluation.py`, `utils.py`, or `tasks.py` core functions
- Preserve exact metrics from paper
- Use existing `evaluate()` function for all scoring (ensures consistency)

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    LegalBench Eval System                        │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌─────────────┐
│  Task Loader     │─────▶│  Agent Executor  │─────▶│  Evaluator  │
│                  │      │                  │      │             │
│ - Load TSV data  │      │ - Build prompts  │      │ - Run evals │
│ - Load templates │      │ - Call Maite     │      │ - Calc stats│
│ - Parse metadata │      │ - Capture trace  │      │ - Normalize │
└──────────────────┘      └──────────────────┘      └─────────────┘
         │                         │                        │
         │                         ▼                        │
         │                ┌──────────────────┐             │
         │                │  Maite Agent     │             │
         │                │  (Claude SDK)    │             │
         │                │                  │             │
         │                │ - Reasoning      │             │
         │                │ - Tool calls     │             │
         └───────────────▶│ - Multi-step     │◀────────────┘
                          └──────────────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │ Results Storage  │
                          │                  │
                          │ - JSON files     │
                          │ - Run metadata   │
                          │ - Traces         │
                          └──────────────────┘
```

### Data Models

**EvaluationRun:**
```python
{
  "run_id": "run_20251028_153000",
  "timestamp": "2025-10-28T15:30:00Z",
  "agent_config": {
    "name": "maite",
    "version": "0.1.0",
    "model": "claude-sonnet-4",
    "temperature": 0.0,
    "tools": ["search", "lookup"]
  },
  "tasks": ["hearsay", "personal_jurisdiction"],
  "samples_per_task": 20,
  "results": [
    {
      "task_name": "hearsay",
      "category": "CONCLUSION_TASKS",
      "samples_evaluated": 20,
      "metric": "balanced_accuracy",
      "score": 0.85,
      "avg_execution_time": 2.3,
      "avg_tokens": 450,
      "avg_tool_calls": 1.2,
      "errors": 0,
      "traces": [...]  // Optional detailed traces
    }
  ],
  "summary": {
    "total_samples": 40,
    "total_time": 92.0,
    "total_cost_usd": 0.45,
    "avg_accuracy": 0.82
  }
}
```

**TaskTrace:**
```python
{
  "sample_id": "hearsay_test_0",
  "input_text": "...",
  "expected_output": "Yes",
  "actual_output": "Yes",
  "is_correct": true,
  "execution_time": 2.1,
  "tokens": {"input": 200, "output": 50, "total": 250},
  "tool_calls": [
    {"tool": "search", "query": "hearsay rule", "result": "..."}
  ],
  "chain_steps": ["reasoning", "conclusion"],
  "error": null
}
```

### File Structure

```
legalbench/
├── eval_maite/                    # NEW - Evaluation framework
│   ├── __init__.py
│   ├── task_loader.py             # Load LegalBench tasks
│   ├── agent_wrapper.py           # Wrap Maite agent
│   ├── evaluator.py               # Run evaluation logic
│   ├── models.py                  # Pydantic models
│   └── utils.py                   # Helper functions
├── scripts/                       # NEW - CLI entry points
│   ├── run_eval.py                # Main evaluation script
│   └── compare_runs.py            # Compare two runs
├── results/                       # NEW - Output directory
│   ├── run_20251028_153000.json
│   └── run_20251028_160000.json
├── tests/                         # NEW - Tests
│   └── test_eval_maite.py
├── evaluation.py                  # EXISTING - No changes
├── tasks.py                       # EXISTING - No changes
├── utils.py                       # EXISTING - No changes
└── README.md                      # UPDATED - Add usage docs
```

## Implementation Phases

### Phase 1: Foundation - Data Loading & Models (~45 minutes)

**Objective:** Create infrastructure to load LegalBench tasks following canonical patterns

**Tasks:**
- [ ] Install required dependencies: `pip install datasets tqdm`
- [ ] Create `eval_maite/` module structure with `__init__.py`
- [ ] Implement `eval_maite/models.py` with Pydantic models:
  - `EvaluationRun`
  - `TaskResult`
  - `TaskTrace`
  - `AgentConfig`
- [ ] Implement `eval_maite/task_loader.py`:
  - `load_task_data(task_name)` - uses `datasets.load_dataset("nguha/legalbench", task_name)`
  - `get_test_df(task_name)` - returns test split as pandas DataFrame
  - `get_train_df(task_name)` - returns train split (for few-shot examples)
  - `load_prompt_template(task_name)` - reads `tasks/{task_name}/base_prompt.txt`
  - `get_task_category(task_name)` - checks ISSUE_TASKS, RULE_TASKS, etc. from tasks.py
- [ ] Add progress bar suppression: `datasets.utils.logging.set_verbosity_error()`
- [ ] Create `results/` directory with `.gitkeep`

**Files Created:**
- `eval_maite/__init__.py`
- `eval_maite/models.py`
- `eval_maite/task_loader.py`
- `results/.gitkeep`

**Verification:**
```bash
# Test HuggingFace loading (canonical method)
python -c "
import datasets
dataset = datasets.load_dataset('nguha/legalbench', 'hearsay')
print(f'Train: {len(dataset[\"train\"])}, Test: {len(dataset[\"test\"])}')
"

# Test our loader wrapper
python -c "
from eval_maite.task_loader import get_test_df, load_prompt_template, get_task_category
df = get_test_df('hearsay')
template = load_prompt_template('hearsay')
category = get_task_category('hearsay')
print(f'Loaded {len(df)} samples, category: {category}')
print(f'Template has placeholder: {\"{{\" in template}')
"
```

---

### Phase 2: Agent Integration (~60 minutes)

**Objective:** Create wrapper to execute Maite agent and capture execution traces

**Tasks:**
- [ ] Implement `eval_maite/agent_wrapper.py`:
  - `MaiteAgentWrapper` class
  - `__init__(agent_config)` - initialize agent
  - `execute(prompt)` - run agent and capture trace
  - `_build_trace(result)` - extract metrics from agent response
- [ ] Implement agent discovery logic:
  - Try importing from `../s_c_workbench` (relative path)
  - Fallback to environment variable `MAITE_AGENT_PATH`
  - Clear error message if agent not found
- [ ] Add error handling and retry logic (3 retries with exponential backoff)
- [ ] Add timeout handling (configurable, default 60s per sample)

**Files Modified/Created:**
- `eval_maite/agent_wrapper.py` (new)
- `eval_maite/utils.py` (new - helper functions)

**Verification:**
```bash
# Test agent execution on single prompt
python -c "from eval_maite.agent_wrapper import MaiteAgentWrapper; \
  agent = MaiteAgentWrapper({'model': 'claude-sonnet-4'}); \
  trace = agent.execute('Is hearsay admissible? Answer Yes or No.'); \
  print(f'Output: {trace.actual_output}, Time: {trace.execution_time}s')"

# Should execute successfully and print output
```

---

### Phase 3: Evaluation Engine (~45 minutes)

**Objective:** Integrate task loading, agent execution, and LegalBench evaluation following canonical patterns

**Tasks:**
- [ ] Implement `eval_maite/evaluator.py`:
  - `LegalBenchEvaluator` class
  - `__init__(agent_wrapper, task_loader)`
  - `evaluate_task(task_name, sample_size)` - run evaluation on one task
  - `evaluate_multiple(task_names, sample_size)` - run on multiple tasks
  - `_calculate_metrics(task_name, traces)` - delegate to evaluation.evaluate()
- [ ] **Critical**: Use canonical LegalBench functions:
  - Import `generate_prompts` from `utils`
  - Import `evaluate` from `evaluation`
  - Import task category lists from `tasks` (ISSUE_TASKS, RULE_TASKS, etc.)
- [ ] Implement evaluation workflow:
  1. Load test DataFrame via task_loader
  2. Load prompt template via task_loader
  3. Generate prompts: `prompts = generate_prompts(template, test_df)`
  4. Execute agent on each prompt (collect generations list)
  5. Extract answers: `answers = test_df["answer"].tolist()`
  6. Calculate score: `score = evaluate(task_name, generations, answers)`
- [ ] Add progress indicators using `tqdm` (matches notebook style)

**Files Modified/Created:**
- `eval_maite/evaluator.py` (new)

**Verification:**
```bash
# Test canonical evaluation pattern
python -c "
from eval_maite.evaluator import LegalBenchEvaluator
from eval_maite.agent_wrapper import MaiteAgentWrapper
from eval_maite.task_loader import TaskLoader

agent = MaiteAgentWrapper({'model': 'claude-sonnet-4'})
loader = TaskLoader()
evaluator = LegalBenchEvaluator(agent, loader)
result = evaluator.evaluate_task('hearsay', sample_size=5)
print(f'Task: {result.task_name}')
print(f'Score: {result.score:.3f}')
print(f'Metric: {result.metric}')
"

# Should run 5 samples using canonical evaluate() function
```

---

### Phase 4: CLI Interface (~30 minutes)

**Objective:** Create user-facing scripts for running and comparing evaluations

**Tasks:**
- [ ] Implement `scripts/run_eval.py`:
  - Argument parsing (--tasks, --category, --samples, --output)
  - Initialize evaluator
  - Run evaluation
  - Save results to JSON
  - Print summary to console
- [ ] Add argument validation:
  - Check task names exist
  - Check category names valid
  - Default sample size = 20
- [ ] Implement result saving:
  - Generate run_id with timestamp
  - Save to `results/run_{timestamp}.json`
  - Pretty-print JSON with indent=2

**Files Created:**
- `scripts/run_eval.py`
- `scripts/__init__.py`

**Verification:**
```bash
# Run evaluation on 2 tasks
python scripts/run_eval.py --tasks hearsay,personal_jurisdiction --samples 10

# Should:
# 1. Run 20 samples total (10 per task)
# 2. Print progress
# 3. Save results/run_*.json
# 4. Print summary
```

---

### Phase 5: Comparison Tool (~30 minutes)

**Objective:** Create tool to compare two evaluation runs

**Tasks:**
- [ ] Implement `scripts/compare_runs.py`:
  - Load two JSON result files
  - Compare task-by-task metrics
  - Calculate deltas (absolute and percentage)
  - Generate comparison report (markdown table)
- [ ] Add comparison visualizations:
  - Side-by-side metrics table
  - Improvement/regression indicators (↑/↓/→)
  - Statistical significance (if applicable)
- [ ] Handle missing tasks (in one run but not other)

**Files Created:**
- `scripts/compare_runs.py`

**Verification:**
```bash
# Run two evaluations
python scripts/run_eval.py --tasks hearsay --samples 20 --output run1.json
python scripts/run_eval.py --tasks hearsay --samples 20 --output run2.json

# Compare them
python scripts/compare_runs.py results/run1.json results/run2.json

# Should print comparison table with metrics
```

---

### Phase 6: Testing & Documentation (~45 minutes)

**Objective:** Ensure reliability and usability

**Tasks:**
- [ ] Write unit tests in `tests/test_eval_maite.py`:
  - Test task loading with mock data
  - Test agent wrapper with mock agent
  - Test evaluation with mock responses
  - Test metric calculations
- [ ] Add integration test:
  - End-to-end test on single task with 2 samples
  - Verify JSON output format
- [ ] Update `README.md`:
  - Add "Evaluating Maite Agent" section
  - Document installation steps
  - Add example commands
  - Explain output format
- [ ] Add `requirements.txt` for eval_maite dependencies:
  - pydantic>=2.0
  - (other new dependencies)

**Files Created/Modified:**
- `tests/test_eval_maite.py` (new)
- `README.md` (updated)
- `requirements.txt` (updated if needed)

**Verification:**
```bash
# Run tests
pytest tests/test_eval_maite.py -v

# All tests should pass

# Verify documentation
cat README.md | grep -A 20 "Evaluating Maite Agent"
# Should show clear instructions
```

---

### INTERIM REPORT (Before Phase 7)

**Before beginning Phase 7 (Polish & Optimization), provide a report covering:**

1. **Functionality Status:**
   - Which tasks can be evaluated successfully?
   - Which tasks are failing? Why?
   - Are metrics matching expected values?

2. **Performance Metrics:**
   - Actual execution time per task
   - Cost per evaluation run
   - Bottlenecks identified

3. **Code Quality:**
   - Test coverage achieved
   - Error handling robustness
   - Code clarity and maintainability

4. **User Experience:**
   - Are CLI commands intuitive?
   - Are error messages helpful?
   - Is output readable?

5. **Deviations from Plan:**
   - What changed during implementation?
   - What assumptions were wrong?
   - What additional features were needed?

6. **Recommendations for Phase 7:**
   - What needs polishing most?
   - What optimizations are highest priority?
   - What edge cases need handling?

---

### Phase 7: Polish & Optimization (~30 minutes)

**Objective:** Refine based on interim report findings

**Tasks (to be determined based on interim report, likely include):**
- [ ] Fix any failing tasks identified in interim report
- [ ] Optimize slow operations (e.g., parallel execution if needed)
- [ ] Improve error messages based on common failures
- [ ] Add configuration file support (e.g., `eval_config.yaml`)
- [ ] Add logging to file (not just console)
- [ ] Handle edge cases discovered during testing
- [ ] Add task selection shortcuts:
  - `--category CONCLUSION_TASKS`
  - `--all` for all tasks
  - `--quick` for 5-task quick smoke test

**Verification:**
```bash
# Run comprehensive test
python scripts/run_eval.py --category CONCLUSION_TASKS --samples 10

# Should run all conclusion tasks successfully

# Run quick smoke test
python scripts/run_eval.py --quick

# Should run 5 tasks (one per category) with 5 samples each
```

## Success Metrics

### Quantitative Metrics
- **Coverage**: Support 100% of LegalBench evaluation metrics (balanced accuracy, F1, stemming, tolerance, contains)
- **Reliability**: <5% failure rate on task execution (excluding agent errors)
- **Speed**: Evaluate 10 samples across 5 tasks in <5 minutes
- **Reproducibility**: Same input yields same metrics within ±2% (accounting for agent stochasticity)

### Qualitative Metrics
- **Usability**: New user can run evaluation with ≤3 commands after reading README
- **Clarity**: Comparison output clearly shows what improved/regressed
- **Maintainability**: Another developer can add new task support in <30 minutes

## Risks & Mitigation

### Technical Risks

**Risk 1: Maite Agent Integration Complexity**
- **Severity:** High
- **Probability:** Medium
- **Impact:** Can't execute agent, project blocked
- **Mitigation:**
  - Phase 2 focuses entirely on agent integration
  - Create mock agent for testing if real agent unavailable
  - Document agent interface requirements clearly
  - User provides agent import path via config

**Risk 2: LegalBench Data Format Changes**
- **Severity:** Low
- **Probability:** Very Low
- **Impact:** Task loading breaks
- **Mitigation:**
  - Use existing LegalBench code (evaluation.py, utils.py)
  - Don't modify core files
  - Add validation on data load

**Risk 3: Agent Performance Too Slow**
- **Severity:** Medium
- **Probability:** Medium
- **Impact:** Evaluations take >1 hour, unusable
- **Mitigation:**
  - Start with small sample sizes (10-20)
  - Add sample size control from day 1
  - Consider parallel execution in Phase 7 if needed
  - Add timeout controls

**Risk 4: Cost Too High**
- **Severity:** Medium
- **Probability:** Low
- **Impact:** Can't afford to run evaluations regularly
- **Mitigation:**
  - Sample size control
  - Quick smoke test mode (5 samples × 5 tasks = 25 samples)
  - Track cost per run in results
  - Consider caching for development

### Project Risks

**Risk 5: Scope Creep**
- **Severity:** Medium
- **Probability:** High
- **Impact:** MVP delayed, never complete
- **Mitigation:**
  - Clear "Won't Have" list in requirements
  - Time-box each phase strictly
  - Defer optimization to Phase 7
  - Use "good enough" over "perfect"

**Risk 6: Dependency on External Agent**
- **Severity:** High
- **Probability:** Low
- **Impact:** Can't test without Maite agent
- **Mitigation:**
  - Create mock agent early (Phase 2)
  - Define clear agent interface
  - Make agent implementation pluggable
  - Test with mock before real agent

## Implementation Notes

### Design Decisions

**Why JSON instead of Database?**
- Simpler for MVP
- Easy to version control
- No setup overhead
- Easy to inspect manually
- Can migrate to DB later if needed

**Why not modify evaluation.py?**
- Preserve LegalBench integrity
- Avoid merge conflicts if upstream updates
- Keep our code separate and modular
- Easier to maintain

**Why CLI instead of API?**
- Simpler for single-user use case
- Faster to implement
- Easy to run in CI/CD
- Can add API later if needed

**Why limited task selection initially?**
- 162 tasks is overwhelming
- 5-10 tasks sufficient for validation
- Faster iteration
- Lower cost
- Easy to expand later

### Configuration Example

```yaml
# eval_config.yaml (future enhancement)
agent:
  name: maite
  version: 0.1.0
  model: claude-sonnet-4
  temperature: 0.0
  timeout: 60

evaluation:
  sample_size: 20
  tasks:
    - hearsay
    - personal_jurisdiction
    - contract_nli_confidentiality_of_agreement
    - definition_extraction
    - ssla_individual_defendants

  quick_test:
    enabled: true
    tasks: ["hearsay", "contract_qa", "diversity_1", "definition_extraction", "overruling"]
    sample_size: 5

output:
  directory: results/
  include_traces: false  # Set true for debugging
  format: json
```

### Example Usage

```bash
# Installation
cd legalbench
pip install -e .
pip install pydantic

# Configure agent path (if not in ../s_c_workbench)
export MAITE_AGENT_PATH=/path/to/maite/agent.py

# Run quick smoke test
python scripts/run_eval.py --quick

# Run specific tasks
python scripts/run_eval.py --tasks hearsay,personal_jurisdiction,contract_nli_limited_use --samples 20

# Run all conclusion tasks
python scripts/run_eval.py --category CONCLUSION_TASKS --samples 10

# Compare two runs
python scripts/compare_runs.py results/run_20251028_153000.json results/run_20251028_160000.json

# Output
┌───────────────────────────┬─────────┬─────────┬─────────┐
│ Task                      │ Run 1   │ Run 2   │ Delta   │
├───────────────────────────┼─────────┼─────────┼─────────┤
│ hearsay                   │ 0.850   │ 0.870   │ +0.020↑ │
│ personal_jurisdiction     │ 0.780   │ 0.760   │ -0.020↓ │
│ contract_nli_limited_use  │ 0.920   │ 0.920   │  0.000→ │
└───────────────────────────┴─────────┴─────────┴─────────┘
Overall: 0.850 → 0.850 (0.000)
```

## Appendix A: Canonical LegalBench Usage Pattern

This section documents the canonical pattern from `UsingLegalBench.ipynb` that our implementation MUST follow:

```python
# ============================================
# CANONICAL LEGALBENCH EVALUATION PATTERN
# (from UsingLegalBench.ipynb)
# ============================================

import datasets
from tasks import TASKS, ISSUE_TASKS, CONCLUSION_TASKS  # etc.
from utils import generate_prompts
from evaluation import evaluate

# Suppress HuggingFace progress bars
datasets.utils.logging.set_verbosity_error()

# 1. Load task data from HuggingFace
task_name = "hearsay"
dataset = datasets.load_dataset("nguha/legalbench", task_name)

# 2. Get test split as pandas DataFrame
test_df = dataset["test"].to_pandas()
train_df = dataset["train"].to_pandas()  # Optional: for few-shot examples

# 3. Load prompt template from local file
with open(f"tasks/{task_name}/base_prompt.txt") as f:
    prompt_template = f.read()

# 4. Generate prompts using canonical function
prompts = generate_prompts(prompt_template=prompt_template, data_df=test_df)
# Returns list of prompts with {{placeholders}} filled from DataFrame columns

# 5. Execute model/agent on each prompt (YOUR CODE HERE)
generations = []
for prompt in prompts:
    output = your_model_or_agent(prompt)  # Replace with actual agent call
    generations.append(output)

# 6. Extract ground truth answers
answers = test_df["answer"].tolist()

# 7. Evaluate using canonical function
score = evaluate(task_name, generations, answers)
# Automatically uses correct metric (balanced accuracy, F1, etc.) per task

print(f"Score: {score:.3f}")

# ============================================
# KEY POINTS:
# ============================================
# - MUST use datasets.load_dataset() (not local TSV files)
# - MUST use generate_prompts() from utils (not custom implementation)
# - MUST use evaluate() from evaluation (handles all special cases)
# - Train split = few-shot examples, Test split = evaluation samples
# - DataFrame must have "answer" column for ground truth
# - Prompt templates are in tasks/{task_name}/base_prompt.txt
# - Task categories are in tasks.py (ISSUE_TASKS, RULE_TASKS, etc.)
```

## Appendix B: Task Selection Strategy

### Recommended Starter Tasks (5-10 tasks covering all categories)

**Issue Spotting (1-2 tasks):**
- `learned_hands_torts` - Simple classification
- `corporate_lobbying` - Binary classification

**Rule Recall (1 task):**
- `international_citizenship_questions` - Tests knowledge retrieval

**Conclusion (2 tasks):**
- `hearsay` - Classic legal reasoning (binary)
- `personal_jurisdiction` - Multi-factor analysis

**Interpretation (3-4 tasks):**
- `contract_nli_confidentiality_of_agreement` - NLI task
- `contract_qa` - Question answering
- `cuad_anti-assignment` - Contract clause extraction
- `privacy_policy_qa` - Policy interpretation

**Rhetoric (1 task):**
- `definition_extraction` - Uses stemming (tests special eval)
- `overruling` - Binary classification

**Special Evaluation Cases:**
- `definition_extraction` - Stemming normalization
- `sara_numeric` - Tolerance-based evaluation
- `ssla_individual_defendants` - F1 score
- `successor_liability` - Multi-label F1

### Quick Smoke Test (5 tasks, one per category)
1. `learned_hands_torts` (Issue)
2. `international_citizenship_questions` (Rule)
3. `hearsay` (Conclusion)
4. `contract_qa` (Interpretation)
5. `overruling` (Rhetoric)

Total: 5 tasks × 5 samples = 25 API calls (~2-3 minutes, <$0.25)

## References

- LegalBench Implementation Guide: `.docs/legalbench_implementation_alpha.md`
- LegalBench Paper: https://arxiv.org/abs/2308.11462
- Cursor Rules: `.cursor/rules/*.mdc`
- Evaluation Metrics: `evaluation.py`
- Task Definitions: `tasks.py`

---

**Document Version:** 1.0
**Created:** 2025-10-28
**Last Updated:** 2025-10-28
**Owner:** Laurent Wiesel
**Reviewers:** TBD
