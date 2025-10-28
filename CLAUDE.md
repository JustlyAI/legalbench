# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LegalBench is a collaborative benchmark with 162 legal reasoning tasks for evaluating LLMs on legal text. Each task includes input-output pairs, prompt templates, task-specific evaluation metrics, and metadata.

Key resources:

- [Website](https://hazyresearch.stanford.edu/legalbench/) | [Dataset](https://huggingface.co/datasets/nguha/legalbench) | [Paper](https://arxiv.org/abs/2308.11462)

## Core Architecture

### Python Modules

- `tasks.py`: Declares task name groups (ISSUE_TASKS, RULE_TASKS, CONCLUSION_TASKS, INTERPRETATION_TASKS, RHETORIC_TASKS)
- `evaluation.py`: Implements evaluation metrics per task type (balanced accuracy default; task-specific handlers for F1, stemming, tolerance)
- `utils.py`: Contains `generate_prompts()` for filling prompt templates with TSV data

### Task Directory Structure

Each task: `tasks/{task_name}/`

- `README.md`: Task description, legal reasoning type, dataset construction
- `base_prompt.txt`: Baseline prompt template with few-shot examples
- `train.tsv`, `test.tsv`: Tab-separated data (columns: index, answer, text, metadata)
- Optional variants: `application_prompt.txt`, `rule_description_prompt.txt`, `claude_prompt.txt`

**Prompt templates**: Use `{{column_name}}` placeholders that map to TSV columns.

## Development

### Setup

```bash
pip install -e .              # Base install
pip install -e ".[dev]"       # With black, ruff, mypy, pytest
```

Requires Python ≥3.10

### Code Quality

```bash
black .         # Format (line length 100)
ruff check .    # Lint
mypy .          # Type check
```

Configuration in `pyproject.toml`.

## Task Categories & Evaluation

### Legal Reasoning Types

- **Issue** (17): Classify legal issues (learned*hands*\*)
- **Rule** (5): Answer questions about laws (rule_qa, international_citizenship_questions)
- **Conclusion** (12): Apply rules to facts (hearsay, diversity\_\*, personal_jurisdiction)
- **Interpretation** (118): Interpret contracts/policies (contract*nli*\_, cuad\_\_, maud*\*, opp115*\*)
- **Rhetoric** (10): Analyze legal rhetoric (definition*extraction, overruling, textualism_tool*\*)

### Evaluation Metrics

**Default**: Balanced accuracy with exact match (handles label imbalance)

**Task-specific handlers**:

- **F1**: `ssla_*`, `successor_liability` (multi-extraction tasks)
- **Stemming normalization**: `definition_extraction`
- **Tolerance**: `sara_numeric` (within 10% of correct answer)
- **Contains match**: `citation_prediction_open` (check if answer in generation)
- **Manual**: `rule_qa`

Check `EXACT_MATCH_BALANCED_ACC_TASKS` list in `evaluation.py` for full coverage.

### Text Normalization

The `normalize()` function in `evaluation.py`:

1. Removes punctuation
2. Strips whitespace
3. Converts to lowercase
4. Optionally applies Porter stemming (task-dependent)

## Agent-Based Evaluation (Advanced)

For extending beyond built-in scripts, consider:

- **Task loader**: Structured loading from local TSV or HuggingFace with Pydantic models
- **Instrumented agent**: Track tool calls, chain steps, token usage, retries
- **Evaluation engine**: Per-task metrics, result aggregation and persistence
- **Prompt manager**: Base prompts, few-shot variants, rule-description heuristics
- **RAG evaluator**: Retrieval indexing, augmented prompts, retrieval metrics
- **Optional API/DB layer**: Background job management

See `.docs/legalbench_implementation_alpha.md` for implementation details.

## Data Integrity Rules

1. **TSV format only**: Do not convert to JSON; prompts assume TSV columns
2. **Preserve column names**: Templates reference exact column names via `{{column_name}}`
3. **Keep ground truth unchanged**: Evaluation depends on exact answer/label values
4. **Use `generate_prompts()`**: From `utils.py` for consistent template filling
5. **Maintain prompt variants**: Keep repository-provided templates for reproducibility

## Important Notes

- Zero-shot tasks (`scalr`, `rule_qa`) have no training examples
- Task licenses vary—check individual `README.md` files
- Data format is **TSV** (tab-separated), not JSONL
- HELM configuration in `helm_prompt_settings.jsonl`
- Keep metric mappings consistent with the paper when extending evaluation

## Quick Reference

```python
# Load and evaluate a task
import pandas as pd
from utils import generate_prompts
from evaluation import evaluate

# Load data
data = pd.read_csv('tasks/hearsay/test.tsv', sep='\t')
prompt_template = open('tasks/hearsay/base_prompt.txt').read()

# Generate prompts
prompts = generate_prompts(prompt_template, data)

# Evaluate (after getting model generations)
score = evaluate('hearsay', generations, data['answer'].tolist())
```

## Navigation

- Main readme: `README.md`
- Task list: `tasks.py`
- Metrics: `evaluation.py`
- Prompt utils: `utils.py`
- Individual tasks: `tasks/{task_name}/README.md`
- Agent guide: `.docs/legalbench_implementation_alpha.md` (if available)

I work on Mac M2 silicon using cursor/visual studio code. This impacts how Docker images are created (buildx) and other consequences you must consider.

## General Guidelines

- reading and writing files should always implement encoding="utf-8"
- add informative print statements every step of the way to debug and see what the agent is doing and thinking
- have termcolor printing with cprint very step of the way to inform the user
- major variables should be all caps Variables on top of the script and not user input taking unless otherwise specified
- if there are models in the script like gpt-4o or gpt-4o-mini or o1-mini or o1-preview or claude-4-5-sonnet-20241022 do not change them as they now exist
- use pydantic
- do not delete requirements.txt unless you are sure it is not needed
- lets implement every project with seperation of concerns in mind
- always provide detailed instructions to the model considering everything carefully
- do not overcomplicate things. you should tend to simplify wherever possible
- do not mock codefiles if you suspect that they might already exist - rather, ask for the codefiles you need
- do not rewrite prompts or data classes unless specifically requested
- keep tests as simple executable and do not use mocks
- Always import python libraries at the top of the codefile
