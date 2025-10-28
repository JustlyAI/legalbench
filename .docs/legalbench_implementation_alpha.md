# LegalBench Evaluation Framework Implementation Guide
## For Claude Agent SDK-based Legal AI Agents

### Table of Contents
1. [Executive Overview](#executive-overview)
2. [LegalBench Framework Deep Dive](#legalbench-framework-deep-dive)
3. [Implementation Architecture](#implementation-architecture)
4. [Setup and Installation](#setup-and-installation)
5. [Core Implementation](#core-implementation)
6. [Agent-Specific Adaptations](#agent-specific-adaptations)
7. [Evaluation Metrics and Instrumentation](#evaluation-metrics-and-instrumentation)
8. [Production Deployment](#production-deployment)
9. [Safety and Compliance](#safety-and-compliance)
10. [Code Examples](#code-examples)
11. [References and Resources](#references-and-resources)

---

## Executive Overview

### What is LegalBench?

LegalBench is a collaboratively built benchmark suite designed to evaluate large language models (LLMs) on legal reasoning tasks. Created through a unique crowdsourcing effort within the legal community, it provides a standardized framework for assessing AI performance on law-relevant tasks.

**Key Statistics:**
- **162 tasks** covering diverse legal reasoning scenarios
- **40+ contributors** from legal practitioners, scholars, and computational legal experts
- **6 reasoning categories** providing comprehensive coverage
- **Open-source** with ongoing community contributions

### Why LegalBench for Agent Evaluation?

While originally designed for single-turn LLM evaluation, LegalBench provides an excellent foundation for evaluating more complex agent workflows:
- **Domain-specific validation**: Tests actual legal reasoning, not generic NLP
- **Taxonomized tasks**: Organized by reasoning type for targeted evaluation
- **Real-world relevance**: Tasks derived from actual legal practice
- **Extensible framework**: Can be adapted for multi-step agent workflows

---

## LegalBench Framework Deep Dive

### Task Taxonomy

LegalBench organizes tasks into six major categories of legal reasoning:

#### 1. **Issue-Spotting**
- **Definition**: Identifying legally relevant facts from scenarios
- **Example Tasks**: `relevant_facts`, `issue_spotting_contracts`
- **Agent Relevance**: Critical for document analysis pipelines

#### 2. **Rule-Recall**
- **Definition**: Recalling and identifying applicable legal rules/statutes
- **Example Tasks**: `rule_qa`, `statutory_definition`
- **Agent Relevance**: Tests knowledge retrieval and statutory understanding

#### 3. **Rule-Application**
- **Definition**: Applying legal rules to facts to produce conclusions
- **Example Tasks**: `hearsay`, `personal_jurisdiction`
- **Agent Relevance**: Core reasoning capability for advisory agents

#### 4. **Rule-Conclusion**
- **Definition**: Predicting outcomes of rule application
- **Example Tasks**: `contract_nli`, `supply_chain_disclosure`
- **Agent Relevance**: Essential for predictive legal analytics

#### 5. **Interpretation**
- **Definition**: Understanding statutory language and legal text
- **Example Tasks**: `definition_extraction`, `statutory_reasoning`
- **Agent Relevance**: Foundation for contract review and compliance

#### 6. **Rhetorical Understanding**
- **Definition**: Identifying function/strategy of legal arguments
- **Example Tasks**: `argument_function`, `rhetorical_role`
- **Agent Relevance**: Important for litigation support and brief analysis

### Task Structure

Each LegalBench task follows a consistent structure:

```json
{
  "task_name": "hearsay",
  "task_type": "classification",
  "reasoning_category": "rule-application",
  "input_format": "text",
  "output_format": "binary",
  "dataset": {
    "train": [...],
    "test": [...],
    "validation": [...]
  },
  "prompt_template": "...",
  "metrics": ["accuracy", "f1"],
  "legal_domain": "evidence",
  "difficulty": "intermediate"
}
```

### Data Sources and Formats

LegalBench data is available through multiple channels:

1. **HuggingFace Dataset**: `nguha/legalbench`
2. **GitHub Repository**: `HazyResearch/legalbench`
3. **Direct Download**: Individual task JSONs

---

## Implementation Architecture

### System Architecture for Agent Evaluation

```mermaid
graph TD
    A[Claude Agent SDK] --> B[Evaluation Orchestrator]
    B --> C[Task Loader]
    B --> D[Agent Executor]
    B --> E[Metrics Collector]
    C --> F[LegalBench Tasks]
    D --> G[Tool Instrumentation]
    D --> H[Chain Tracking]
    E --> I[Results Database]
    E --> J[Dashboard/Reports]
```

### Technology Stack

- **Core Agent**: Claude Agent SDK
- **API Framework**: FastAPI
- **Data Validation**: Pydantic, Pydantic-AI
- **Database**: PostgreSQL
- **Evaluation Engine**: Custom Python framework
- **Monitoring**: OpenTelemetry + custom instrumentation

---

## Setup and Installation

### Prerequisites

```bash
# Python 3.10+
python --version

# Virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Dependencies

```bash
# requirements.txt
claude-agent-sdk>=0.1.0
fastapi>=0.104.0
pydantic>=2.5.0
pydantic-ai>=0.1.0
psycopg2-binary>=2.9.0
sqlalchemy>=2.0.0
datasets>=2.14.0  # For HuggingFace
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0  # For metrics
pytest>=7.4.0
pytest-asyncio>=0.21.0
httpx>=0.25.0  # For async HTTP
tenacity>=8.2.0  # For retries
rich>=13.5.0  # For CLI output
```

### Installation Steps

```bash
# 1. Clone evaluation repository
git clone https://github.com/your-org/legalbench-eval.git
cd legalbench-eval

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download LegalBench data
python scripts/download_legalbench.py

# 4. Setup database
python scripts/setup_database.py

# 5. Verify installation
python -m pytest tests/test_setup.py
```

---

## Core Implementation

### 1. Task Loader Module

```python
# legalbench_eval/loaders/task_loader.py

from typing import Dict, List, Optional, Any
from datasets import load_dataset
from pydantic import BaseModel, Field, validator
from pathlib import Path
import json

class LegalBenchTask(BaseModel):
    """Pydantic model for LegalBench tasks"""
    task_name: str
    task_type: str  # classification, multi_extraction, single_extraction, rule_application, etc.
    reasoning_category: str
    input_format: str
    output_format: str
    prompt_template: str
    legal_domain: str
    difficulty: str
    metrics: List[str] = Field(default_factory=list)
    
    # Task-specific evaluation configurations
    use_balanced_accuracy: bool = False
    use_f1_score: bool = False
    requires_manual_eval: bool = False
    tolerance: Optional[float] = None  # For numeric tasks like sara_numeric
    requires_stemming: bool = False    # For definition_extraction
    
    @validator('reasoning_category')
    def validate_category(cls, v):
        valid_categories = [
            'issue-spotting', 'rule-recall', 'rule-application',
            'rule-conclusion', 'interpretation', 'rhetorical-understanding'
        ]
        if v not in valid_categories:
            raise ValueError(f"Invalid reasoning category: {v}")
        return v
    
    @validator('task_type')
    def validate_task_type(cls, v):
        valid_types = [
            'classification', 'multi_extraction', 'single_extraction',
            'rule_application', 'generation', 'numeric'
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid task type: {v}")
        return v

# Task-specific configurations based on paper's Appendix E
TASK_CONFIGURATIONS = {
    # Multi-extraction tasks (use F1 score)
    'ssla_individual_defendants': {'task_type': 'multi_extraction', 'use_f1_score': True},
    'ssla_individual_plaintiffs': {'task_type': 'multi_extraction', 'use_f1_score': True},
    'ssla_plaintiff_agencies': {'task_type': 'multi_extraction', 'use_f1_score': True},
    'successor_liability': {'task_type': 'multi_extraction', 'use_f1_score': True},
    
    # Single extraction with stemming
    'definition_extraction': {
        'task_type': 'single_extraction',
        'requires_stemming': True
    },
    
    # Numeric task with tolerance
    'sara_numeric': {
        'task_type': 'numeric',
        'tolerance': 0.1  # 10% tolerance
    },
    
    # Rule application tasks (require manual evaluation)
    'rule_qa': {
        'task_type': 'rule_application',
        'requires_manual_eval': True
    },
    
    # Classification tasks (use balanced accuracy)
    'hearsay': {'task_type': 'classification', 'use_balanced_accuracy': True},
    'personal_jurisdiction': {'task_type': 'classification', 'use_balanced_accuracy': True},
    'ucc_v_common_law': {'task_type': 'classification', 'use_balanced_accuracy': True},
    
    # Add more task-specific configurations as needed
}

class LegalBenchLoader:
    """Load and manage LegalBench tasks"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/legalbench")
        self.dataset = None
        self.tasks_metadata = {}
        
    def load_from_huggingface(self, cache_dir: Optional[Path] = None):
        """Load LegalBench dataset from HuggingFace"""
        self.dataset = load_dataset(
            "nguha/legalbench",
            cache_dir=str(cache_dir) if cache_dir else None
        )
        return self.dataset
    
    def load_task(self, task_name: str) -> Dict[str, Any]:
        """Load a specific task by name"""
        task_path = self.data_dir / f"{task_name}.json"
        
        if task_path.exists():
            with open(task_path, 'r') as f:
                return json.load(f)
        
        # Fallback to HuggingFace dataset
        if self.dataset is None:
            self.load_from_huggingface()
        
        if task_name in self.dataset:
            return self.dataset[task_name]
        
        raise ValueError(f"Task {task_name} not found")
    
    def get_tasks_by_category(self, category: str) -> List[str]:
        """Get all tasks in a specific reasoning category"""
        return [
            task for task, meta in self.tasks_metadata.items()
            if meta.get('reasoning_category') == category
        ]
    
    def load_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Load all available tasks"""
        tasks = {}
        
        # Load from local directory
        for task_file in self.data_dir.glob("*.json"):
            task_name = task_file.stem
            with open(task_file, 'r') as f:
                tasks[task_name] = json.load(f)
        
        return tasks
```

### 2. Agent Executor with Instrumentation

```python
# legalbench_eval/executors/agent_executor.py

from typing import Dict, Any, List, Optional
from claude_agent_sdk import Agent, Tool, Chain
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

class ExecutionTrace(BaseModel):
    """Track agent execution details"""
    task_name: str
    input_text: str
    expected_output: str
    actual_output: Optional[str] = None
    execution_time: float = 0.0
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    chain_steps: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tokens_used: Dict[str, int] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class InstrumentedAgent(Agent):
    """Extended Claude Agent with instrumentation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.execution_trace = None
        self.tool_interceptor = ToolInterceptor()
        
    async def execute_with_trace(
        self, 
        prompt: str, 
        task_name: str,
        expected_output: str
    ) -> ExecutionTrace:
        """Execute agent with full instrumentation"""
        
        trace = ExecutionTrace(
            task_name=task_name,
            input_text=prompt,
            expected_output=expected_output
        )
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Execute agent
            result = await self._execute_with_retry(prompt)
            trace.actual_output = result.output
            
            # Collect metrics
            trace.tool_calls = self.tool_interceptor.get_calls()
            trace.chain_steps = self._extract_chain_steps(result)
            trace.retrieval_calls = self._extract_retrieval_calls(result)
            trace.tokens_used = self._extract_token_usage(result)
            
        except Exception as e:
            trace.error = str(e)
        
        trace.execution_time = asyncio.get_event_loop().time() - start_time
        return trace
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _execute_with_retry(self, prompt: str):
        """Execute with retry logic for resilience"""
        return await self.run(prompt)
    
    def _extract_chain_steps(self, result) -> List[Dict[str, Any]]:
        """Extract chain execution steps"""
        steps = []
        if hasattr(result, 'chain_history'):
            for step in result.chain_history:
                steps.append({
                    'step_name': step.name,
                    'input': step.input,
                    'output': step.output,
                    'duration': step.duration
                })
        return steps
    
    def _extract_retrieval_calls(self, result) -> List[Dict[str, Any]]:
        """Extract retrieval/RAG calls"""
        calls = []
        for tool_call in self.tool_interceptor.get_calls():
            if tool_call['tool_name'] in ['retrieve', 'search', 'lookup']:
                calls.append({
                    'query': tool_call['input'],
                    'results': tool_call['output'],
                    'relevance_scores': tool_call.get('scores', [])
                })
        return calls
    
    def _extract_token_usage(self, result) -> Dict[str, int]:
        """Extract token usage statistics"""
        return {
            'input_tokens': getattr(result, 'input_tokens', 0),
            'output_tokens': getattr(result, 'output_tokens', 0),
            'total_tokens': getattr(result, 'total_tokens', 0),
            'reasoning_tokens': getattr(result, 'reasoning_tokens', 0)
        }

class ToolInterceptor:
    """Intercept and log tool calls"""
    
    def __init__(self):
        self.calls = []
    
    def log_call(self, tool_name: str, input_data: Any, output_data: Any):
        self.calls.append({
            'tool_name': tool_name,
            'input': input_data,
            'output': output_data,
            'timestamp': datetime.utcnow()
        })
    
    def get_calls(self) -> List[Dict[str, Any]]:
        return self.calls
    
    def clear(self):
        self.calls = []
```

### 3. Evaluation Engine

```python
# legalbench_eval/evaluation/evaluator.py

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, balanced_accuracy_score
import numpy as np
from pathlib import Path
import json
import asyncio
from datetime import datetime

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    task_name: str
    reasoning_category: str
    accuracy: float
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    # Rule-application specific metrics
    correctness_score: Optional[float] = None  # Manual eval
    analysis_score: Optional[float] = None     # Manual eval
    avg_execution_time: float = 0.0
    avg_tokens_used: int = 0
    avg_tool_calls: float = 0.0
    avg_chain_steps: float = 0.0
    error_rate: float = 0.0
    sample_size: int = 0
    timestamp: datetime = None

class LegalBenchEvaluator:
    """Main evaluation engine for LegalBench tasks"""
    
    def __init__(
        self, 
        agent: InstrumentedAgent,
        loader: LegalBenchLoader,
        results_dir: Optional[Path] = None
    ):
        self.agent = agent
        self.loader = loader
        self.results_dir = results_dir or Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
    async def evaluate_task(
        self, 
        task_name: str,
        sample_size: Optional[int] = None,
        verbose: bool = True
    ) -> EvaluationResult:
        """Evaluate agent on a specific task"""
        
        # Load task data
        task_data = self.loader.load_task(task_name)
        task_metadata = self.loader.tasks_metadata.get(task_name, {})
        
        # Get test samples
        test_samples = task_data.get('test', task_data.get('validation', []))
        if sample_size:
            test_samples = test_samples[:sample_size]
        
        # Run evaluations
        traces = []
        predictions = []
        ground_truths = []
        
        for sample in test_samples:
            # Build prompt from template
            prompt = self._build_prompt(
                task_data['prompt_template'],
                sample['input']
            )
            
            # Execute with trace
            trace = await self.agent.execute_with_trace(
                prompt=prompt,
                task_name=task_name,
                expected_output=sample['output']
            )
            
            traces.append(trace)
            
            if trace.actual_output:
                predictions.append(self._normalize_output(trace.actual_output))
                ground_truths.append(self._normalize_output(sample['output']))
            
            if verbose:
                self._print_progress(len(traces), len(test_samples))
        
        # Calculate metrics
        result = self._calculate_metrics(
            task_name=task_name,
            reasoning_category=task_metadata.get('reasoning_category', 'unknown'),
            predictions=predictions,
            ground_truths=ground_truths,
            traces=traces
        )
        
        # Save results
        self._save_results(result, traces)
        
        return result
    
    async def evaluate_category(
        self, 
        category: str,
        sample_size_per_task: Optional[int] = None
    ) -> List[EvaluationResult]:
        """Evaluate all tasks in a reasoning category"""
        
        tasks = self.loader.get_tasks_by_category(category)
        results = []
        
        for task in tasks:
            result = await self.evaluate_task(
                task_name=task,
                sample_size=sample_size_per_task
            )
            results.append(result)
        
        return results
    
    async def evaluate_all(
        self,
        sample_size_per_task: Optional[int] = None,
        categories: Optional[List[str]] = None
    ) -> Dict[str, List[EvaluationResult]]:
        """Evaluate all tasks or specified categories"""
        
        if categories is None:
            categories = [
                'issue-spotting', 'rule-recall', 'rule-application',
                'rule-conclusion', 'interpretation', 'rhetorical-understanding'
            ]
        
        all_results = {}
        
        for category in categories:
            results = await self.evaluate_category(
                category=category,
                sample_size_per_task=sample_size_per_task
            )
            all_results[category] = results
        
        return all_results
    
    def _build_prompt(self, template: str, input_data: Any) -> str:
        """Build prompt from template and input"""
        if isinstance(input_data, dict):
            return template.format(**input_data)
        else:
            return template.format(input=input_data)
    
    def _normalize_output(self, output: str) -> str:
        """Normalize output for comparison"""
        return output.strip().lower()
    
    def _calculate_metrics(
        self,
        task_name: str,
        reasoning_category: str,
        predictions: List[str],
        ground_truths: List[str],
        traces: List[ExecutionTrace],
        task_metadata: Dict[str, Any]
    ) -> EvaluationResult:
        """Calculate comprehensive evaluation metrics per LegalBench paper"""
        
        task_type = task_metadata.get('task_type', 'classification')
        
        # Calculate metrics based on task type (following paper's Appendix E)
        if task_type == 'classification':
            # Use BALANCED ACCURACY for classification tasks (crucial for label imbalance)
            from sklearn.metrics import balanced_accuracy_score
            accuracy = balanced_accuracy_score(ground_truths, predictions) if predictions else 0.0
            
            # Also calculate F1 for reference
            precision, recall, f1, _ = precision_recall_fscore_support(
                ground_truths, predictions, average='weighted', zero_division=0
            )
            
        elif task_type == 'multi_extraction':
            # F1 score for tasks with multiple outputs (ssla_*, successor_liability)
            from sklearn.metrics import f1_score
            f1 = f1_score(ground_truths, predictions, average='micro')
            accuracy = f1  # Use F1 as primary metric
            precision = None
            recall = None
            
        elif task_type == 'single_extraction':
            # Accuracy with normalization for single-output extraction
            if task_name == 'definition_extraction':
                # Apply stemming normalization
                from nltk.stem import PorterStemmer
                stemmer = PorterStemmer()
                normalized_preds = [stemmer.stem(p.lower()) for p in predictions]
                normalized_truths = [stemmer.stem(t.lower()) for t in ground_truths]
                accuracy = accuracy_score(normalized_truths, normalized_preds)
            elif task_name == 'sara_numeric':
                # Accuracy within 10% tolerance
                correct = sum(
                    1 for p, t in zip(predictions, ground_truths)
                    if self._within_tolerance(float(p), float(t), 0.1)
                )
                accuracy = correct / len(predictions) if predictions else 0.0
            else:
                accuracy = accuracy_score(ground_truths, predictions) if predictions else 0.0
            
            f1 = None
            precision = None
            recall = None
            
        elif task_type == 'rule_application':
            # Manual evaluation required - return placeholder metrics
            accuracy = None  # Will be filled by manual evaluation
            f1 = None
            precision = None
            recall = None
            
        else:
            # Default to standard accuracy
            accuracy = accuracy_score(ground_truths, predictions) if predictions else 0.0
            f1 = None
            precision = None
            recall = None
        
        # Execution metrics
        execution_times = [t.execution_time for t in traces if t.execution_time]
        token_counts = [t.tokens_used.get('total_tokens', 0) for t in traces]
        tool_call_counts = [len(t.tool_calls) for t in traces]
        chain_step_counts = [len(t.chain_steps) for t in traces]
        error_count = sum(1 for t in traces if t.error is not None)
        
        return EvaluationResult(
            task_name=task_name,
            reasoning_category=reasoning_category,
            accuracy=accuracy,
            f1=f1,
            precision=precision,
            recall=recall,
            avg_execution_time=np.mean(execution_times) if execution_times else 0.0,
            avg_tokens_used=int(np.mean(token_counts)) if token_counts else 0,
            avg_tool_calls=np.mean(tool_call_counts) if tool_call_counts else 0.0,
            avg_chain_steps=np.mean(chain_step_counts) if chain_step_counts else 0.0,
            error_rate=error_count / len(traces) if traces else 0.0,
            sample_size=len(traces),
            timestamp=datetime.utcnow()
        )
    
    def _within_tolerance(self, predicted: float, actual: float, tolerance: float) -> bool:
        """Check if prediction is within tolerance of actual value"""
        if actual == 0:
            return predicted == 0
        relative_error = abs(predicted - actual) / abs(actual)
        return relative_error <= tolerance
    
    def _save_results(self, result: EvaluationResult, traces: List[ExecutionTrace]):
        """Save evaluation results and traces"""
        
        # Save summary
        summary_path = self.results_dir / f"{result.task_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(result.__dict__, f, indent=2, default=str)
        
        # Save detailed traces
        traces_path = self.results_dir / f"{result.task_name}_traces.json"
        with open(traces_path, 'w') as f:
            json.dump(
                [trace.dict() for trace in traces],
                f, indent=2, default=str
            )
    
    def _print_progress(self, current: int, total: int):
        """Print evaluation progress"""
        progress = current / total * 100
        print(f"Progress: {current}/{total} ({progress:.1f}%)", end='\r')
```

---

## Agent-Specific Adaptations

### Manual Evaluation for Rule-Application Tasks

Per the LegalBench paper, rule-application tasks require manual evaluation on two dimensions:

```python
# legalbench_eval/manual_evaluation/rule_application_evaluator.py

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum

class ErrorType(Enum):
    """Types of errors in rule-application explanations"""
    MISSTATED_FACTS = "misstated_facts"
    MISSTATED_RULE = "misstated_rule" 
    WRONG_OUTCOME = "wrong_outcome"
    LOGIC_ERROR = "logic_error"
    ARITHMETIC_ERROR = "arithmetic_error"
    NO_ERROR = "no_error"

class RuleApplicationEvaluation(BaseModel):
    """Manual evaluation for rule-application tasks"""
    task_name: str
    sample_id: str
    generated_explanation: str
    
    # Correctness dimension (binary)
    correctness: bool = Field(
        description="Is the explanation free of errors?"
    )
    errors_found: List[ErrorType] = Field(
        default_factory=list,
        description="List of errors in the explanation"
    )
    
    # Analysis dimension (binary)
    analysis: bool = Field(
        description="Does explanation contain actual reasoning/inferences?"
    )
    required_inferences_present: List[str] = Field(
        default_factory=list,
        description="Which required inferences were present"
    )
    missing_inferences: List[str] = Field(
        default_factory=list,
        description="Which required inferences were missing"
    )
    
    # Additional notes
    evaluator_notes: Optional[str] = None
    
class ManualEvaluationInterface:
    """Interface for manual evaluation of rule-application tasks"""
    
    def __init__(self, answer_guides_dir: Path):
        self.answer_guides = self._load_answer_guides(answer_guides_dir)
        
    def _load_answer_guides(self, guides_dir: Path) -> Dict[str, Dict]:
        """Load task-specific answer guides"""
        guides = {}
        for guide_file in guides_dir.glob("*.json"):
            task_name = guide_file.stem
            with open(guide_file, 'r') as f:
                guides[task_name] = json.load(f)
        return guides
    
    def evaluate_explanation(
        self,
        task_name: str,
        sample_id: str,
        generated_explanation: str,
        facts: str,
        rule: str,
        expected_outcome: str
    ) -> RuleApplicationEvaluation:
        """Evaluate a single rule-application explanation"""
        
        # Get answer guide for this task
        guide = self.answer_guides.get(task_name, {})
        required_inferences = guide.get('required_inferences', [])
        common_errors = guide.get('common_errors', [])
        
        evaluation = RuleApplicationEvaluation(
            task_name=task_name,
            sample_id=sample_id,
            generated_explanation=generated_explanation
        )
        
        # Check correctness (absence of errors)
        errors = self._check_for_errors(
            generated_explanation,
            facts,
            rule,
            expected_outcome,
            common_errors
        )
        
        evaluation.errors_found = errors
        evaluation.correctness = (len(errors) == 0 or errors == [ErrorType.NO_ERROR])
        
        # Check analysis (presence of required inferences)
        present, missing = self._check_inferences(
            generated_explanation,
            required_inferences
        )
        
        evaluation.required_inferences_present = present
        evaluation.missing_inferences = missing
        
        # Analysis is True only if explanation is correct AND contains inferences
        # (not just restatements of facts/rule/conclusion)
        evaluation.analysis = (
            evaluation.correctness and 
            len(present) > 0 and
            len(missing) == 0
        )
        
        return evaluation
    
    def _check_for_errors(
        self,
        explanation: str,
        facts: str,
        rule: str,
        expected_outcome: str,
        common_errors: List[Dict]
    ) -> List[ErrorType]:
        """Check for various types of errors in the explanation"""
        
        errors = []
        
        # This would need sophisticated NLP or manual review
        # For now, placeholder for manual checking interface
        # In production, this would present the explanation to a human evaluator
        # along with the facts, rule, and expected outcome
        
        return errors
    
    def _check_inferences(
        self,
        explanation: str,
        required_inferences: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Check which required inferences are present"""
        
        present = []
        missing = []
        
        # This would need sophisticated analysis or manual review
        # to determine if the explanation contains the required reasoning steps
        
        for inference in required_inferences:
            # Placeholder for checking if inference is present
            # In production, this would be a manual check or sophisticated NLP
            if self._contains_inference(explanation, inference):
                present.append(inference)
            else:
                missing.append(inference)
        
        return present, missing
    
    def _contains_inference(self, explanation: str, inference: str) -> bool:
        """Check if explanation contains a specific inference"""
        # Placeholder - would need implementation based on answer guides
        return False

class ManualEvaluationUI:
    """Web UI for manual evaluation (FastAPI endpoints)"""
    
    @app.post("/manual_evaluation/submit")
    async def submit_evaluation(
        task_name: str,
        sample_id: str,
        correctness: bool,
        analysis: bool,
        errors: List[ErrorType],
        notes: Optional[str] = None
    ):
        """Submit manual evaluation for a sample"""
        
        # Store in database
        evaluation = RuleApplicationEvaluation(
            task_name=task_name,
            sample_id=sample_id,
            correctness=correctness,
            analysis=analysis,
            errors_found=errors,
            evaluator_notes=notes
        )
        
        # Save to database
        db.save_manual_evaluation(evaluation)
        
        return {"status": "saved", "sample_id": sample_id}
    
    @app.get("/manual_evaluation/next")
    async def get_next_for_evaluation(task_name: str):
        """Get next sample requiring manual evaluation"""
        
        # Fetch from database
        next_sample = db.get_unevaluated_sample(task_name)
        
        if next_sample:
            # Load answer guide
            guide = answer_guides.get(task_name)
            
            return {
                "sample": next_sample,
                "answer_guide": guide,
                "required_inferences": guide.get("required_inferences", []),
                "common_errors": guide.get("common_errors", [])
            }
        
        return {"message": "No samples remaining for evaluation"}
```

### Prompt Configuration and Base Prompts

Following the paper's methodology, we need specific prompt configurations:

```python
# legalbench_eval/prompts/prompt_manager.py

from typing import Dict, List, Optional, Any
from pathlib import Path
import json

class PromptConfiguration(BaseModel):
    """Configuration for LegalBench prompts"""
    temperature: float = 0.0  # Paper uses temp=0.0 for reproducibility
    max_tokens: int = 2048
    use_base_prompt: bool = True
    num_few_shot_examples: int = 5  # Paper tests 0-8 examples
    example_selection_strategy: str = "provided"  # Use provided examples

class LegalBenchPromptManager:
    """Manage prompts and few-shot examples for LegalBench tasks"""
    
    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir
        self.base_prompts = self._load_base_prompts()
        self.few_shot_examples = self._load_few_shot_examples()
        
    def _load_base_prompts(self) -> Dict[str, str]:
        """Load base prompts provided by LegalBench"""
        base_prompts = {}
        prompts_file = self.prompts_dir / "base_prompts.json"
        
        if prompts_file.exists():
            with open(prompts_file, 'r') as f:
                base_prompts = json.load(f)
        
        return base_prompts
    
    def _load_few_shot_examples(self) -> Dict[str, List[Dict]]:
        """Load few-shot examples for each task"""
        examples = {}
        examples_file = self.prompts_dir / "few_shot_examples.json"
        
        if examples_file.exists():
            with open(examples_file, 'r') as f:
                examples = json.load(f)
        
        return examples
    
    def build_prompt(
        self,
        task_name: str,
        input_text: str,
        config: PromptConfiguration
    ) -> str:
        """Build prompt with base instructions and few-shot examples"""
        
        # Get base prompt (instructions)
        if config.use_base_prompt and task_name in self.base_prompts:
            base_prompt = self.base_prompts[task_name]
        else:
            base_prompt = "Answer the following legal question:"
        
        # Get few-shot examples
        examples = []
        if config.num_few_shot_examples > 0 and task_name in self.few_shot_examples:
            available_examples = self.few_shot_examples[task_name]
            examples = available_examples[:config.num_few_shot_examples]
        
        # Build full prompt
        prompt_parts = [base_prompt]
        
        # Add few-shot examples
        if examples:
            prompt_parts.append("\nExamples:")
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"Input: {example['input']}")
                prompt_parts.append(f"Output: {example['output']}")
        
        # Add actual input
        prompt_parts.append("\nNow answer for this input:")
        prompt_parts.append(f"Input: {input_text}")
        prompt_parts.append("Output:")
        
        return "\n".join(prompt_parts)
    
    def get_prompt_variants(
        self,
        task_name: str,
        input_text: str
    ) -> List[Tuple[str, PromptConfiguration]]:
        """Generate different prompt variants for testing"""
        
        variants = []
        
        # Test different numbers of few-shot examples (0-8 as in paper)
        for n_examples in [0, 2, 5, 8]:
            config = PromptConfiguration(
                num_few_shot_examples=n_examples,
                temperature=0.0
            )
            prompt = self.build_prompt(task_name, input_text, config)
            variants.append((prompt, config))
        
        return variants
    
    def should_use_rule_description(self, task_name: str) -> bool:
        """Determine if rule description should be included (per Figure 1)"""
        # Tasks that benefit from rule description
        rule_beneficial_tasks = [
            'rule_conclusion_tasks',  # List from paper
            'ucc_v_common_law',
            # Add more based on Figure 1 results
        ]
        return task_name in rule_beneficial_tasks
```

### Multi-Step Reasoning Instrumentation

For agents that perform multi-step reasoning, we need additional metrics:

```python
# legalbench_eval/metrics/agent_metrics.py

class AgentSpecificMetrics:
    """Metrics specific to agent workflows"""
    
    @staticmethod
    def calculate_process_rate(traces: List[ExecutionTrace]) -> float:
        """
        Calculate process rate: percentage of correct intermediate steps
        Similar to LegalAgentBench methodology
        """
        correct_steps = 0
        total_steps = 0
        
        for trace in traces:
            for step in trace.chain_steps:
                total_steps += 1
                if AgentSpecificMetrics._is_step_correct(step, trace.expected_output):
                    correct_steps += 1
        
        return correct_steps / total_steps if total_steps > 0 else 0.0
    
    @staticmethod
    def calculate_retrieval_precision(traces: List[ExecutionTrace]) -> float:
        """Calculate precision of retrieval calls"""
        relevant_retrievals = 0
        total_retrievals = 0
        
        for trace in traces:
            for retrieval in trace.retrieval_calls:
                total_retrievals += 1
                if AgentSpecificMetrics._is_retrieval_relevant(
                    retrieval, trace.expected_output
                ):
                    relevant_retrievals += 1
        
        return relevant_retrievals / total_retrievals if total_retrievals > 0 else 0.0
    
    @staticmethod
    def calculate_tool_efficiency(traces: List[ExecutionTrace]) -> Dict[str, float]:
        """Calculate tool usage efficiency metrics"""
        tool_usage = {}
        
        for trace in traces:
            for tool_call in trace.tool_calls:
                tool_name = tool_call['tool_name']
                if tool_name not in tool_usage:
                    tool_usage[tool_name] = {'count': 0, 'successful': 0}
                
                tool_usage[tool_name]['count'] += 1
                if tool_call.get('output') is not None:
                    tool_usage[tool_name]['successful'] += 1
        
        efficiency = {}
        for tool, stats in tool_usage.items():
            if stats['count'] > 0:
                efficiency[tool] = stats['successful'] / stats['count']
        
        return efficiency
    
    @staticmethod
    def _is_step_correct(step: Dict, expected_output: str) -> bool:
        """Determine if intermediate step is correct"""
        # Implement step validation logic
        # This could involve checking against annotated correct steps
        # or using heuristics/patterns
        pass
    
    @staticmethod
    def _is_retrieval_relevant(retrieval: Dict, expected_output: str) -> bool:
        """Determine if retrieval is relevant"""
        # Implement retrieval relevance checking
        # Could use embedding similarity or keyword overlap
        pass
```

### RAG-Specific Adaptations

For retrieval-augmented generation workflows:

```python
# legalbench_eval/rag/rag_evaluator.py

class LegalBenchRAGEvaluator:
    """Evaluation for RAG-enhanced legal agents"""
    
    def __init__(self, agent: InstrumentedAgent, retriever: Any):
        self.agent = agent
        self.retriever = retriever
    
    async def evaluate_with_retrieval(
        self,
        task_name: str,
        document_corpus: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Evaluate task with retrieval component"""
        
        # Index documents
        self.retriever.index_documents(document_corpus)
        
        # Load task
        task_data = self.loader.load_task(task_name)
        
        results = []
        for sample in task_data['test']:
            # Retrieve relevant documents
            retrieved_docs = await self.retriever.retrieve(
                query=sample['input'],
                k=5
            )
            
            # Build augmented prompt
            augmented_prompt = self._build_rag_prompt(
                original_prompt=sample['input'],
                retrieved_docs=retrieved_docs
            )
            
            # Execute agent
            result = await self.agent.execute_with_trace(
                prompt=augmented_prompt,
                task_name=task_name,
                expected_output=sample['output']
            )
            
            # Add retrieval metrics
            result.retrieval_metrics = {
                'num_docs_retrieved': len(retrieved_docs),
                'avg_relevance_score': np.mean([d['score'] for d in retrieved_docs]),
                'retrieval_latency': retrieved_docs[0].get('latency', 0)
            }
            
            results.append(result)
        
        return self._aggregate_rag_results(results)
    
    def _build_rag_prompt(
        self, 
        original_prompt: str, 
        retrieved_docs: List[Dict]
    ) -> str:
        """Build prompt with retrieved context"""
        context = "\n\n".join([
            f"Document {i+1}: {doc['content']}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        return f"""Based on the following legal documents:

{context}

Please answer the following question:
{original_prompt}"""
    
    def _aggregate_rag_results(self, results: List[Any]) -> Dict[str, Any]:
        """Aggregate RAG-specific results"""
        return {
            'accuracy': np.mean([r.is_correct for r in results]),
            'avg_docs_used': np.mean([r.retrieval_metrics['num_docs_retrieved'] for r in results]),
            'avg_relevance': np.mean([r.retrieval_metrics['avg_relevance_score'] for r in results]),
            'retrieval_impact': self._calculate_retrieval_impact(results)
        }
    
    def _calculate_retrieval_impact(self, results: List[Any]) -> float:
        """Calculate how much retrieval improves accuracy"""
        # Compare with/without retrieval if baseline available
        pass
```

---

## Evaluation Metrics and Instrumentation

### Comprehensive Metrics Framework

```python
# legalbench_eval/metrics/comprehensive_metrics.py

from typing import Dict, List, Any
import pandas as pd
from dataclasses import dataclass

@dataclass
class ComprehensiveMetrics:
    """Complete metrics for legal agent evaluation"""
    
    # Core Accuracy Metrics
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    
    # Agent Behavior Metrics
    avg_chain_length: float
    avg_tool_calls: float
    tool_usage_distribution: Dict[str, int]
    avg_reasoning_steps: int
    
    # Retrieval Metrics (if applicable)
    retrieval_precision: float
    retrieval_recall: float
    avg_documents_retrieved: float
    retrieval_latency: float
    
    # Efficiency Metrics
    avg_execution_time: float
    avg_tokens_used: int
    cost_per_query: float
    throughput_qps: float
    
    # Reliability Metrics
    error_rate: float
    timeout_rate: float
    retry_rate: float
    consistency_score: float  # Same input, same output
    
    # Legal-Specific Metrics
    citation_accuracy: float  # Correct legal citations
    statute_reference_precision: float
    case_law_relevance: float
    legal_terminology_accuracy: float
    
    # Process Quality Metrics
    process_rate: float  # Correct intermediate steps
    reasoning_coherence: float
    explanation_quality: float

class MetricsCalculator:
    """Calculate comprehensive metrics from evaluation traces"""
    
    def calculate_all_metrics(
        self,
        traces: List[ExecutionTrace],
        task_metadata: Dict[str, Any]
    ) -> ComprehensiveMetrics:
        """Calculate all metrics from execution traces"""
        
        # Basic accuracy metrics
        predictions = [t.actual_output for t in traces if t.actual_output]
        ground_truths = [t.expected_output for t in traces]
        
        accuracy_metrics = self._calculate_accuracy_metrics(predictions, ground_truths)
        
        # Agent behavior metrics
        behavior_metrics = self._calculate_behavior_metrics(traces)
        
        # Retrieval metrics
        retrieval_metrics = self._calculate_retrieval_metrics(traces)
        
        # Efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(traces)
        
        # Reliability metrics
        reliability_metrics = self._calculate_reliability_metrics(traces)
        
        # Legal-specific metrics
        legal_metrics = self._calculate_legal_metrics(traces, task_metadata)
        
        # Process quality metrics
        process_metrics = self._calculate_process_metrics(traces)
        
        return ComprehensiveMetrics(
            **accuracy_metrics,
            **behavior_metrics,
            **retrieval_metrics,
            **efficiency_metrics,
            **reliability_metrics,
            **legal_metrics,
            **process_metrics
        )
    
    def generate_report(
        self,
        metrics: ComprehensiveMetrics,
        output_format: str = "markdown"
    ) -> str:
        """Generate evaluation report"""
        
        if output_format == "markdown":
            return self._generate_markdown_report(metrics)
        elif output_format == "html":
            return self._generate_html_report(metrics)
        else:
            return self._generate_json_report(metrics)
    
    def _generate_markdown_report(self, metrics: ComprehensiveMetrics) -> str:
        """Generate markdown format report"""
        
        report = f"""# LegalBench Evaluation Report

## Core Performance
- **Accuracy**: {metrics.accuracy:.2%}
- **F1 Score**: {metrics.f1_score:.2%}
- **Precision**: {metrics.precision:.2%}
- **Recall**: {metrics.recall:.2%}

## Agent Behavior
- **Average Chain Length**: {metrics.avg_chain_length:.1f} steps
- **Average Tool Calls**: {metrics.avg_tool_calls:.1f}
- **Average Reasoning Steps**: {metrics.avg_reasoning_steps}

## Efficiency
- **Average Execution Time**: {metrics.avg_execution_time:.2f}s
- **Average Tokens Used**: {metrics.avg_tokens_used:,}
- **Cost per Query**: ${metrics.cost_per_query:.4f}
- **Throughput**: {metrics.throughput_qps:.2f} QPS

## Reliability
- **Error Rate**: {metrics.error_rate:.2%}
- **Timeout Rate**: {metrics.timeout_rate:.2%}
- **Consistency Score**: {metrics.consistency_score:.2%}

## Legal-Specific Performance
- **Citation Accuracy**: {metrics.citation_accuracy:.2%}
- **Statute Reference Precision**: {metrics.statute_reference_precision:.2%}
- **Legal Terminology Accuracy**: {metrics.legal_terminology_accuracy:.2%}

## Process Quality
- **Process Rate**: {metrics.process_rate:.2%}
- **Reasoning Coherence**: {metrics.reasoning_coherence:.2%}
- **Explanation Quality**: {metrics.explanation_quality:.2%}
"""
        return report
```

---

## Production Deployment

### API Integration

```python
# legalbench_eval/api/evaluation_api.py

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import uuid

app = FastAPI(title="LegalBench Evaluation API")

class EvaluationRequest(BaseModel):
    """Request model for evaluation"""
    task_names: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    sample_size: Optional[int] = None
    agent_config: Optional[Dict[str, Any]] = None

class EvaluationResponse(BaseModel):
    """Response model for evaluation"""
    evaluation_id: str
    status: str
    message: str

class EvaluationManager:
    """Manage evaluation jobs"""
    
    def __init__(self):
        self.jobs = {}
        self.results_store = {}
    
    async def start_evaluation(
        self,
        request: EvaluationRequest,
        background_tasks: BackgroundTasks
    ) -> str:
        """Start evaluation job"""
        
        eval_id = str(uuid.uuid4())
        
        # Configure agent
        agent = self._configure_agent(request.agent_config)
        
        # Setup evaluator
        loader = LegalBenchLoader()
        evaluator = LegalBenchEvaluator(agent, loader)
        
        # Start background task
        background_tasks.add_task(
            self._run_evaluation,
            eval_id,
            evaluator,
            request
        )
        
        self.jobs[eval_id] = {
            'status': 'running',
            'started_at': datetime.utcnow(),
            'request': request
        }
        
        return eval_id
    
    async def _run_evaluation(
        self,
        eval_id: str,
        evaluator: LegalBenchEvaluator,
        request: EvaluationRequest
    ):
        """Run evaluation in background"""
        
        try:
            if request.task_names:
                # Evaluate specific tasks
                results = {}
                for task in request.task_names:
                    result = await evaluator.evaluate_task(
                        task_name=task,
                        sample_size=request.sample_size
                    )
                    results[task] = result
            
            elif request.categories:
                # Evaluate by categories
                results = await evaluator.evaluate_all(
                    categories=request.categories,
                    sample_size_per_task=request.sample_size
                )
            
            else:
                # Evaluate all
                results = await evaluator.evaluate_all(
                    sample_size_per_task=request.sample_size
                )
            
            # Store results
            self.results_store[eval_id] = results
            self.jobs[eval_id]['status'] = 'completed'
            
        except Exception as e:
            self.jobs[eval_id]['status'] = 'failed'
            self.jobs[eval_id]['error'] = str(e)
    
    def get_status(self, eval_id: str) -> Dict[str, Any]:
        """Get evaluation job status"""
        if eval_id not in self.jobs:
            raise ValueError(f"Evaluation {eval_id} not found")
        
        return self.jobs[eval_id]
    
    def get_results(self, eval_id: str) -> Any:
        """Get evaluation results"""
        if eval_id not in self.results_store:
            raise ValueError(f"Results for {eval_id} not available")
        
        return self.results_store[eval_id]

# Initialize manager
evaluation_manager = EvaluationManager()

@app.post("/evaluate", response_model=EvaluationResponse)
async def start_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks
):
    """Start new evaluation"""
    
    eval_id = await evaluation_manager.start_evaluation(request, background_tasks)
    
    return EvaluationResponse(
        evaluation_id=eval_id,
        status="started",
        message="Evaluation started successfully"
    )

@app.get("/evaluate/{eval_id}/status")
async def get_evaluation_status(eval_id: str):
    """Get evaluation status"""
    
    try:
        status = evaluation_manager.get_status(eval_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/evaluate/{eval_id}/results")
async def get_evaluation_results(eval_id: str):
    """Get evaluation results"""
    
    try:
        results = evaluation_manager.get_results(eval_id)
        return results
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

### Database Schema

```python
# legalbench_eval/database/models.py

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class EvaluationRun(Base):
    """Store evaluation run metadata"""
    __tablename__ = 'evaluation_runs'
    
    id = Column(String, primary_key=True)
    agent_version = Column(String)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    status = Column(String)
    config = Column(JSON)
    
class TaskResult(Base):
    """Store task-level results"""
    __tablename__ = 'task_results'
    
    id = Column(Integer, primary_key=True)
    evaluation_run_id = Column(String)
    task_name = Column(String)
    reasoning_category = Column(String)
    accuracy = Column(Float)
    f1_score = Column(Float)
    execution_time = Column(Float)
    tokens_used = Column(Integer)
    error_rate = Column(Float)
    metrics = Column(JSON)  # Store all metrics
    
class ExecutionTrace(Base):
    """Store detailed execution traces"""
    __tablename__ = 'execution_traces'
    
    id = Column(Integer, primary_key=True)
    task_result_id = Column(Integer)
    input_text = Column(Text)
    expected_output = Column(Text)
    actual_output = Column(Text)
    tool_calls = Column(JSON)
    chain_steps = Column(JSON)
    error = Column(Text)
    
# Database connection
DATABASE_URL = "postgresql://user:password@localhost/legalbench_eval"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)
```

---

## Safety and Compliance

### Legal Domain Considerations

```python
# legalbench_eval/safety/legal_safety.py

class LegalSafetyChecker:
    """Safety checks for legal domain evaluation"""
    
    def __init__(self):
        self.hallucination_detector = HallucinationDetector()
        self.citation_validator = CitationValidator()
        
    def validate_output(
        self,
        output: str,
        task_type: str,
        legal_domain: str
    ) -> Dict[str, Any]:
        """Validate agent output for legal safety"""
        
        validations = {
            'has_hallucination': False,
            'invalid_citations': [],
            'confidence_score': 0.0,
            'warnings': []
        }
        
        # Check for hallucinated cases/statutes
        hallucination_check = self.hallucination_detector.check(output)
        validations['has_hallucination'] = hallucination_check['detected']
        
        # Validate legal citations
        citations = self.citation_validator.extract_citations(output)
        for citation in citations:
            if not self.citation_validator.is_valid(citation):
                validations['invalid_citations'].append(citation)
        
        # Add domain-specific warnings
        if legal_domain in ['criminal_law', 'family_law', 'immigration']:
            validations['warnings'].append(
                "High-stakes domain - human review required"
            )
        
        # Calculate confidence
        validations['confidence_score'] = self._calculate_confidence(
            output, task_type, validations
        )
        
        return validations
    
    def _calculate_confidence(
        self,
        output: str,
        task_type: str,
        validations: Dict
    ) -> float:
        """Calculate confidence score for output"""
        
        confidence = 1.0
        
        if validations['has_hallucination']:
            confidence -= 0.5
        
        if validations['invalid_citations']:
            confidence -= 0.1 * len(validations['invalid_citations'])
        
        # Adjust by task complexity
        complexity_factors = {
            'classification': 1.0,
            'extraction': 0.9,
            'generation': 0.7,
            'reasoning': 0.6
        }
        
        confidence *= complexity_factors.get(task_type, 0.5)
        
        return max(0.0, min(1.0, confidence))

class HallucinationDetector:
    """Detect hallucinated legal content"""
    
    def __init__(self):
        # Load known legal entities
        self.valid_cases = self._load_valid_cases()
        self.valid_statutes = self._load_valid_statutes()
    
    def check(self, text: str) -> Dict[str, Any]:
        """Check for hallucinations"""
        
        # Extract legal references
        case_refs = self._extract_case_references(text)
        statute_refs = self._extract_statute_references(text)
        
        hallucinated_cases = [
            ref for ref in case_refs 
            if not self._is_valid_case(ref)
        ]
        
        hallucinated_statutes = [
            ref for ref in statute_refs
            if not self._is_valid_statute(ref)
        ]
        
        return {
            'detected': bool(hallucinated_cases or hallucinated_statutes),
            'hallucinated_cases': hallucinated_cases,
            'hallucinated_statutes': hallucinated_statutes
        }
```

### Deployment Safety Checklist

```python
# legalbench_eval/safety/deployment_checklist.py

class DeploymentSafetyChecklist:
    """Pre-deployment safety checks"""
    
    @staticmethod
    def run_safety_checks(
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Run comprehensive safety checks"""
        
        checks = {
            'accuracy_threshold': False,
            'no_critical_errors': False,
            'citation_validation': False,
            'consistency_check': False,
            'bias_detection': False,
            'hallucination_rate': False,
            'human_review_completed': False
        }
        
        # Check accuracy thresholds
        min_accuracy = {
            'issue-spotting': 0.85,
            'rule-recall': 0.90,
            'rule-application': 0.80,
            'rule-conclusion': 0.75,
            'interpretation': 0.85,
            'rhetorical-understanding': 0.70
        }
        
        for category, results in evaluation_results.items():
            required_accuracy = min_accuracy.get(category, 0.80)
            actual_accuracy = np.mean([r.accuracy for r in results])
            
            if actual_accuracy >= required_accuracy:
                checks['accuracy_threshold'] = True
        
        # Check error rates
        error_rates = [r.error_rate for r in results]
        checks['no_critical_errors'] = max(error_rates) < 0.05
        
        # Additional checks...
        
        return checks
```

---

## Code Examples

### Complete Evaluation Script

```python
# scripts/run_evaluation.py

import asyncio
import argparse
from pathlib import Path
from legalbench_eval import (
    LegalBenchLoader,
    LegalBenchEvaluator,
    InstrumentedAgent,
    MetricsCalculator,
    LegalSafetyChecker
)
from claude_agent_sdk import Agent, Tool
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description='Run LegalBench evaluation')
    parser.add_argument('--tasks', nargs='+', help='Specific tasks to evaluate')
    parser.add_argument('--categories', nargs='+', help='Categories to evaluate')
    parser.add_argument('--sample-size', type=int, default=100, help='Samples per task')
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    
    # Initialize components
    logger.info("Initializing evaluation components...")
    
    # Create instrumented agent
    agent = InstrumentedAgent(
        api_key="your-api-key",
        model="claude-3-opus-20240229",
        tools=[
            Tool(name="legal_search", ...),
            Tool(name="statute_lookup", ...),
            Tool(name="case_retrieval", ...)
        ]
    )
    
    # Load LegalBench
    loader = LegalBenchLoader()
    loader.load_from_huggingface()
    
    # Create evaluator
    evaluator = LegalBenchEvaluator(
        agent=agent,
        loader=loader,
        results_dir=args.output_dir
    )
    
    # Run evaluation
    logger.info("Starting evaluation...")
    
    if args.tasks:
        # Evaluate specific tasks
        for task in args.tasks:
            logger.info(f"Evaluating task: {task}")
            result = await evaluator.evaluate_task(
                task_name=task,
                sample_size=args.sample_size
            )
            logger.info(f"Task {task} accuracy: {result.accuracy:.2%}")
    
    elif args.categories:
        # Evaluate categories
        results = await evaluator.evaluate_all(
            categories=args.categories,
            sample_size_per_task=args.sample_size
        )
        
        # Generate report
        calculator = MetricsCalculator()
        for category, category_results in results.items():
            logger.info(f"\nCategory: {category}")
            for result in category_results:
                logger.info(f"  {result.task_name}: {result.accuracy:.2%}")
    
    else:
        # Evaluate all
        results = await evaluator.evaluate_all(
            sample_size_per_task=args.sample_size
        )
    
    # Run safety checks
    logger.info("\nRunning safety checks...")
    safety_checker = LegalSafetyChecker()
    
    # Generate final report
    logger.info(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Testing Framework

```python
# tests/test_evaluation.py

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from legalbench_eval import InstrumentedAgent, LegalBenchEvaluator

@pytest.fixture
def mock_agent():
    """Create mock agent for testing"""
    agent = Mock(spec=InstrumentedAgent)
    agent.execute_with_trace = AsyncMock()
    return agent

@pytest.fixture
def sample_task():
    """Sample task for testing"""
    return {
        'task_name': 'hearsay',
        'test': [
            {
                'input': 'Witness testimony about what they heard someone else say',
                'output': 'Yes'
            }
        ],
        'prompt_template': 'Is this hearsay? {input}'
    }

@pytest.mark.asyncio
async def test_evaluate_task(mock_agent, sample_task):
    """Test task evaluation"""
    
    # Setup mock response
    mock_agent.execute_with_trace.return_value = Mock(
        actual_output='Yes',
        execution_time=0.5,
        tool_calls=[],
        chain_steps=[],
        tokens_used={'total_tokens': 100},
        error=None
    )
    
    # Create evaluator
    loader = Mock()
    loader.load_task.return_value = sample_task
    
    evaluator = LegalBenchEvaluator(
        agent=mock_agent,
        loader=loader
    )
    
    # Run evaluation
    result = await evaluator.evaluate_task('hearsay', sample_size=1)
    
    # Assertions
    assert result.task_name == 'hearsay'
    assert result.accuracy == 1.0
    assert result.avg_execution_time == 0.5
    assert result.avg_tokens_used == 100

@pytest.mark.asyncio
async def test_agent_instrumentation():
    """Test agent instrumentation captures metrics"""
    
    agent = InstrumentedAgent()
    
    # Mock tool calls
    agent.tool_interceptor.log_call(
        tool_name='legal_search',
        input_data={'query': 'hearsay rule'},
        output_data={'results': ['Rule 801...']}
    )
    
    trace = await agent.execute_with_trace(
        prompt="Test prompt",
        task_name="test_task",
        expected_output="Expected"
    )
    
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0]['tool_name'] == 'legal_search'
```

---

## References and Resources

### Official LegalBench Resources
- **Paper**: [LEGALBENCH: A Collaboratively Built Benchmark](https://papers.nips.cc/paper_files/paper/2023/file/89e44582fd28ddfea1ea4dcb0ebbf4b0-Paper-Datasets_and_Benchmarks.pdf)
- **Website**: [hazyresearch.stanford.edu/legalbench](https://hazyresearch.stanford.edu/legalbench/)
- **GitHub**: [github.com/HazyResearch/legalbench](https://github.com/HazyResearch/legalbench)
- **HuggingFace**: [huggingface.co/datasets/nguha/legalbench](https://huggingface.co/datasets/nguha/legalbench)

### Related Benchmarks and Extensions
- **LegalBench-RAG**: [Retrieval-Augmented Generation Benchmark](https://arxiv.org/abs/2408.10343)
- **LegalAgentBench**: [Agent-Specific Legal Benchmark](https://arxiv.org/html/2412.17259v1)

### Claude Agent SDK Documentation
- **Official Docs**: [Claude Agent SDK Documentation]
- **API Reference**: [Claude API Reference]

### Additional Resources
- **Prompting Guide**: [docs.claude.com/prompting](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview)
- **Safety Guidelines**: Legal AI Safety Best Practices

---

## Appendix: Quick Start Commands

```bash
# Clone and setup
git clone https://github.com/your-org/legalbench-eval.git
cd legalbench-eval
pip install -r requirements.txt

# Download data
python scripts/download_legalbench.py

# Run basic evaluation
python scripts/run_evaluation.py --tasks hearsay definition_extraction --sample-size 50

# Run category evaluation
python scripts/run_evaluation.py --categories rule-application --sample-size 100

# Start API server
uvicorn legalbench_eval.api.evaluation_api:app --reload

# Run tests
pytest tests/ -v

# Generate report
python scripts/generate_report.py --eval-id <evaluation_id> --format markdown
```

---

*This document provides a comprehensive foundation for implementing LegalBench evaluations with Claude Agent SDK. Continue to iterate based on your specific requirements and findings from initial evaluations.*