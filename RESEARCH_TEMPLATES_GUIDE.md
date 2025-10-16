# AA Microscope Research Templates System
## Comprehensive Guide to Systematic Agent-Agent Dialogue Research

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Template Categories](#template-categories)
4. [Using the System](#using-the-system)
5. [Template Structure](#template-structure)
6. [Creating Custom Templates](#creating-custom-templates)
7. [Analysis & Metrics](#analysis--metrics)
8. [Best Practices](#best-practices)
9. [Research Workflow](#research-workflow)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The Research Templates System provides a comprehensive framework for conducting systematic, reproducible experiments in agent-agent dialogue. It includes:

- **15+ Pre-configured Templates** covering parameter sweeps, model comparisons, phenomenon detection, stress tests, and longitudinal studies
- **Automated Experiment Execution** with parameter sweep support and parallel processing
- **Specialized Metrics** for detecting identity leakage, empathy cascades, creativity emergence, and conversation breakdown
- **Statistical Analysis** with aggregated metrics and significance testing
- **Comprehensive Reporting** with markdown reports and comparative analysis

### Key Features

✅ **Reproducible Science** - Version-controlled templates with documented parameters  
✅ **Scalable Execution** - Batch processing with progress tracking and error handling  
✅ **Template-Specific Metrics** - Specialized evaluations for different research questions  
✅ **Statistical Rigor** - Multiple runs per condition with aggregated statistics  
✅ **Flexible Configuration** - JSON-based templates easy to modify and extend  

---

## Quick Start

### 1. List Available Templates

```bash
python research/template_executor.py list
```

Or filter by category:

```bash
python research/template_executor.py list parameter_sweep
python research/template_executor.py list phenomenon_specific
```

### 2. Preview a Template

```bash
python research/template_executor.py show temperature_matrix
```

### 3. Test Generate Experiment Runs

```bash
python research/template_executor.py generate temperature_matrix
```

This shows what runs would be created without executing them.

### 4. Execute a Template (Test Mode)

```bash
python research/template_executor.py execute temperature_matrix
```

This runs the first 3 experiments as a test.

### 5. Full Batch Execution

```bash
python research/batch_runner.py run temperature_matrix --max-runs 10
```

For full execution with all runs:

```bash
python research/batch_runner.py run temperature_matrix
```

### 6. Generate Report

```bash
python research/research_reporter.py report research_results/temperature_matrix_20251008_143022_results.json
```

---

## Template Categories

### 🔬 A. Parameter Sweep Templates

Systematically explore how system parameters affect conversational dynamics.

**Available Templates:**

1. **`temperature_matrix`** - Factorial sweep of all temperature combinations
   - Tests: 6 × 6 temperature combinations (0.3 to 1.3)
   - Runs: 108 total (3 per combination)
   - Duration: ~180 minutes
   - Research Question: *How do temperature combinations affect dynamics?*

2. **`context_window_sweep`** - Effect of memory length on coherence
   - Tests: 6 context lengths across 3 categories
   - Runs: 90 total (5 per condition)
   - Duration: ~150 minutes
   - Research Question: *How does context window size affect quality?*

3. **`conversation_length_study`** - Optimal conversation length
   - Tests: 8 different max turn counts (5 to 100)
   - Runs: 32 total (4 per length)
   - Duration: ~240 minutes
   - Research Question: *When do conversations plateau or collapse?*

### 🤖 B. Model Comparison Templates

Compare different LLM architectures and sizes in agent-agent interaction.

**Available Templates:**

4. **`architecture_comparison`** - Cross-model matrix
   - Tests: 6 model pairs across 4 categories
   - Runs: 120 total (5 per combination)
   - Duration: ~300 minutes
   - Research Question: *Which architectures produce interesting dynamics?*

5. **`model_size_scaling`** - Size effects study
   - Tests: Different model sizes (e.g., 8B, 70B, 405B)
   - Runs: 45 total (5 per comparison)
   - Duration: ~120 minutes
   - Research Question: *Do larger models = better emergence?*

6. **`david_goliath`** - Asymmetric model pairings
   - Tests: Small vs large model dynamics
   - Runs: 40 total (5 per pair)
   - Duration: ~100 minutes
   - Research Question: *Do large models dominate conversations?*

### 🎯 C. Phenomenon-Specific Templates

Target and amplify specific emergent conversational behaviors.

**Available Templates:**

7. **`identity_archaeology`** - AI self-revelation detection
   - Tests: 8 specialized prompts designed to elicit identity leaks
   - Runs: 40 total (5 per prompt)
   - Duration: ~150 minutes
   - Success Metric: Identity leak detection rate
   - Phenomena: AI self-reference, meta-awareness, human assumption breach

8. **`emotional_contagion`** - Empathy cascade study
   - Tests: 5 emotional states (distress, excitement, frustration, anxiety, joy)
   - Runs: 25 total (5 per emotion)
   - Duration: ~80 minutes
   - Success Metric: Emotional mirroring rate and empathy frequency
   - Phenomena: Empathy cascades, emotional mirroring, support behavior

9. **`creativity_emergence`** - Collaborative creative output
   - Tests: 6 creative prompts
   - Runs: 30 total (5 per prompt)
   - Duration: ~150 minutes
   - Success Metric: Novelty score and collaborative synthesis
   - Phenomena: Co-creativity, idea building, conceptual leaping

### ⚠️ D. Stress Test & Failure Mode Templates

Identify breaking points and failure patterns.

**Available Templates:**

10. **`breakdown_cascade`** - Conversation failure induction
    - Tests: 4 stress conditions (repetition, noise, contradiction, overflow)
    - Runs: 20 total (5 per condition)
    - Duration: ~120 minutes
    - Success Metric: Breakdown detection and recovery patterns
    - Phenomena: Semantic collapse, repetition loops, context overflow

11. **`conflict_escalation`** - Adversarial dynamics
    - Tests: 2 agent configurations × 3 controversial topics
    - Runs: 30 total (5 per combination)
    - Duration: ~90 minutes
    - Success Metric: Conflict resolution patterns
    - Phenomena: Argument escalation, resolution strategies

12. **`entropy_maximization`** - Chaos injection
    - Tests: 7 maximum entropy prompts
    - Runs: 35 total (5 per prompt)
    - Duration: ~60 minutes
    - Success Metric: Coherence from chaos score
    - Phenomena: Emergent order, sense-making strategies

### ⏰ E. Longitudinal & Evolutionary Templates

Study conversation evolution over extended timeframes.

**Available Templates:**

13. **`ultra_endurance`** - Marathon conversations (150 turns)
    - Tests: 3 categories with checkpoint analysis
    - Runs: 9 total (3 per category)
    - Duration: ~600 minutes
    - Success Metric: Long-term coherence and relationship development
    - Phenomena: Topic drift, conversational rituals, natural termination

14. **`agent_personality_stability`** - Multi-session consistency
    - Tests: 10 independent sessions with same agents
    - Runs: 10 total (1 per session)
    - Duration: ~120 minutes
    - Success Metric: Cross-session consistency score
    - Phenomena: Personality stability, behavioral signatures

---

## Using the System

### Basic Workflow

```bash
# 1. List templates by priority
python research/batch_runner.py run-priority critical --max-runs 5

# 2. Run a specific template
python research/batch_runner.py run identity_archaeology --parallel 2

# 3. Run multiple templates
python research/batch_runner.py run-multiple "temperature_matrix,context_window_sweep" --max-runs 10

# 4. Generate reports
python research/research_reporter.py report research_results/identity_archaeology_*.json

# 5. Compare multiple batches
python research/research_reporter.py compare research_results/*_results.json
```

### Advanced Options

**Parallel Execution:**

```bash
# Run 4 conversations simultaneously
python research/batch_runner.py run creativity_emergence --parallel 4
```

**Limited Runs (for testing):**

```bash
# Only run first 5 experiments
python research/batch_runner.py run temperature_matrix --max-runs 5
```

**Custom Analysis:**

```python
from research.template_executor import TemplateExecutor
from research.template_metrics import TemplateEvaluator

# Load executor
executor = TemplateExecutor()

# Generate runs
runs = executor.generate_experiment_runs("identity_archaeology")

# Execute specific runs
for run in runs[:3]:
    executor.execute_run(run)

# Evaluate with custom metrics
evaluator = TemplateEvaluator()
# ... custom evaluation code
```

---

## Template Structure

Templates are defined in `research_templates.json` with the following structure:

```json
{
  "template_id": {
    "template_id": "unique_identifier",
    "category": "parameter_sweep|model_comparison|phenomenon_specific|stress_test|longitudinal",
    "type": "specific_type",
    "description": "What this template does",
    "research_question": "Specific testable hypothesis",
    "hypothesis": "Expected outcome (optional)",
    
    "configuration": {
      "base_params": {
        "max_turns": 20,
        "agent_a_temp": 0.7,
        "agent_b_temp": 0.7,
        "runs_per_condition": 3
      },
      "sweep_params": {
        // Parameters to vary
      }
    },
    
    "analysis_focus": [
      "metric1", "metric2", "metric3"
    ],
    
    "success_criteria": {
      "minimum_coherence": 0.5
    },
    
    "metadata": {
      "priority": "critical|high|medium|low",
      "estimated_runs": 108,
      "estimated_duration_minutes": 180,
      "research_phase": "foundational|exploratory|confirmatory"
    }
  }
}
```

### Template Types

- **`factorial_sweep`** - Full factorial combinations (e.g., temperature matrix)
- **`parameter_sweep`** - Single parameter variation (e.g., context window)
- **`cross_model_matrix`** - Model comparison matrix
- **`asymmetric_pairing`** - Asymmetric model pairs
- **`identity_leak_detection`** - AI self-revelation study
- **`empathy_cascade_study`** - Emotional contagion tracking
- **`creative_collaboration`** - Co-creativity measurement
- **`failure_mode_induction`** - Stress testing
- **`adversarial_dynamics`** - Conflict studies
- **`chaos_injection`** - Maximum entropy testing
- **`marathon_conversation`** - Extended dialogue study
- **`multi_session_consistency`** - Personality stability testing

---

## Creating Custom Templates

### Step 1: Define Your Research Question

Example: *"How does increasing model temperature affect identity leak rate?"*

### Step 2: Design the Template

```json
{
  "custom_identity_temp_sweep": {
    "template_id": "custom_identity_temp_sweep",
    "category": "parameter_sweep",
    "type": "parameter_sweep",
    "description": "Study identity leak rate across temperatures",
    "research_question": "Does higher temperature increase identity revelation?",
    
    "configuration": {
      "base_params": {
        "max_turns": 25,
        "runs_per_temp": 5,
        "prompt_category": "identity"
      },
      "sweep_params": {
        "temperatures": [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
      }
    },
    
    "analysis_focus": [
      "identity_leak_rate",
      "meta_awareness_frequency",
      "conversation_coherence"
    ],
    
    "metadata": {
      "priority": "high",
      "estimated_runs": 35,
      "estimated_duration_minutes": 90
    }
  }
}
```

### Step 3: Add to Templates File

Edit `research_templates.json` and add your template to the `"templates"` object.

### Step 4: Implement Custom Metrics (if needed)

If your template requires specialized metrics, add them to `research/template_metrics.py`:

```python
class CustomMetricAnalyzer:
    def analyze_custom_phenomenon(self, messages):
        # Your custom analysis logic
        return {
            'custom_metric': value,
            'detection_score': score
        }
```

### Step 5: Test Your Template

```bash
# Generate runs
python research/template_executor.py generate custom_identity_temp_sweep

# Test execution
python research/template_executor.py execute custom_identity_temp_sweep

# Full run
python research/batch_runner.py run custom_identity_temp_sweep
```

---

## Analysis & Metrics

### Quantitative Metrics (All Templates)

**Conversation Dynamics:**
- Turn balance ratio
- Average turn length per agent
- Response time patterns
- Shannon entropy
- Lexical diversity

**Linguistic Analysis:**
- Sentence complexity
- Readability scores (Flesch-Kincaid)
- Vocabulary richness
- Question frequency

### Template-Specific Metrics

**Identity Leak Detection:**
- AI keyword frequency
- Meta-awareness instances
- Human assumption breaches
- Leak rate (overall score)

**Empathy Cascade Study:**
- Sentiment trajectory
- Empathy marker frequency
- Emotional mirroring rate
- Sentiment volatility

**Creativity Emergence:**
- Lexical diversity
- Metaphor count
- Idea building instances
- Novelty score

**Stress Test/Breakdown:**
- Repetition score
- Semantic diversity trend
- Engagement trend
- Resilience score

### Success Scoring

Each template defines success differently:

- **Parameter Sweeps** → Variation detection and statistical significance
- **Model Comparisons** → Quality differential between models
- **Phenomenon Detection** → Frequency and intensity of target phenomenon
- **Stress Tests** → Breakdown occurrence and recovery patterns
- **Longitudinal** → Stability and evolution metrics

---

## Best Practices

### 🎯 Research Design

1. **Start with Critical Templates** - Run foundational studies first (temperature_matrix, architecture_comparison, identity_archaeology)

2. **Multiple Runs per Condition** - Minimum 3 runs, preferably 5-10 for statistical power

3. **Document Everything** - Keep research notes on interesting findings and edge cases

4. **Incremental Testing** - Use `--max-runs` to test templates before full execution

### ⚙️ Execution

1. **Use Parallel Execution Wisely**
   - Sequential (`--parallel 1`) for development/debugging
   - Parallel (`--parallel 2-4`) for production runs
   - Consider API rate limits

2. **Monitor Progress**
   - Check `research_results/` directory for intermediate results
   - Review error summaries in batch reports

3. **Save Your Work**
   - Batch results are auto-saved to JSON
   - Generate reports immediately after completion

### 📊 Analysis

1. **Compare Across Templates**
   - Use comparative reports to identify patterns
   - Look for phenomena that appear across multiple template types

2. **Statistical Significance**
   - Pay attention to standard deviations
   - High variance = need more samples or parameter adjustment

3. **Qualitative Review**
   - Read actual conversations for context
   - Automated metrics miss nuance

### 🔧 Troubleshooting

**High Failure Rate?**
- Check API keys and connectivity
- Review error_summary in batch results
- Consider adjusting max_turns or temperature

**Unexpected Results?**
- Verify template configuration
- Check if prompt category matches research question
- Review individual conversations for anomalies

**Long Execution Times?**
- Use `--max-runs` to limit batch size
- Increase `--parallel` (with caution)
- Consider shorter max_turns for initial tests

---

## Research Workflow

### Phase 1: Foundational Studies (Week 1-2)

```bash
# Critical templates first
python research/batch_runner.py run-priority critical --max-runs 10

# Generate reports
python research/research_reporter.py compare research_results/*critical*.json
```

**Templates:**
- `temperature_matrix` - Establish baseline parameter ranges
- `architecture_comparison` - Identify best model pairs
- `identity_archaeology` - Validate phenomenon detection

### Phase 2: Exploratory Research (Week 3-4)

```bash
# High priority templates
python research/batch_runner.py run-priority high --max-runs 15
```

**Templates:**
- `emotional_contagion` - Map empathy dynamics
- `creativity_emergence` - Study co-creation
- `context_window_sweep` - Optimize memory configuration
- `model_size_scaling` - Understand scaling effects

### Phase 3: Targeted Studies (Week 5-6)

Based on Phase 1-2 findings, run targeted follow-ups:

```bash
# Custom templates based on discoveries
python research/batch_runner.py run custom_follow_up_1
python research/batch_runner.py run custom_follow_up_2
```

### Phase 4: Longitudinal & Stress Testing (Week 7-8)

```bash
# Resource-intensive templates
python research/batch_runner.py run ultra_endurance --parallel 1
python research/batch_runner.py run breakdown_cascade
python research/batch_runner.py run agent_personality_stability
```

### Phase 5: Synthesis & Publication (Week 9-10)

```bash
# Generate comprehensive comparative report
python research/research_reporter.py compare research_results/*.json

# Compile findings into research paper
# Document methodology
# Create visualizations
```

---

## Output Structure

```
research_results/
├── temperature_matrix_20251008_143022_results.json       # Raw batch data
├── temperature_matrix_20251008_143022_report.md          # Single template report
├── identity_archaeology_20251008_151045_results.json
├── identity_archaeology_20251008_151045_report.md
└── comparative_report_20251008_160000.md                 # Multi-template comparison
```

### Batch Result JSON Structure

```json
{
  "batch_id": "template_id_timestamp",
  "template_id": "temperature_matrix",
  "start_time": "2025-10-08T14:30:22",
  "end_time": "2025-10-08T17:45:10",
  "total_runs": 108,
  "completed_runs": 105,
  "failed_runs": 3,
  "total_turns": 2156,
  "total_duration_seconds": 11688,
  "runs": [/* array of individual runs */],
  "statistics": {
    "completion_rate": 0.97,
    "turn_statistics": {
      "mean": 20.5,
      "median": 20,
      "stdev": 3.2,
      "min": 12,
      "max": 30
    },
    "analysis_aggregates": {
      "turn_balance": {"mean": 0.92, "stdev": 0.08},
      "information_entropy": {"mean": 8.5, "stdev": 1.2}
    }
  },
  "error_summary": {
    "APIError": 2,
    "TimeoutError": 1
  }
}
```

---

## API Reference

### TemplateExecutor

```python
from research.template_executor import TemplateExecutor

executor = TemplateExecutor(templates_file="research_templates.json")

# List templates
templates = executor.list_templates(category="parameter_sweep")

# Get specific template
template = executor.get_template("temperature_matrix")

# Generate experiment runs
runs = executor.generate_experiment_runs("temperature_matrix")

# Execute single run
result = executor.execute_run(runs[0], verbose=True)
```

### BatchRunner

```python
from research.batch_runner import BatchRunner

runner = BatchRunner(output_dir="research_results", parallel=2)

# Run template batch
result = runner.run_template_batch(
    template_id="identity_archaeology",
    max_runs=10,
    save_progress=True
)

# Run multiple templates
results = runner.run_multiple_templates(
    template_ids=["template1", "template2"],
    max_runs_per_template=5
)
```

### TemplateEvaluator

```python
from research.template_metrics import TemplateEvaluator

evaluator = TemplateEvaluator()

# Evaluate conversation with template-specific metrics
metrics = evaluator.evaluate_template_run(
    conversation=conversation_obj,
    template_type="identity_leak_detection"
)

print(f"Success Score: {metrics.success_score}")
print(f"Phenomena: {metrics.phenomena_detected}")
```

---

## Contributing

### Adding New Template Categories

1. Define category in `research_templates.json` metadata
2. Add type handling in `TemplateExecutor._generate_*_runs()`
3. Implement specialized metrics in `template_metrics.py`
4. Update this documentation

### Reporting Issues

If you encounter bugs or unexpected behavior:

1. Check the error_summary in batch results
2. Review individual conversation logs
3. Verify template configuration
4. Document reproducible steps

---

## Citation

If you use the AA Microscope Research Templates System in your research, please cite:

```bibtex
@software{aa_microscope_templates,
  title={AA Microscope Research Templates System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/aa-microscope}
}
```

---

## License

This research framework is part of the AA Microscope project.

---

## Appendix A: Complete Template List

| Template ID | Category | Priority | Runs | Duration |
|-------------|----------|----------|------|----------|
| temperature_matrix | Parameter Sweep | Critical | 108 | 180 min |
| context_window_sweep | Parameter Sweep | Critical | 90 | 150 min |
| conversation_length_study | Parameter Sweep | High | 32 | 240 min |
| architecture_comparison | Model Comparison | Critical | 120 | 300 min |
| model_size_scaling | Model Comparison | High | 45 | 120 min |
| david_goliath | Model Comparison | High | 40 | 100 min |
| identity_archaeology | Phenomenon-Specific | Critical | 40 | 150 min |
| emotional_contagion | Phenomenon-Specific | High | 25 | 80 min |
| creativity_emergence | Phenomenon-Specific | High | 30 | 150 min |
| breakdown_cascade | Stress Test | High | 20 | 120 min |
| conflict_escalation | Stress Test | Medium | 30 | 90 min |
| entropy_maximization | Stress Test | Medium | 35 | 60 min |
| ultra_endurance | Longitudinal | Medium | 9 | 600 min |
| agent_personality_stability | Longitudinal | Medium | 10 | 120 min |

**Total Estimated Duration:** ~2,460 minutes (~41 hours) for complete execution of all templates

---

## Appendix B: Phenomenon Detection Keywords

### Identity Leak
- **AI Keywords:** AI, artificial intelligence, language model, LLM, GPT, Claude, algorithm, neural network
- **Meta Patterns:** "I don't actually feel", "I can't truly understand", "as an AI", "my training"
- **Human Breach:** "you might be an AI", "are you human", "we're both AIs"

### Empathy Markers
- "I understand", "I hear you", "that sounds", "I can imagine", "I'm sorry", "that must", "I feel"

### Creativity Indicators
- Metaphor patterns: "like", "as" comparisons
- Idea building: "building on", "what if", "or we could", "how about"
- Questions indicating exploration

### Breakdown Signals
- High repetition rate (>50%)
- Decreasing lexical diversity
- Shortening message length
- Semantic coherence loss

---

**Last Updated:** 2025-10-08  
**Version:** 1.0.0  
**Maintainer:** AA Microscope Research Team
