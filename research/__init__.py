"""
Research Templates System for AA Microscope

Systematic, reproducible experiments for agent-agent dialogue research.

Main Components:
- TemplateExecutor: Execute individual template runs
- BatchRunner: Batch execution with progress tracking
- TemplateEvaluator: Template-specific evaluation metrics
- ResearchReporter: Generate comprehensive reports

Quick Start:
    >>> from research.template_executor import TemplateExecutor
    >>> executor = TemplateExecutor()
    >>> templates = executor.list_templates()
    >>> runs = executor.generate_experiment_runs("identity_archaeology")
    >>> executor.execute_run(runs[0])
"""

__version__ = "1.0.0"
__author__ = "AA Microscope Research Team"

from .template_executor import TemplateExecutor, ExperimentRun
from .batch_runner import BatchRunner, BatchExperimentResult
from .template_metrics import (
    TemplateEvaluator,
    TemplateMetrics,
    IdentityLeakDetector,
    EmotionalContagionAnalyzer,
    CreativityMeasure,
    ConversationBreakdownDetector
)
from .research_reporter import ResearchReporter

__all__ = [
    # Main classes
    'TemplateExecutor',
    'BatchRunner',
    'TemplateEvaluator',
    'ResearchReporter',
    
    # Data classes
    'ExperimentRun',
    'BatchExperimentResult',
    'TemplateMetrics',
    
    # Specialized analyzers
    'IdentityLeakDetector',
    'EmotionalContagionAnalyzer',
    'CreativityMeasure',
    'ConversationBreakdownDetector',
]
