"""
SIH 2026 Model Comparator Package
Step-by-step Baseline CSV vs AutoML Engine CSV Feature Engineering & Performance Evaluator
"""

from model_comparator.baseline_pipeline import BaselinePipeline
from model_comparator.engine_pipeline import EnginePipeline
from model_comparator.comparator import ModelComparator

__all__ = ["BaselinePipeline", "EnginePipeline", "ModelComparator"]
