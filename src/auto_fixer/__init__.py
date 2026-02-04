"""
Auto Test Fixer Module

Automatically fixes failing tests by:
1. Running pytest and parsing failures
2. Classifying failures (test mistake vs code bug)
3. Generating fixes for test mistakes
4. Applying fixes and re-running tests
"""

from .orchestrator import AutoTestFixerOrchestrator, FixResult
from .failure_parser import FailureParser, TestFailure
from .rule_classifier import RuleBasedClassifier
from .llm_classifier import LLMClassifier, LLMClassification
from .ast_context_extractor import ASTContextExtractor
from .llm_fixer import LLMFixer
from .ast_patcher import ASTPatcher

__all__ = [
    'AutoTestFixerOrchestrator',
    'FixResult',
    'FailureParser',
    'TestFailure',
    'RuleBasedClassifier',
    'LLMClassifier',
    'LLMClassification',
    'ASTContextExtractor',
    'LLMFixer',
    'ASTPatcher',
]
