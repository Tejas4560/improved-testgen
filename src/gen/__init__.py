# src/gen/__init__.py
"""
Professional test generation module.

This module provides comprehensive test generation capabilities for Python codebases,
creating robust pytest test suites that follow industry best practices.

Key features:
- Intelligent import handling with fallback mocking
- Strategic test file organization (4-5 files max)
- Comprehensive coverage of functions, classes, and API endpoints  
- Professional test patterns (Arrange-Act-Assert)
- Robust error handling and edge case testing
- Framework-agnostic support (FastAPI, Flask, Django)
"""

from .enhanced_generate import generate_all, main

__version__ = "2.0.0"
__author__ = "Test Generation System"

__all__ = ["generate_all", "main"]

# Module-level configuration
DEFAULT_CONFIG = {
    "max_unit_files": 4,
    "max_integration_files": 3, 
    "max_e2e_files": 2,
    "professional_patterns": True,
    "robust_imports": True,
    "comprehensive_coverage": True
}