"""
Rule-Based Classifier

Classifies test failures as "test_mistake" or "unknown" based on patterns.
"""

import re
from typing import Literal
from .failure_parser import TestFailure


ClassificationType = Literal["test_mistake", "unknown"]


class RuleBasedClassifier:
    """
    Classifies test failures using pattern-matching rules.

    Returns "test_mistake" for common test-writing errors,
    "unknown" otherwise (could be code bug or complex test mistake).
    """

    # Patterns that indicate test mistakes
    TEST_MISTAKE_PATTERNS = [
        # Import errors in tests
        (r"ImportError|ModuleNotFoundError", "Missing import in test"),
        (r"cannot import name", "Wrong import in test"),

        # Fixture errors
        (r"fixture .* not found", "Missing or misspelled fixture"),
        (r"fixture .* doesn't exist", "Missing or misspelled fixture"),

        # AttributeError in test setup/assertions
        (r"AttributeError.*Mock|MagicMock", "Incorrect mock usage"),
        (r"AttributeError.*has no attribute", "Wrong attribute access in test"),

        # TypeError in test code
        (r"TypeError.*takes \d+ positional argument", "Wrong number of arguments in test"),
        (r"TypeError.*missing \d+ required positional argument", "Missing arguments in test"),
        (r"TypeError.*got an unexpected keyword argument", "Wrong keyword argument in test"),

        # NameError (undefined variable in test)
        (r"NameError.*name .* is not defined", "Undefined variable in test"),

        # Assertion errors with specific patterns
        (r"assert None", "Asserting on None (likely test setup issue)"),
        (r"AssertionError.*is not True", "Incorrect assertion pattern"),

        # Indentation errors in test
        (r"IndentationError", "Indentation error in test"),
        (r"SyntaxError", "Syntax error in test"),

        # Test configuration errors
        (r"pytest.*error|pytest.*failed", "Pytest configuration error"),

        # Database/ORM errors in test setup
        (r"DatabaseError.*no such table", "Database not set up in test"),
        (r"OperationalError.*no such table", "Database not set up in test"),

        # File not found in test
        (r"FileNotFoundError", "File not found (test setup issue)"),

        # JSON decode errors (bad test data)
        (r"JSONDecodeError", "Invalid JSON in test data"),

        # Key errors (missing key in test data)
        (r"KeyError.*in test", "Missing key in test data"),

        # Client/request errors in tests
        (r"No route matches", "Incorrect route in test"),
        (r"404.*Not Found", "Wrong URL in test request"),

        # Async errors in tests
        (r"RuntimeError.*cannot be called from a running event loop", "Async setup issue in test"),
        (r"asyncio.*was never awaited", "Missing await in test"),
    ]

    # Patterns that suggest code bugs (not test mistakes)
    CODE_BUG_PATTERNS = [
        (r"ZeroDivisionError", "Code bug: division by zero"),
        (r"ValueError.*invalid literal", "Code bug: invalid value"),
        (r"IndexError.*out of range", "Code bug: index out of range"),
        (r"RecursionError", "Code bug: infinite recursion"),
    ]

    def __init__(self):
        pass

    def classify(self, failure: TestFailure) -> ClassificationType:
        """
        Classify a test failure.

        Args:
            failure: TestFailure object

        Returns:
            "test_mistake" or "unknown"
        """
        # Combine error message and traceback for analysis
        error_context = f"{failure.exception_type} {failure.error_message} {failure.traceback}"

        # Check test mistake patterns first
        for pattern, description in self.TEST_MISTAKE_PATTERNS:
            if re.search(pattern, error_context, re.IGNORECASE):
                return "test_mistake"

        # Check if it looks like a code bug
        for pattern, description in self.CODE_BUG_PATTERNS:
            if re.search(pattern, error_context, re.IGNORECASE):
                # Could be code bug - return unknown so LLM can decide
                return "unknown"

        # Additional heuristics based on traceback location
        if self._failure_in_test_file(failure):
            # If the error originates in the test file itself, more likely a test mistake
            if failure.exception_type in ["AttributeError", "TypeError", "NameError", "ImportError"]:
                return "test_mistake"

        # Default: unknown (let LLM decide)
        return "unknown"

    def _failure_in_test_file(self, failure: TestFailure) -> bool:
        """
        Check if the failure originated in the test file.

        Args:
            failure: TestFailure object

        Returns:
            True if failure is in test file
        """
        # Check if test file appears near the end of traceback
        traceback_lines = failure.traceback.split('\n')

        for line in reversed(traceback_lines[-10:]):  # Check last 10 lines
            if failure.test_file in line:
                return True

        return False

    def get_classification_reason(self, failure: TestFailure) -> str:
        """
        Get a human-readable reason for the classification.

        Args:
            failure: TestFailure object

        Returns:
            Reason string
        """
        error_context = f"{failure.exception_type} {failure.error_message} {failure.traceback}"

        # Find matching pattern
        for pattern, description in self.TEST_MISTAKE_PATTERNS:
            if re.search(pattern, error_context, re.IGNORECASE):
                return description

        for pattern, description in self.CODE_BUG_PATTERNS:
            if re.search(pattern, error_context, re.IGNORECASE):
                return description

        return "No specific pattern matched"
