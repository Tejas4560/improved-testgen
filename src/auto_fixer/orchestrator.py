"""
Auto Test Fixer Orchestrator

Main orchestrator that coordinates the entire test fixing workflow.
"""

import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import os

# Use absolute imports to avoid issues when loaded in different contexts
from .failure_parser import FailureParser, TestFailure
from .rule_classifier import RuleBasedClassifier
from .llm_classifier import LLMClassifier, LLMClassification
from .ast_context_extractor import ASTContextExtractor
from .embedding_context_extractor import EmbeddingContextExtractor
from .llm_fixer import LLMFixer
from .ast_patcher import ASTPatcher


@dataclass
class FixResult:
    """Result of a fix attempt."""
    test_file: str
    test_name: str
    classification: str
    fix_attempted: bool
    fix_successful: bool
    reason: str


class AutoTestFixerOrchestrator:
    """
    Orchestrates the automatic test fixing workflow.

    Workflow:
    1. Run pytest with JSON output
    2. Parse all failures
    3. Classify each failure (rule-based + LLM)
    4. Extract AST context for test mistakes
    5. Generate fixes using LLM
    6. Apply fixes using AST patcher
    7. Re-run tests
    8. Repeat until all test mistakes fixed or max iterations reached
    """

    def __init__(
        self,
        test_directory: str = "tests",
        project_root: str = ".",
        max_iterations: int = 3
    ):
        self.test_directory = test_directory
        self.project_root = project_root
        self.max_iterations = max_iterations

        # Check for verbose mode
        verbose = os.getenv("AUTOFIXER_VERBOSE", "").lower() in ("true", "1", "yes")

        # Check if embeddings should be used (default: yes)
        use_embeddings = os.getenv("USE_EMBEDDINGS", "true").lower() in ("true", "1", "yes")

        # Initialize components
        self.failure_parser = FailureParser(test_directory)
        self.rule_classifier = RuleBasedClassifier()
        self.llm_classifier = LLMClassifier()

        # Use embedding-enhanced context extractor (hybrid AST + embeddings)
        if use_embeddings:
            self.context_extractor = EmbeddingContextExtractor(
                project_root=project_root,
                use_embeddings=True,
                verbose=verbose
            )
            if verbose:
                print("Using embedding-enhanced context extraction")
        else:
            # Fallback to pure AST extraction
            self.context_extractor = ASTContextExtractor(project_root, verbose=verbose)
            if verbose:
                print("Using AST-only context extraction")

        self.llm_fixer = LLMFixer()
        self.ast_patcher = ASTPatcher()

        # Track results
        self.fix_history: List[FixResult] = []
        self.code_bugs: List[TestFailure] = []

    def run(self, extra_pytest_args: List[str] = None) -> Dict[str, Any]:
        """
        Run the auto-fixer workflow.

        Args:
            extra_pytest_args: Additional pytest arguments

        Returns:
            Summary of fixes and results
        """
        print("=" * 80)
        print("AUTO TEST FIXER - STARTING")
        print("=" * 80)

        iteration = 0
        all_tests_fixed = False

        while iteration < self.max_iterations and not all_tests_fixed:
            iteration += 1
            print(f"\n{'=' * 80}")
            print(f"ITERATION {iteration}/{self.max_iterations}")
            print(f"{'=' * 80}\n")

            # Step 1: Run pytest and parse failures
            print("Step 1: Running pytest and parsing failures...")
            failures = self.failure_parser.run_and_parse(extra_pytest_args)

            if not failures:
                print("No test failures found!")
                all_tests_fixed = True
                break

            print(f"Found {len(failures)} failing test(s)")

            # Step 2-6: Process each failure
            test_mistakes_fixed = []
            code_bugs_found = []

            for idx, failure in enumerate(failures, 1):
                print(f"\n Processing failure {idx}/{len(failures)}")
                print(f"Test: {failure.test_name} in {failure.test_file}")

                result = self._process_failure(failure)
                self.fix_history.append(result)

                if result.classification == "code_bug":
                    code_bugs_found.append(failure)
                    print(f"Classification: CODE BUG (skipped)")
                elif result.fix_successful:
                    test_mistakes_fixed.append(failure)
                    print(f"Classification: TEST MISTAKE (fixed)")
                else:
                    print(f"Classification: TEST MISTAKE (fix failed)")

            print(f"\n{'=' * 80}")
            print(f"Iteration {iteration} Summary:")
            print(f"Test mistakes fixed: {len(test_mistakes_fixed)}")
            print(f"Code bugs found: {len(code_bugs_found)}")
            print(f"{'=' * 80}")

            # Update code bugs list
            self.code_bugs.extend(code_bugs_found)

            # Only stop if we made NO progress at all (no fixes, no attempts)
            # Continue if we fixed some tests OR if we're still attempting fixes
            if len(test_mistakes_fixed) == 0 and iteration > 1:
                # Give it at least 2 iterations before stopping
                # Only stop if we've tried and made zero progress
                print("\nNo test mistakes fixed in this iteration and we've tried multiple times. Stopping.")
                break

        # Final summary
        return self._generate_summary(iteration)

    def _process_failure(self, failure: TestFailure) -> FixResult:
        """
        Process a single test failure.

        Args:
            failure: TestFailure object

        Returns:
            FixResult object
        """
        # Step 2: Rule-based classification
        rule_classification = self.rule_classifier.classify(failure)

        # Extract context ONCE here (will be reused for both classification and fixing)
        # This is a significant optimization - avoids 3-4x duplicate context extraction
        test_code = self._read_test_function(failure)
        source_code = self.context_extractor.get_full_context_string(
            failure.test_file,
            failure.test_name,
            failure.error_message
        )

        if rule_classification == "test_mistake":
            print(f"Rule classifier: test_mistake")
            return self._fix_test_mistake(failure, "rule-based classification", test_code, source_code)

        # Step 3: LLM classification
        print(f"Rule classifier: unknown, using LLM...")

        # LLM classification (reuse extracted context)
        llm_result = self.llm_classifier.classify(failure, test_code, source_code)
        print(f"LLM classifier: {llm_result.classification} ({llm_result.reason})")

        if llm_result.classification == "test_mistake":
            # Try to use LLM's suggested fix first
            if llm_result.fixed_code:
                success = self._apply_fix(failure, llm_result.fixed_code)
                if success:
                    return FixResult(
                        test_file=failure.test_file,
                        test_name=failure.test_name,
                        classification="test_mistake",
                        fix_attempted=True,
                        fix_successful=True,
                        reason=llm_result.reason
                    )

            # If LLM fix didn't work, generate a new fix (reuse context)
            return self._fix_test_mistake(failure, llm_result.reason, test_code, source_code)

        # Code bug - don't fix
        return FixResult(
            test_file=failure.test_file,
            test_name=failure.test_name,
            classification="code_bug",
            fix_attempted=False,
            fix_successful=False,
            reason=llm_result.reason
        )

    def _fix_test_mistake(
        self,
        failure: TestFailure,
        reason: str,
        test_code: str,
        source_code: str
    ) -> FixResult:
        """
        Fix a test mistake with multi-attempt learning.

        If the first fix fails validation, we try again with feedback about
        WHY it failed, giving the LLM a chance to learn and improve.

        Args:
            failure: TestFailure object
            reason: Reason for classification
            test_code: Pre-extracted test function code (cached)
            source_code: Pre-extracted source context (cached)

        Returns:
            FixResult object
        """
        # Context is now passed in (cached from _process_failure)
        # This eliminates 3x duplicate extraction (one per fix attempt)

        max_attempts = 3
        previous_fix = None
        previous_failure_output = None

        for attempt in range(1, max_attempts + 1):
            # Step 5: Generate fix
            if attempt == 1:
                print(f"Generating fix...")
            else:
                print(f"Generating fix (attempt {attempt}/{max_attempts})...")
                print(f"Learning from previous failure...")

            fixed_code = self.llm_fixer.fix_test(
                failure,
                test_code,
                source_code,
                previous_fix_attempt=previous_fix,
                previous_failure_output=previous_failure_output
            )

            if not fixed_code:
                if attempt == max_attempts:
                    return FixResult(
                        test_file=failure.test_file,
                        test_name=failure.test_name,
                        classification="test_mistake",
                        fix_attempted=True,
                        fix_successful=False,
                        reason=f"{reason} (fix generation failed after {attempt} attempts)"
                    )
                continue  # Try again

            # Step 6: Apply fix (returns success flag and failure output if failed)
            success, failure_output = self._apply_fix_with_feedback(failure, fixed_code)

            if success:
                print(f"Fix successful on attempt {attempt}!")
                return FixResult(
                    test_file=failure.test_file,
                    test_name=failure.test_name,
                    classification="test_mistake",
                    fix_attempted=True,
                    fix_successful=True,
                    reason=reason
                )

            # Fix failed - prepare for next attempt
            previous_fix = fixed_code
            previous_failure_output = failure_output

            if attempt < max_attempts:
                print(f"Fix attempt {attempt} failed, will retry with feedback...")
            else:
                print(f"All {max_attempts} fix attempts failed")

        return FixResult(
            test_file=failure.test_file,
            test_name=failure.test_name,
            classification="test_mistake",
            fix_attempted=True,
            fix_successful=False,
            reason=f"{reason} (fix validation failed after {max_attempts} attempts)"
        )

    def _apply_fix(self, failure: TestFailure, fixed_code: str) -> bool:
        """
        Apply a fix to a test file.

        Args:
            failure: TestFailure object
            fixed_code: Fixed function code

        Returns:
            True if patch successful
        """
        print(f"Applying fix...")

        # Strip parameter suffix for parameterized tests
        # e.g., "test_foo[param]" → "test_foo"
        base_test_name = self._strip_test_parameters(failure.test_name)

        success = self.ast_patcher.patch_test_function(
            failure.test_file,
            base_test_name,
            fixed_code
        )

        if success:
            # Validate the patch
            if self.ast_patcher.validate_patch(failure.test_file):
                print(f"Fix applied successfully")
                return True
            else:
                print(f"Fix validation failed")
                return False
        else:
            print(f"Fix application failed")
            return False

    def _apply_fix_with_feedback(self, failure: TestFailure, fixed_code: str) -> tuple[bool, str]:
        """
        Apply a fix and return detailed feedback if it fails.

        Args:
            failure: TestFailure object
            fixed_code: Fixed function code

        Returns:
            Tuple of (success, failure_output)
            - success: True if patch successful
            - failure_output: Pytest output if failed, empty string if succeeded
        """
        print(f"Applying fix...")

        # Strip parameter suffix for parameterized tests
        base_test_name = self._strip_test_parameters(failure.test_name)

        success, failure_output = self.ast_patcher.patch_test_function_with_feedback(
            failure.test_file,
            base_test_name,
            fixed_code
        )

        if success:
            # Validate the patch
            if self.ast_patcher.validate_patch(failure.test_file):
                print(f"Fix applied successfully")
                return True, ""
            else:
                print(f"Fix validation failed")
                return False, "Syntax validation failed after applying fix"
        else:
            if failure_output:
                print(f"Fix validation failed - test still fails")
            else:
                print(f"Fix application failed")
            return False, failure_output

    def _strip_test_parameters(self, test_name: str) -> str:
        """
        Strip pytest parameter suffix from test name.

        Parameterized tests have names like "test_foo[param]" but the
        actual function in AST is just "test_foo".

        Args:
            test_name: Full test name with parameters

        Returns:
            Base test name without parameters
        """
        if '[' in test_name:
            return test_name.split('[')[0]
        return test_name

    def _read_test_function(self, failure: TestFailure) -> str:
        """
        Read the failing test function code.

        Args:
            failure: TestFailure object

        Returns:
            Test function source code
        """
        try:
            with open(failure.test_file, 'r') as f:
                content = f.read()

            # Try to extract just the function
            import ast
            tree = ast.parse(content)

            # Strip parameter suffix for parameterized tests
            base_test_name = self._strip_test_parameters(failure.test_name)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == base_test_name:
                    return ast.unparse(node)

            return content  # Fallback to full file

        except Exception as e:
            print(f"Error reading test function: {e}")
            return ""

    def _generate_summary(self, iterations: int) -> Dict[str, Any]:
        """
        Generate final summary report.

        Args:
            iterations: Number of iterations run

        Returns:
            Summary dictionary
        """
        # Count unique failures (not all retry attempts across iterations)
        # Group by test_file + test_name to get unique tests
        unique_failures = {}
        for result in self.fix_history:
            key = (result.test_file, result.test_name)
            # Keep the most recent result for each unique test
            # (later iterations have more information)
            unique_failures[key] = result

        unique_results = list(unique_failures.values())
        total_failures = len(unique_results)
        test_mistakes = [r for r in unique_results if r.classification == "test_mistake"]
        code_bugs = [r for r in unique_results if r.classification == "code_bug"]
        successful_fixes = [r for r in test_mistakes if r.fix_successful]

        summary = {
            "iterations": iterations,
            "total_failures": total_failures,
            "test_mistakes": len(test_mistakes),
            "code_bugs": len(code_bugs),
            "successful_fixes": len(successful_fixes),
            "failed_fixes": len(test_mistakes) - len(successful_fixes),
            "fix_history": [asdict(r) for r in self.fix_history],
            "unique_failures": [asdict(r) for r in unique_results],
            "code_bugs_list": [f.to_dict() for f in self.code_bugs]
        }

        # Print summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"Iterations: {iterations}/{self.max_iterations}")
        print(f"Total unique failures: {total_failures}")
        print(f"Test mistakes: {len(test_mistakes)}")
        print(f"  - Fixed: {len(successful_fixes)}")
        print(f"  - Failed to fix: {len(test_mistakes) - len(successful_fixes)}")
        print(f"Code bugs (not fixed): {len(code_bugs)}")
        print("=" * 80)

        # Save report
        with open("auto_fixer_report.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\nDetailed report saved to: auto_fixer_report.json")

        return summary