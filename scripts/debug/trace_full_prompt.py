#!/usr/bin/env python3
"""
FULL PROMPT TRACER - Shows EXACTLY what gets sent to the LLM

This script runs pytest, captures a real failure, and shows you EXACTLY
what variables get constructed and sent to the LLM, with detailed breakdowns.

Usage:
    python trace_full_prompt.py --test-file <path> --project-root <path>

Example:
    python trace_full_prompt.py \\
        --test-file generated_tests/test_e2e_20251122_214922_01.py \\
        --project-root /home/sigmoid/test-repos/backend_code
"""

import sys
import os
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from auto_fixer.failure_parser import FailureParser
from auto_fixer.embedding_context_extractor import EmbeddingContextExtractor


def run_pytest(test_file: str) -> tuple[str, list]:
    """
    Run pytest on a test file and capture failures.

    Returns:
        (pytest_output, failures)
    """
    print("=" * 80)
    print("STEP 1: Running pytest to capture REAL failure")
    print("=" * 80)
    print(f"\n Running: pytest {test_file} -v --tb=long\n")

    result = subprocess.run(
        ["pytest", test_file, "-v", "--tb=short"],
        capture_output=True,
        text=True
    )

    output = result.stdout + "\n" + result.stderr

    # Parse failures using FailureParser
    parser = FailureParser()
    json_data = parser._parse_text_output(output)
    failures = parser.parse_failures(json_data)

    print(f"Pytest completed")
    print(f"   Exit code: {result.returncode}")
    print(f"   Output length: {len(output)} chars, {output.count(chr(10))} lines")
    print(f"   Failures found: {len(failures)}")

    if failures:
        print(f"\n First failure details:")
        failure = failures[0]
        print(f"   Test file: {failure.test_file}")
        print(f"   Test name: {failure.test_name}")
        print(f"   Exception: {failure.exception_type}")
        print(f"   Error: {failure.error_message[:100]}...")

    return output, failures


def extract_test_code(failure, test_file_path: str) -> str:
    """
    Extract test code using the SAME logic as orchestrator.py.

    This shows what `test_code` variable contains.
    """
    print("\n" + "=" * 80)
    print("STEP 2: Extracting TEST CODE (test_code variable)")
    print("=" * 80)

    # Read test file
    with open(test_file_path, 'r') as f:
        content = f.read()

    total_lines = content.count('\n') + 1
    total_chars = len(content)

    print(f"\n Test file stats:")
    print(f"   Path: {test_file_path}")
    print(f"   Total lines: {total_lines}")
    print(f"   Total chars: {total_chars:,}")
    print(f"   Est. tokens: ~{total_chars // 4:,}")

    # Try AST extraction
    print(f"\n Attempting extraction methods...")

    import ast
    import re

    try:
        tree = ast.parse(content)
        base_test_name = failure.test_name.split('[')[0]

        print(f"   Looking for function: '{base_test_name}'")

        # AST extraction
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == base_test_name:
                    test_code = ast.unparse(node)
                    lines = test_code.count('\n') + 1
                    chars = len(test_code)

                    print(f"\n AST extraction SUCCESS!")
                    print(f"      Extracted lines: {lines}")
                    print(f"      Extracted chars: {chars:,}")
                    print(f"      Est. tokens: ~{chars // 4:,}")
                    print(f"      Reduction: {((total_lines - lines) / total_lines * 100):.1f}%")

                    return test_code

        # AST failed - try regex
        print(f"AST extraction FAILED - trying regex...")

        pattern = rf'^(@.*\n)*(?:async\s+)?def\s+{re.escape(base_test_name)}\s*\([^)]*\):.*?(?=\n(?:def\s+|class\s+|@|$))'
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)

        if match:
            test_code = match.group(0)
            lines = test_code.count('\n') + 1
            chars = len(test_code)

            print(f"\n Regex extraction SUCCESS!")
            print(f"      Extracted lines: {lines}")
            print(f"      Extracted chars: {chars:,}")
            print(f"      Est. tokens: ~{chars // 4:,}")
            print(f"      Reduction: {((total_lines - lines) / total_lines * 100):.1f}%")

            return test_code

        # Both failed - use full file
        print(f"\n BOTH extractions FAILED!")
        print(f"      Using FULL FILE - THIS IS THE BLOAT!")
        print(f"      Full file lines: {total_lines}")
        print(f"      Full file chars: {total_chars:,}")
        print(f"      Est. tokens: ~{total_chars // 4:,}")

        return content

    except Exception as e:
        print(f"\n Exception during extraction: {e}")
        print(f"      Using FULL FILE as fallback")
        return content


def extract_source_code(failure, project_root: str) -> str:
    """
    Extract source code using embeddings (SAME logic as orchestrator.py).

    This shows what `source_code` variable contains.
    """
    print("\n" + "=" * 80)
    print("STEP 3: Extracting SOURCE CODE from embeddings (source_code variable)")
    print("=" * 80)

    print(f"\n Initializing embedding context extractor...")
    print(f"   Project root: {project_root}")

    extractor = EmbeddingContextExtractor(
        project_root=project_root,
        max_source_lines=300,
        use_embeddings=True,
        verbose=True
    )

    print(f"\n Extracting context for failure...")
    print(f"   Test file: {failure.test_file}")
    print(f"   Test name: {failure.test_name}")
    print(f"   Error: {failure.error_message[:100]}...")

    source_code = extractor.get_full_context_string(
        failure.test_file,
        failure.test_name,
        failure.error_message
    )

    lines = source_code.count('\n') + 1
    chars = len(source_code)

    print(f"\n Source code extracted")
    print(f"   Total lines: {lines}")
    print(f"   Total chars: {chars:,}")
    print(f"   Est. tokens: ~{chars // 4:,}")

    # Count elements
    element_count = source_code.count('# function:') + source_code.count('# class:') + source_code.count('# http_endpoint:')
    print(f"   Elements included: {element_count}")

    return source_code


def build_llm_prompt(failure, test_code: str, source_code: str,
                     previous_fix_attempt: Optional[str] = None,
                     previous_failure_output: Optional[str] = None) -> str:
    """
    Build the EXACT prompt that gets sent to the LLM.

    This is the SAME logic as llm_fixer.py _build_prompt().
    """
    print("\n" + "=" * 80)
    print("STEP 4: Building FULL LLM PROMPT")
    print("=" * 80)

    # Component 1: Static header
    prompt = f"""# Fix This Failing Test

## Original Test Code
```python
{test_code}
```

## Error Information
**Exception:** {failure.exception_type}
**Message:** {failure.error_message}

## Traceback
```
{failure.traceback}
```

## Source Code Being Tested
```python
{source_code}
```
"""

    # Component 2: Previous attempts (if any)
    if previous_fix_attempt and previous_failure_output:
        prompt += f"""
## Previous Fix Attempt (Failed)
```python
{previous_fix_attempt}
```

## Why the Previous Fix Failed
When we ran pytest on the above fix, it still failed with this output:

```
{previous_failure_output[:2000]}
```

**IMPORTANT:** Analyze WHY this fix failed:
- Is the API key dependency still not being handled?
- Are there other dependencies or fixtures that need to be mocked?
- Is the mock setup incorrect?
- Are there missing imports?
- Does the test need different assertions?

Generate a NEW fix that addresses these specific failure reasons. Don't repeat the same approach!
"""
    elif previous_fix_attempt:
        prompt += f"""
## Previous Fix Attempt (Failed)
```python
{previous_fix_attempt}
```

The previous fix attempt failed. Try a different approach.
"""

    # Component 3: Static instructions
    prompt += """
## Task
Generate a fixed version of the test function that will pass.

**Common patterns to fix:**
1. **Missing API key handling:**
   - Mock the `verify_api_key` dependency
   - Or set `REQUIRE_API_KEY=false` in environment
   - Example: `monkeypatch.setenv("REQUIRE_API_KEY", "false")`

2. **Missing fixture mocking:**
   - Use `monkeypatch.setattr()` to mock required attributes
   - Use `@patch()` decorator for external dependencies
   - Mock database connections, external APIs, etc.

3. **Incorrect imports:**
   - Ensure all required imports are present
   - Use correct module paths

4. **Wrong test setup:**
   - Initialize required fixtures properly
   - Set up test data correctly
   - Clean up after test if needed

Return ONLY the complete fixed test function code (include decorators, docstring, everything).
"""

    return prompt


def analyze_prompt_components(prompt: str, test_code: str, source_code: str,
                               failure, previous_fix: Optional[str],
                               previous_output: Optional[str]) -> None:
    """
    Break down the prompt to show EXACTLY where each line comes from.
    """
    print("\n" + "=" * 80)
    print("STEP 5: DETAILED PROMPT BREAKDOWN")
    print("=" * 80)

    # Calculate component sizes
    test_code_lines = test_code.count('\n') + 1
    test_code_chars = len(test_code)
    test_code_tokens = test_code_chars // 4

    source_code_lines = source_code.count('\n') + 1
    source_code_chars = len(source_code)
    source_code_tokens = source_code_chars // 4

    traceback_lines = failure.traceback.count('\n') + 1
    traceback_chars = len(failure.traceback)
    traceback_tokens = traceback_chars // 4

    error_msg_lines = failure.error_message.count('\n') + 1
    error_msg_chars = len(failure.error_message)
    error_msg_tokens = error_msg_chars // 4

    prev_fix_lines = prev_fix_chars = prev_fix_tokens = 0
    if previous_fix:
        prev_fix_lines = previous_fix.count('\n') + 1
        prev_fix_chars = len(previous_fix)
        prev_fix_tokens = prev_fix_chars // 4

    prev_output_lines = prev_output_chars = prev_output_tokens = 0
    if previous_output:
        prev_output_truncated = previous_output[:2000]
        prev_output_lines = prev_output_truncated.count('\n') + 1
        prev_output_chars = len(prev_output_truncated)
        prev_output_tokens = prev_output_chars // 4

    # Static content (headers, markdown, instructions)
    static_lines = 60
    static_chars = static_lines * 60  # Rough estimate
    static_tokens = static_chars // 4

    # Total
    total_prompt_lines = prompt.count('\n') + 1
    total_prompt_chars = len(prompt)
    total_prompt_tokens = total_prompt_chars // 4

    print("\n COMPONENT BREAKDOWN:\n")

    print(f"{'Component':<30} {'Lines':>8} {'Chars':>12} {'Tokens':>10} {'%':>6}")
    print("-" * 80)

    print(f"{'1. TEST CODE (test_code)':<30} {test_code_lines:8,} {test_code_chars:12,} {test_code_tokens:10,} {test_code_lines/total_prompt_lines*100:5.1f}%")
    print(f"{'2. SOURCE CODE (embeddings)':<30} {source_code_lines:8,} {source_code_chars:12,} {source_code_tokens:10,} {source_code_lines/total_prompt_lines*100:5.1f}%")
    print(f"{'3. TRACEBACK':<30} {traceback_lines:8,} {traceback_chars:12,} {traceback_tokens:10,} {traceback_lines/total_prompt_lines*100:5.1f}%")
    print(f"{'4. ERROR MESSAGE':<30} {error_msg_lines:8,} {error_msg_chars:12,} {error_msg_tokens:10,} {error_msg_lines/total_prompt_lines*100:5.1f}%")

    if prev_fix_lines > 0:
        print(f"{'5. PREVIOUS FIX ATTEMPT':<30} {prev_fix_lines:8,} {prev_fix_chars:12,} {prev_fix_tokens:10,} {prev_fix_lines/total_prompt_lines*100:5.1f}%")

    if prev_output_lines > 0:
        print(f"{'6. PREVIOUS ERROR OUTPUT':<30} {prev_output_lines:8,} {prev_output_chars:12,} {prev_output_tokens:10,} {prev_output_lines/total_prompt_lines*100:5.1f}%")

    print(f"{'7. STATIC INSTRUCTIONS':<30} {static_lines:8,} {static_chars:12,} {static_tokens:10,} {static_lines/total_prompt_lines*100:5.1f}%")

    print("-" * 80)
    print(f"{'TOTAL PROMPT':<30} {total_prompt_lines:8,} {total_prompt_chars:12,} {total_prompt_tokens:10,} {'100.0%':>6}")

    print("\n" + "=" * 80)
    print("BLOAT ANALYSIS")
    print("=" * 80)

    # Identify bloat sources
    if test_code_lines > 100:
        print(f"\n TEST CODE BLOAT DETECTED!")
        print(f"   Test code is {test_code_lines} lines")
        print(f"   This suggests full test file is being sent, not just the function")
        print(f"   Expected: ~15-30 lines for a single test function")
        print(f"   Actual: {test_code_lines} lines")
        print(f"   Bloat: {test_code_lines - 30} extra lines (~{(test_code_lines - 30) * 4:,} tokens)")
    else:
        print(f"\n Test code size looks good: {test_code_lines} lines")

    if source_code_lines > 500:
        print(f"\n SOURCE CODE BLOAT DETECTED!")
        print(f"   Source code is {source_code_lines} lines")
        print(f"   For a 230-line backend, this suggests full functions are stored")
        print(f"   Expected: ~100-200 lines (smart summaries)")
        print(f"   Actual: {source_code_lines} lines")
        print(f"   Bloat: {source_code_lines - 200} extra lines (~{(source_code_lines - 200) * 4:,} tokens)")
    else:
        print(f"\n Source code size looks good: {source_code_lines} lines")

    # Show biggest contributor
    components = [
        ("Test code", test_code_lines, test_code_tokens),
        ("Source code", source_code_lines, source_code_tokens),
        ("Traceback", traceback_lines, traceback_tokens),
        ("Error message", error_msg_lines, error_msg_tokens),
        ("Previous fix", prev_fix_lines, prev_fix_tokens),
        ("Previous output", prev_output_lines, prev_output_tokens),
        ("Static content", static_lines, static_tokens),
    ]

    biggest = max(components, key=lambda x: x[1])

    print(f"\n BIGGEST CONTRIBUTOR: {biggest[0]}")
    print(f"   Lines: {biggest[1]:,} ({biggest[1]/total_prompt_lines*100:.1f}% of total)")
    print(f"   Tokens: ~{biggest[2]:,}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Trace the full prompt construction to identify token bloat",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--test-file",
        required=True,
        help="Path to test file to run"
    )

    parser.add_argument(
        "--project-root",
        required=True,
        help="Path to project root (for embeddings)"
    )

    parser.add_argument(
        "--previous-fix",
        help="Optional: simulate a retry with previous fix attempt"
    )

    parser.add_argument(
        "--save-prompt",
        help="Optional: save full prompt to this file"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("FULL PROMPT TRACER - Find Token Bloat Source")
    print("=" * 80)
    print(f"\nTest file: {args.test_file}")
    print(f"Project root: {args.project_root}")

    # Step 1: Run pytest
    pytest_output, failures = run_pytest(args.test_file)

    if not failures:
        print("\n No failures found! The tests might all be passing.")
        print("   Try a test file that has failing tests.")
        return

    failure = failures[0]

    # Step 2: Extract test code
    test_code = extract_test_code(failure, args.test_file)

    # Step 3: Extract source code
    source_code = extract_source_code(failure, args.project_root)

    # Step 4: Build prompt
    previous_fix = None
    previous_output = None
    if args.previous_fix:
        with open(args.previous_fix, 'r') as f:
            previous_fix = f.read()
        # Simulate previous output
        previous_output = "Previous attempt still failed with similar error..."

    prompt = build_llm_prompt(failure, test_code, source_code, previous_fix, previous_output)

    # Step 5: Analyze
    analyze_prompt_components(prompt, test_code, source_code, failure, previous_fix, previous_output)

    # Save prompt if requested
    if args.save_prompt:
        with open(args.save_prompt, 'w') as f:
            f.write(prompt)
        print(f"\n Full prompt saved to: {args.save_prompt}")
        print(f"   You can inspect it to see EXACTLY what gets sent to the LLM")

    print("\n" + "=" * 80)
    print("TRACE COMPLETE!")
    print("=" * 80)
    print(f"\nNow you know EXACTLY where the {prompt.count(chr(10)) + 1:,} lines come from!")


if __name__ == "__main__":
    main()