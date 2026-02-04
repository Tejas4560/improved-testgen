"""
LLM-Based Classifier

Uses LLM to classify test failures and suggest fixes.
"""

import json
from typing import Literal, Optional
from dataclasses import dataclass
from .failure_parser import TestFailure
import sys
import os

# Add parent directory to path to import gen modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


ClassificationType = Literal["test_mistake", "code_bug"]


@dataclass
class LLMClassification:
    """Result from LLM classification."""
    classification: ClassificationType
    reason: str
    fixed_code: Optional[str] = None
    confidence: float = 0.0


class LLMClassifier:
    """
    Uses LLM to classify test failures and suggest fixes.

    Distinguishes between:
    - test_mistake: Error in the test code itself
    - code_bug: Error in the source code being tested
    """

    SYSTEM_PROMPT = """You are an expert test debugging assistant. Your job is to analyze failing tests and determine if the failure is due to:

1. **test_mistake**: The test code itself has an error (wrong imports, bad fixtures, incorrect assertions, wrong test setup, etc.)
2. **code_bug**: The source code being tested has a bug (logic error, wrong implementation, etc.)

When you identify a **test_mistake**, you should also provide the fixed version of the test code.

CRITICAL: You MUST respond with ONLY a valid JSON object, nothing else. No explanations, no reasoning text, no markdown.

Respond with this exact JSON structure:
{
  "classification": "test_mistake" or "code_bug",
  "reason": "Brief explanation of why this classification was chosen",
  "fixed_code": "Fixed version of the failing test function (only for test_mistake, null otherwise)",
  "confidence": 0.8
}

Example valid response:
{"classification": "test_mistake", "reason": "Import error - wrong module path", "fixed_code": "def test_foo():\n    ...", "confidence": 0.9}

Be conservative: if you're unsure, classify as "code_bug" to avoid incorrectly modifying tests."""

    def __init__(self, verbose: bool = False):
        """Initialize LLM classifier."""
        self.verbose = verbose
        self.client = None
        self.using_ollama = False

        # Check if Ollama should be used (local LLM)
        # Only check OLLAMA_MODEL for LLM provider (OLLAMA_HOST is for embeddings)
        ollama_model = os.getenv("OLLAMA_MODEL", "").strip()
        if ollama_model:
            try:
                # Load Ollama client dynamically
                import importlib.util
                from pathlib import Path

                current_file = Path(__file__).resolve()
                ollama_client_path = current_file.parent.parent / 'gen' / 'ollama_client.py'

                spec = importlib.util.spec_from_file_location(
                    "ollama_client_llm_classifier",
                    str(ollama_client_path)
                )
                ollama_module = importlib.util.module_from_spec(spec)
                sys.modules['ollama_client_llm_classifier'] = ollama_module
                spec.loader.exec_module(ollama_module)

                self.client = ollama_module.get_ollama_llm_client()
                self.using_ollama = True
                if verbose:
                    print(f"Using Ollama LLM: {os.getenv('OLLAMA_MODEL', 'deepseek-r1:latest')}")
            except Exception as e:
                if verbose:
                    print(f"Could not initialize Ollama LLM client: {e}")
                    print(f"Falling back to Azure OpenAI")

        # Fall back to Azure OpenAI if Ollama not configured or failed
        if self.client is None:
            try:
                # Load OpenAI client dynamically (avoid gen/__init__.py relative imports)
                import importlib.util
                from pathlib import Path

                current_file = Path(__file__).resolve()
                openai_client_path = current_file.parent.parent / 'gen' / 'openai_client.py'

                spec = importlib.util.spec_from_file_location(
                    "openai_client_llm_classifier",
                    str(openai_client_path)
                )
                openai_module = importlib.util.module_from_spec(spec)
                sys.modules['openai_client_llm_classifier'] = openai_module
                spec.loader.exec_module(openai_module)

                self.client = openai_module.get_openai_client()
                if verbose:
                    print(f"Using Azure OpenAI")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
                self.client = None

    def classify(
        self,
        failure: TestFailure,
        test_code: str,
        source_code: str,
    ) -> LLMClassification:
        """
        Classify a test failure using LLM.

        Args:
            failure: TestFailure object
            test_code: The failing test function code
            source_code: Relevant source code being tested (from AST extraction)

        Returns:
            LLMClassification object
        """
        if not self.client:
            # Fallback classification
            return LLMClassification(
                classification="code_bug",
                reason="LLM client not available",
                confidence=0.0
            )

        # Build the prompt
        user_prompt = self._build_prompt(failure, test_code, source_code)

        # DEBUG: Show prompt size with element breakdown
        prompt_lines = user_prompt.count('\n')
        prompt_chars = len(user_prompt)
        estimated_tokens = prompt_chars // 4  # Rough estimate: 4 chars per token

        # Count elements in source code
        element_count = source_code.count('# function:') + source_code.count('# class:') + source_code.count('# http_endpoint:')

        if self.verbose:
            print(f"Input size:")
            print(f"Prompt: {prompt_lines} lines, {prompt_chars} chars (~{estimated_tokens} tokens)")
            print(f"Elements sent to LLM: {element_count}")
            print(f"Source code: {len(source_code)} chars (~{len(source_code)//4} tokens)")

        try:
            # Call LLM with retry logic
            # Get model name based on provider
            if self.using_ollama:
                model_name = os.getenv("OLLAMA_MODEL", "deepseek-r1:latest")
            else:
                # Azure OpenAI
                model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
                if not model_name:
                    raise ValueError("AZURE_OPENAI_DEPLOYMENT environment variable not set")

            # Build request parameters
            request_params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                # NO max_completion_tokens limit - let reasoning models use what they need
                # (deepseek-r1 generates 8k-15k tokens of reasoning before the answer)
            }

            # Only set temperature if environment variable is set
            # Some Azure deployments don't support custom temperature
            temp = os.getenv("AUTOFIXER_LLM_TEMPERATURE")
            if temp is not None:
                request_params["temperature"] = float(temp)

            # Timing for Ollama models (to show how slow reasoning models are)
            import time
            start_time = time.time()
            if self.verbose and self.using_ollama:
                print(f"Calling {model_name}... (this may take 30s-15min for reasoning models)")

            # Retry logic with exponential backoff
            response = self._call_llm_with_retry(request_params, max_retries=3)

            elapsed_time = time.time() - start_time
            if self.verbose and self.using_ollama:
                mins = int(elapsed_time // 60)
                secs = int(elapsed_time % 60)
                if mins > 0:
                    print(f"Completed in {mins}m {secs}s")

            # Parse response
            content = response.choices[0].message.content.strip()

            # DEBUG: Show raw LLM response for Ollama models (to debug reasoning models)
            if self.verbose and self.using_ollama:
                output_tokens = len(content) // 4  # Estimate
                print(f"Output size:")
                print(f"Response: {len(content)} chars (~{output_tokens} tokens)")
                print(f"Raw LLM response preview (first 300 chars):")
                # Show first 300 chars to see if it's JSON or reasoning text
                preview = content[:300] if len(content) > 300 else content
                print(f"{preview}")
                if len(content) > 300:
                    print(f" ... ({len(content) - 300} more chars)")

            # Extract JSON from response (handle various formats)
            json_str = self._extract_json(content)

            # DEBUG: Show extracted JSON
            if self.verbose and self.using_ollama and json_str != content:
                print(f"Extracted JSON ({len(json_str)} chars):")
                print(f"{json_str[:200]}")

            result = json.loads(json_str)

            return LLMClassification(
                classification=result.get("classification", "code_bug"),
                reason=result.get("reason", "No reason provided"),
                fixed_code=result.get("fixed_code"),
                confidence=result.get("confidence", 0.5)
            )

        except json.JSONDecodeError as e:
            print(f"Error parsing LLM JSON response: {e}")
            if 'content' in locals():
                print(f"Raw response preview (first 400 chars):")
                print(f"{content[:400]}")
                if 'json_str' in locals() and json_str != content:
                    print(f"Extracted JSON attempt:")
                    print(f"{json_str[:400]}")
            # Conservative fallback
            return LLMClassification(
                classification="code_bug",
                reason=f"JSON parse error: {str(e)}",
                confidence=0.0
            )
        except Exception as e:
            print(f"Error in LLM classification: {e}")
            # Conservative fallback
            return LLMClassification(
                classification="code_bug",
                reason=f"Classification failed: {str(e)}",
                confidence=0.0
            )

    def _call_llm_with_retry(self, request_params: dict, max_retries: int = 3):
        """
        Call LLM API with exponential backoff retry logic.

        Args:
            request_params: Parameters for the API call
            max_retries: Maximum number of retry attempts

        Returns:
            API response

        Raises:
            Exception: If all retries fail
        """
        import time

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**request_params)

                # Validate response has content
                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty response from LLM")

                content = response.choices[0].message.content.strip()
                if not content:
                    raise ValueError("Empty content in LLM response")

                return response

            except Exception as e:
                if attempt < max_retries - 1:
                    # Calculate backoff time: 2^attempt seconds (2s, 4s, 8s)
                    backoff_time = 2 ** attempt
                    print(f"LLM API error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {backoff_time}s...")
                    time.sleep(backoff_time)
                else:
                    # Final attempt failed
                    print(f"LLM API failed after {max_retries} attempts: {e}")
                    raise

    def _extract_json(self, content: str) -> str:
        """
        Extract JSON from LLM response, handling various formats.

        Args:
            content: Raw LLM response

        Returns:
            JSON string
        """
        # Try markdown code blocks first
        if "```json" in content:
            return content.split("```json")[1].split("```")[0].strip()

        if "```" in content:
            parts = content.split("```")
            for part in parts[1::2]:  # Every other part (inside code blocks)
                part = part.strip()
                if part.startswith('{') and part.endswith('}'):
                    return part

        # Try to find JSON object directly - improved regex for nested objects
        import re

        # Pattern 1: Look for complete JSON objects with proper nesting
        # This handles nested braces better
        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)

        if matches:
            # Try to parse each match and return the first valid one
            for match in sorted(matches, key=len, reverse=True):
                try:
                    json.loads(match)  # Validate it's real JSON
                    return match
                except:
                    continue

        # Pattern 2: Extract everything between first { and last }
        first_brace = content.find('{')
        last_brace = content.rfind('}')

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            potential_json = content[first_brace:last_brace+1]
            try:
                json.loads(potential_json)  # Validate
                return potential_json
            except:
                pass

        # Pattern 3: Look for JSON-like structure after thinking text
        # Reasoning models often output: "thinking text... {json here}"
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('{'):
                # Try to parse from this line onwards
                potential_json = '\n'.join(lines[i:])
                first_brace = potential_json.find('{')
                last_brace = potential_json.rfind('}')
                if first_brace != -1 and last_brace != -1:
                    candidate = potential_json[first_brace:last_brace+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except:
                        continue

        # Fallback: return as-is and hope it's valid JSON
        return content

    def _build_prompt(
        self,
        failure: TestFailure,
        test_code: str,
        source_code: str
    ) -> str:
        """
        Build the user prompt for classification.

        Args:
            failure: TestFailure object
            test_code: The failing test function code
            source_code: Relevant source code

        Returns:
            Formatted prompt string
        """
        # NO truncation - send all source code (embeddings already filtered to relevant code)
        # Modern LLMs (deepseek-r1: 64k, gpt-4o-mini: 128k) can handle large contexts
        # The AST/embedding extractors already limit to relevant functions only
        source_code_display = source_code
        prompt = f"""# Test Failure Analysis

## Failing Test
**File:** {failure.test_file}
**Test Name:** {failure.test_name}
**Line:** {failure.line_number or 'Unknown'}

## Error Information
**Exception Type:** {failure.exception_type}
**Error Message:** {failure.error_message}

## Traceback
```
{failure.traceback}
```

## Test Code
```python
{test_code}
```

## Source Code Being Tested
```python
{source_code_display}
```

## Task
Analyze this failure and determine:
1. Is this a **test_mistake** (error in test code) or **code_bug** (error in source code)?
2. Why?
3. If it's a test_mistake, provide the fixed test code.

IMPORTANT: Respond with ONLY a valid JSON object. Do not include any explanatory text, reasoning, or markdown formatting. Your entire response must be parseable JSON."""

        # Add extra reminder for Ollama/reasoning models
        if self.using_ollama:
            prompt += "\n\nReminder: Output ONLY the JSON object, nothing else. Start your response with { and end with }"

        return prompt