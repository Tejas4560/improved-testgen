# Auto Test Fixer

Automatically fixes failing tests by distinguishing between test mistakes and code bugs.

## Overview

The Auto Test Fixer analyzes failing pytest tests and:

1. **Classifies failures** into:
   - **Test Mistakes**: Errors in the test code itself (wrong imports, bad fixtures, incorrect assertions, etc.)
   - **Code Bugs**: Errors in the source code being tested

2. **Fixes test mistakes** automatically using:
   - Rule-based pattern matching
   - LLM-powered classification and fix generation
   - AST-based precise code patching

3. **Leaves code bugs untouched** for manual review

## Architecture

### Components

1. **FailureParser** (`failure_parser.py`)
   - Runs pytest with JSON output
   - Parses failures into structured `TestFailure` objects
   - Extracts: test file, test name, exception type, error message, traceback, line number

2. **RuleBasedClassifier** (`rule_classifier.py`)
   - Fast pattern-matching classifier
   - Returns "test_mistake" or "unknown"
   - Handles common patterns: import errors, fixture issues, mock problems, etc.

3. **LLMClassifier** (`llm_classifier.py`)
   - AI-powered deep classification
   - Analyzes test code, source code, and traceback
   - Returns: classification, reason, suggested fix, confidence score

4. **ASTContextExtractor** (`ast_context_extractor.py`)
   - Extracts relevant source code based on test imports
   - Provides context for LLM classification and fixing
   - Intelligently filters out stdlib/third-party imports

5. **LLMFixer** (`llm_fixer.py`)
   - Generates fixed versions of failing test functions
   - Can fix individual functions or entire test files
   - Uses source code context for accurate fixes

6. **ASTPatcher** (`ast_patcher.py`)
   - Precisely replaces failing test functions using AST manipulation
   - Preserves all other code, imports, and formatting
   - Validates patches before writing

7. **AutoTestFixerOrchestrator** (`orchestrator.py`)
   - Coordinates the entire workflow
   - Implements iterative fix-and-rerun loop
   - Stops after max iterations or when all test mistakes are fixed

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Run pytest with JSON output                              │
│    - Collect all test failures                              │
│    - Extract structured failure information                 │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ 2. For each failure:                                        │
│    a. Rule-based classification                             │
│       - Fast pattern matching                               │
│       - Returns "test_mistake" or "unknown"                 │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ 3. If unknown, use LLM classifier:                          │
│    a. Extract AST context from imports                      │
│    b. Send to LLM with failure details                      │
│    c. Get classification + fix suggestion                   │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ 4. If test_mistake:                                         │
│    a. Generate fix using LLM (if not already provided)      │
│    b. Apply fix using AST patcher                           │
│    c. Validate the patched code                             │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ 5. Re-run pytest                                            │
│    - Check if fixes worked                                  │
│    - Identify any new failures                              │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ 6. Repeat until:                                            │
│    - All test mistakes are fixed, OR                        │
│    - Max iterations reached (default: 3)                    │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Command Line

```bash
# Basic usage
python run_auto_fixer.py

# Custom test directory
python run_auto_fixer.py --test-dir tests/generated

# More iterations
python run_auto_fixer.py --max-iterations 5

# With pytest arguments
python run_auto_fixer.py --pytest-args "-v,-x,-k,test_user"
```

### Python API

```python
from src.auto_fixer import AutoTestFixerOrchestrator

# Create orchestrator
orchestrator = AutoTestFixerOrchestrator(
    test_directory="tests",
    project_root=".",
    max_iterations=3
)

# Run the fixer
summary = orchestrator.run(extra_pytest_args=["-v"])

# Check results
print(f"Fixed: {summary['successful_fixes']}")
print(f"Code bugs: {summary['code_bugs']}")
```

## Configuration

The auto-fixer uses environment variables for LLM access:

```bash
# Azure OpenAI (required)
export AZURE_OPENAI_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="gpt-4"
```

## Output

The auto-fixer generates:

1. **Console output**: Real-time progress and results
2. **auto_fixer_report.json**: Detailed JSON report with:
   - Iteration count
   - All failures processed
   - Classification results
   - Fix success/failure
   - List of remaining code bugs

## Examples

### Example 1: Import Error (Test Mistake)

**Original Test:**
```python
def test_user_creation():
    user = User(name="John")  # User not imported
    assert user.name == "John"
```

**Error:**
```
NameError: name 'User' is not defined
```

**Classification:** test_mistake (missing import)

**Fixed Test:**
```python
from models.user import User

def test_user_creation():
    user = User(name="John")
    assert user.name == "John"
```

### Example 2: Wrong Fixture (Test Mistake)

**Original Test:**
```python
def test_api_endpoint(clent):  # Typo: 'clent' instead of 'client'
    response = clent.get("/api/users")
    assert response.status_code == 200
```

**Error:**
```
fixture 'clent' not found
```

**Classification:** test_mistake (fixture typo)

**Fixed Test:**
```python
def test_api_endpoint(client):  # Fixed typo
    response = client.get("/api/users")
    assert response.status_code == 200
```

### Example 3: Logic Error (Code Bug)

**Test:**
```python
def test_calculate_total():
    result = calculate_total([10, 20, 30])
    assert result == 60
```

**Error:**
```
AssertionError: assert 70 == 60
```

**Classification:** code_bug (incorrect implementation)

**Action:** Skipped (requires manual code fix)

## Limitations

- **Max iterations**: Default 3 (configurable)
- **LLM dependency**: Requires Azure OpenAI access
- **Code bugs**: Not fixed automatically (by design)
- **Complex failures**: May require multiple iterations
- **Test framework**: Currently supports pytest only

## Best Practices

1. **Run on CI**: Integrate into your CI/CD pipeline
2. **Review fixes**: Always review auto-generated fixes
3. **Iterate gradually**: Start with max_iterations=1 for safety
4. **Monitor costs**: LLM calls can add up with many failures
5. **Version control**: Commit before running to easily revert

## Troubleshooting

### No fixes applied

- Check LLM credentials are set
- Verify test directory path is correct
- Check pytest can run normally

### Fixes don't work

- Increase max_iterations
- Check auto_fixer_report.json for details
- Review generated fixes manually

### Too many LLM calls

- Use rule_classifier first (it's free)
- Reduce number of failing tests
- Batch similar failures

## License

Part of Tech Demo Project 2
