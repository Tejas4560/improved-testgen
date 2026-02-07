# AI Test Generation Pipeline - Complete Deep Dive

> **End-to-end explanation**: From triggering the workflow in a consumer repo to the final output.
> Every file, every operation, in execution order.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Complete Execution Flow](#2-complete-execution-flow)
3. [Phase 1: GitHub Actions Workflow](#3-phase-1-github-actions-workflow)
4. [Phase 2: Pipeline Runner (Shell Orchestrator)](#4-phase-2-pipeline-runner-shell-orchestrator)
5. [Phase 3: Code Analysis](#5-phase-3-code-analysis)
6. [Phase 4: Framework Detection](#6-phase-4-framework-detection)
7. [Phase 5: AI Test Generation](#7-phase-5-ai-test-generation)
8. [Phase 6: Auto-Fixer](#8-phase-6-auto-fixer)
9. [Phase 7: Multi-Iteration Orchestrator](#9-phase-7-multi-iteration-orchestrator)
10. [Phase 8: Reporting and Deployment](#10-phase-8-reporting-and-deployment)
11. [File Reference Map](#11-file-reference-map)
12. [Data Flow Between Files](#12-data-flow-between-files)
13. [Environment Variables Reference](#13-environment-variables-reference)
14. [Output Artifacts](#14-output-artifacts)

---

## 1. High-Level Architecture

```
CONSUMER REPO (e.g., your Flask/FastAPI/Django app)
    │
    │  Triggers workflow (push, PR, or manual)
    ▼
┌─────────────────────────────────────────────────────────────────┐
│              GitHub Actions Workflow (CI/CD Layer)               │
│              .github/workflows/ai-test-pipeline-v2.yml          │
│                                                                 │
│  Clones pipeline repo ──► Clones target repo ──► Runs pipeline  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              pipeline_runner.sh (Main Orchestrator)              │
│                                                                 │
│  Detects project ──► Finds tests ──► Runs coverage ──►         │
│  Generates AI tests ──► Auto-fixes ──► Reports                 │
└──────────┬──────────────────┬───────────────────┬──────────────┘
           │                  │                   │
           ▼                  ▼                   ▼
    ┌──────────────┐  ┌───────────────┐  ┌───────────────────┐
    │  src/gen/     │  │ src/auto_     │  │ multi_iteration_  │
    │  (AI Test     │  │ fixer/        │  │ orchestrator.py   │
    │  Generation)  │  │ (Test Repair) │  │ (Coverage Loop)   │
    └──────┬───────┘  └───────┬───────┘  └───────────────────┘
           │                  │
           ▼                  ▼
    ┌──────────────┐  ┌───────────────┐
    │ Azure OpenAI │  │ pytest        │
    │ (LLM API)    │  │ (Test Runner) │
    └──────────────┘  └───────────────┘
```

---

## 2. Complete Execution Flow

Here is the exact order of operations from start to finish:

```
 1. Consumer repo triggers workflow
 2. GitHub Actions checks out pipeline repo
 3. GitHub Actions clones target (consumer) repo
 4. GitHub Actions sets up Python environment
 5. GitHub Actions measures initial coverage (if manual tests exist)
 6. pipeline_runner.sh starts
 7.   ├── detect_python_root() finds Python backend directory
 8.   ├── Clean all previous artifacts
 9.   ├── Create root conftest.py (sys.path safety net)
10.   ├── detect_manual_tests.py scans for existing tests
11.   ├── IF manual tests found:
12.   │   ├── Copy tests to tests/manual/
13.   │   ├── Install target project dependencies
14.   │   ├── Run pytest with coverage on manual tests
15.   │   ├── IF tests fail → run auto-fixer → re-run
16.   │   ├── Parse coverage.xml → get coverage %
17.   │   ├── IF coverage >= 90% → DONE (exit success)
18.   │   ├── IF coverage < 90%:
19.   │   │   ├── coverage_gap_analyzer.py analyzes gaps
20.   │   │   ├── multi_iteration_orchestrator.py starts
21.   │   │   │   ├── ITERATION 1:
22.   │   │   │   │   ├── Analyze gaps (coverage_gaps.json)
23.   │   │   │   │   ├── python -m src.gen (AI generates tests)
24.   │   │   │   │   │   ├── analyzer.py scans project
25.   │   │   │   │   │   ├── Framework detection
26.   │   │   │   │   │   ├── Build prompts (enhanced_prompt.py)
27.   │   │   │   │   │   ├── Call Azure OpenAI (openai_client.py)
28.   │   │   │   │   │   ├── Post-process code (postprocess.py)
29.   │   │   │   │   │   ├── Fix imports (enhanced_generate.py)
30.   │   │   │   │   │   └── Write test files (writer.py)
31.   │   │   │   │   ├── Run pytest (manual + generated)
32.   │   │   │   │   └── Re-analyze coverage
33.   │   │   │   ├── ITERATION 2 (if coverage still < 90%)
34.   │   │   │   └── ITERATION 3 (if coverage still < 90%)
35.   │   │   ├── IF tests fail → run auto-fixer → re-run
36.   │   │   └── Final coverage report
37.   │   └── Copy tests to target repo, commit, push
38.   ├── IF no manual tests:
39.   │   ├── Install target project dependencies
40.   │   ├── python -m src.gen (single-pass AI generation)
41.   │   ├── Run pytest on generated tests
42.   │   ├── IF tests fail → run auto-fixer → re-run
43.   │   ├── Final coverage report
44.   │   └── Copy tests to target repo, commit, push
45.   └── SonarQube upload (if configured)
46. GitHub Actions extracts metrics
47. GitHub Actions uploads artifacts
48. GitHub Actions pushes tests to branch (optional)
49. GitHub Actions deploys coverage to GitHub Pages (optional)
50. GitHub Actions generates job summary + PR comment
51. GitHub Actions checks thresholds → pass/fail
```

---

## 3. Phase 1: GitHub Actions Workflow

**File**: `.github/workflows/ai-test-pipeline-v2.yml`

### What triggers it

The workflow runs in the **consumer repo** (not the pipeline repo). It can be triggered by:

| Trigger | When |
|---------|------|
| `workflow_dispatch` | Manual trigger from GitHub Actions UI |
| `workflow_call` | Called by another workflow (reusable workflow) |
| `pull_request` | When a PR is opened/updated on main/master |

### Inputs (configurable per run)

| Input | Default | Description |
|-------|---------|-------------|
| `repo_url` | Auto-detect | URL of target repo to test |
| `repo_branch` | Auto-detect | Branch to test |
| `min_coverage` | `90` | Minimum coverage % threshold |
| `delta_threshold` | `0` | Minimum coverage improvement required |
| `deploy_pages` | `true` | Deploy HTML coverage to GitHub Pages |
| `auto_push_tests` | `false` | Push generated tests to a branch |
| `tests_branch` | `ai-generated-tests` | Branch name for generated tests |
| `sonar_project_key` | (empty) | SonarQube project key |

### Secrets required

| Secret | Required | Used by |
|--------|----------|---------|
| `AZURE_OPENAI_API_KEY` | Yes | LLM test generation |
| `AZURE_OPENAI_ENDPOINT` | Yes | LLM API endpoint |
| `AZURE_OPENAI_DEPLOYMENT` | Yes | LLM model deployment name |
| `AZURE_OPENAI_API_VERSION` | No | API version (default: 2024-02-15-preview) |
| `GIT_PUSH_TOKEN` | No | Push tests to remote branch |
| `SONAR_HOST_URL` | No | SonarQube integration |
| `SONAR_TOKEN` | No | SonarQube authentication |

### Step-by-step execution

```
Step 1: Checkout Pipeline Repo
   ├── Clones: Tejas4560/improved-testgen (branch: T1)
   ├── Path: pipeline/
   └── Contains all the AI test generation code

Step 2: Setup Python 3.10
   ├── Installs Python 3.10
   └── Enables pip caching

Step 3: Detect Target Repository
   ├── Auto-detects repo URL from github.repository
   ├── Auto-detects branch from PR head_ref or ref_name
   └── Extracts repo name for display

Step 4: Clone Target Repository
   ├── Full clone (not shallow) to pipeline/target_repo/
   └── Checks out the specified branch

Step 5: Install Pipeline Dependencies
   ├── Creates virtual environment
   ├── pip install -r requirements.txt (pipeline deps)
   └── Ensures pytest, pytest-cov, coverage are installed

Step 6: Measure Initial Coverage
   ├── Searches for test_*.py or *_test.py in target repo
   ├── If found: runs pytest --cov on those tests only
   ├── Parses coverage.xml for line-rate attribute
   └── Stores as COVERAGE_BEFORE output

Step 7: Run Pipeline
   ├── cd pipeline/
   ├── Sets environment variables:
   │   ├── TARGET_ROOT, PYTHONPATH → target repo path
   │   ├── MIN_COVERAGE_THRESHOLD → from inputs
   │   ├── AZURE_OPENAI_* → from secrets
   │   ├── GENERATION_TIMEOUT=600
   │   ├── OPENAI_TIMEOUT=120
   │   └── OPENAI_REQUEST_TIMEOUT=180
   └── bash ./pipeline_runner.sh

Step 8: Extract Metrics
   ├── Parse coverage.xml → COVERAGE_AFTER
   ├── Count tests/generated/*.py → TESTS_GENERATED
   ├── Parse .pytest_*.json → TESTS_PASSED, TESTS_FAILED
   ├── Calculate COVERAGE_DELTA = after - before
   ├── Check: threshold_passed = (after >= min_coverage)
   └── Check: delta_passed = (delta >= delta_threshold)

Step 9: Upload Artifacts
   ├── coverage.xml
   ├── test-results.xml
   ├── htmlcov/ (HTML coverage report)
   ├── tests/generated/ (AI-generated tests)
   └── *.json (analysis files)

Step 10: Push Tests to Branch (if auto_push_tests=true)
   ├── Clone target repo
   ├── Create/checkout ai-generated-tests branch
   ├── Copy tests/generated/* to repo
   ├── Commit with coverage details in message
   └── Force-push to branch

Step 11: Deploy to GitHub Pages (if deploy_pages=true)
   ├── Clone/create gh-pages branch
   ├── Copy htmlcov/ to coverage/run-{number}/
   └── Push to gh-pages

Step 12: Generate Summary
   ├── Markdown table in job summary
   ├── Coverage before/after/delta
   ├── Test counts (generated/passed/failed)
   └── Links to artifacts and reports

Step 13: PR Comment (if triggered by PR)
   ├── Sticky comment with coverage table
   ├── Threshold check results
   └── Links to coverage report

Step 14: Check Thresholds
   ├── If coverage < min_coverage → exit 1 (FAIL)
   ├── If delta < delta_threshold → exit 1 (FAIL)
   └── Otherwise → exit 0 (PASS)
```

---

## 4. Phase 2: Pipeline Runner (Shell Orchestrator)

**File**: `pipeline_runner.sh`

This is the heart of the pipeline. It orchestrates everything that happens after GitHub Actions calls `bash ./pipeline_runner.sh`.

### Step-by-step operations

#### 4.1 Environment Setup (Lines 1-85)

```bash
# Sets strict error handling
set -euo pipefail

# Defines paths
CURRENT_DIR="$(pwd)"        # Pipeline repo root
TARGET_DIR="$(pwd)/target_repo"  # Cloned consumer repo

# Auto-detect Python backend root (for fullstack repos)
detect_python_root()
  # 1. Check repo root for requirements.txt/setup.py/pyproject.toml
  # 2. Check common subdirs: backend/, server/, api/, app/, src/
  # 3. Search 2 levels deep (excluding node_modules, venv, frontend)
  # 4. Fallback: repo root

PYTHON_ROOT=$(detect_python_root "$TARGET_DIR")
TARGET_ROOT="$PYTHON_ROOT"    # The actual Python code location
PYTHONPATH="$PYTHON_ROOT"     # Python import path
```

**Why this matters**: In a fullstack repo like `my-app/frontend/ + my-app/backend/`, the Python code lives in `backend/`. This function finds it automatically.

#### 4.2 Cleanup (Lines 112-138)

```bash
# Removes ALL previous artifacts to ensure clean run:
rm -f .coverage coverage.xml
rm -rf htmlcov/ .pytest_cache/
rm -rf tests/manual tests/generated
rm -f .pytest_*.json coverage_gaps.json iteration_report.json
rm -f manual_test_result.json auto_fixer_report.json
```

#### 4.3 Root Conftest Creation (Lines 140-160)

```bash
create_root_conftest()
  # Creates tests/conftest.py that adds TARGET_ROOT to sys.path
  # This is a safety net so ALL test files can import from the target project
```

**File created**: `tests/conftest.py` with:
```python
import os, sys
_python_root = os.environ.get("TARGET_ROOT", os.environ.get("PYTHONPATH", ""))
if _python_root and _python_root not in sys.path:
    sys.path.insert(0, _python_root)
```

#### 4.4 Manual Test Detection (Lines 167-190)

```bash
python src/detect_manual_tests.py "$TARGET_DIR"
# Reads: manual_test_result.json
# Checks: manual_tests_found = true/false
```

**Calls**: `src/detect_manual_tests.py`

#### 4.5 CASE 1: Manual Tests Found (Lines 200-676)

```
IF manual_tests_found == true:
  │
  ├── Install target project dependencies
  │   └── pip install -r requirements.txt
  │
  ├── Copy tests to tests/manual/ (preserving directory structure)
  │
  ├── Run pytest with coverage
  │   └── pytest tests/manual --cov=PYTHON_ROOT --import-mode=importlib
  │       Outputs: coverage.xml, .pytest_manual.json, test-results.xml
  │
  ├── IF tests fail:
  │   ├── Run auto-fixer: python run_auto_fixer.py --test-dir tests/manual
  │   └── Re-run pytest
  │
  ├── Parse coverage from coverage.xml
  │
  ├── IF coverage >= 90%:
  │   ├── "Quality Gate Passed! No AI test generation needed!"
  │   ├── Upload to SonarQube (if configured)
  │   └── EXIT 0
  │
  └── IF coverage < 90%:
      ├── Run coverage_gap_analyzer.py → coverage_gaps.json
      ├── Set GAP_FOCUSED_MODE=true
      ├── Run multi_iteration_orchestrator.py (up to 3 iterations)
      ├── Run combined tests (manual + generated)
      ├── IF tests fail → auto-fixer → re-run
      ├── Copy AI tests to target repo
      ├── Git commit + push (if GIT_PUSH_TOKEN set)
      ├── Upload to SonarQube (if configured)
      └── check_final_coverage → EXIT 0 or 1
```

#### 4.6 CASE 2: No Manual Tests (Lines 678-902)

```
IF manual_tests_found == false:
  │
  ├── Install target project dependencies
  │
  ├── Run single-pass AI generation:
  │   └── python -m src.gen --target PYTHON_ROOT --outdir tests/generated --force
  │
  ├── Run pytest on generated tests
  │   └── pytest tests/generated --cov=PYTHON_ROOT --import-mode=importlib
  │
  ├── IF tests fail:
  │   ├── Run auto-fixer
  │   └── Re-run pytest
  │
  ├── Copy AI tests to target repo
  ├── Git commit + push
  ├── Upload to SonarQube
  └── check_final_coverage → EXIT 0 or 1
```

---

## 5. Phase 3: Code Analysis

### `src/detect_manual_tests.py`

**When called**: Early in pipeline_runner.sh (line 169)
**Command**: `python src/detect_manual_tests.py "$TARGET_DIR"`

**What it does**:
1. Recursively scans the target repo for test files
2. Looks for files matching: `test_*.py`, `*_test.py`, `conftest.py`
3. Skips: `__pycache__`, `.git`, `venv`, `env`, `node_modules`
4. Finds common test root directory (e.g., `tests/`)
5. Preserves relative paths to avoid import conflicts

**Output**: `manual_test_result.json`
```json
{
  "manual_tests_found": true,
  "test_root": "/path/to/target_repo/tests",
  "manual_test_paths": ["/path/to/tests"],
  "test_files_count": 5,
  "files_by_relative_path": {
    "test_user.py": "/full/path/test_user.py",
    "unit/test_models.py": "/full/path/unit/test_models.py"
  }
}
```

### `src/analyzer.py`

**When called**: Inside `python -m src.gen` (via `enhanced_generate.py`)
**Function**: `analyze_python_tree(root)`

**What it does**:
1. Discovers ALL `.py` files in the target project (skips test directories)
2. Parses each file using Python's AST module
3. Extracts:
   - **Functions**: name, args, decorators, line numbers, file path
   - **Classes**: name, base classes, method count, file path
   - **Methods**: name, class, args, decorators (property, classmethod, staticmethod)
   - **Routes**: HTTP method, path, handler name (Flask/FastAPI/Django)
   - **Imports**: all `import` and `from...import` statements
   - **Django patterns**: models, serializers, views, forms, admin, urls
   - **FastAPI routes**: with tags, status_code, response_model
   - **Project structure**: package names, module paths

**Output**: Analysis dict (also saved as `ast_full_analysis.json`)

### `src/coverage_gap_analyzer.py`

**When called**: After initial manual test run, if coverage < 90%
**Command**: `python src/coverage_gap_analyzer.py --target PYTHON_ROOT --current-dir . --output coverage_gaps.json`

**What it does**:
1. Parses `coverage.xml` (Cobertura format from pytest-cov)
2. For each source file: extracts covered lines and missing lines
3. Uses AST to map missing lines to specific functions/classes
4. Identifies:
   - Files with gaps (sorted by missing line count)
   - Uncovered functions (with line ranges)
   - Uncovered classes (with uncovered methods)
   - Overall coverage percentage

**Output**: `coverage_gaps.json`
```json
{
  "overall_coverage": 45.5,
  "total_statements": 1000,
  "missing_statements": 545,
  "files_with_gaps": {
    "app/models.py": {
      "covered_lines": [1, 2, 5],
      "missing_lines": [10, 15, 20],
      "coverage_percentage": 50.0
    }
  },
  "uncovered_functions": [
    {
      "file": "app/api.py",
      "name": "calculate_score",
      "line_start": 50,
      "line_end": 75,
      "uncovered_lines": [60, 62, 65]
    }
  ],
  "uncovered_classes": [
    {
      "file": "app/models.py",
      "name": "User",
      "uncovered_methods": [{"name": "validate", "uncovered_lines": [32, 34]}]
    }
  ],
  "needs_ai_generation": true
}
```

---

## 6. Phase 4: Framework Detection

### `src/framework_handlers/manager.py`

**When called**: During `analyzer.py`'s analysis and during `enhanced_generate.py`

**Detection order** (first match wins):
```
1. DjangoHandler   → checks for: django imports, manage.py, settings.py, urls.py
2. FastAPIHandler  → checks for: fastapi imports, @app.get(), APIRouter, uvicorn
3. FlaskHandler    → checks for: flask imports, @app.route(), Flask(__name__)
4. UniversalHandler → always matches (fallback)
```

**Multi-framework conflict resolution** (`_resolve_conflicts()`):
- If multiple frameworks detected (e.g., hybrid repo):
  - Django wins if: `manage.py` or `settings.py` present
  - FastAPI wins if: `FastAPI()` or `APIRouter` found in code
  - Flask wins if: `@app.route` or `Flask(__name__)` found
  - Tie-breaker: alphabetical sort (deterministic)

### Framework Handlers

| File | Framework | Detection Signals |
|------|-----------|-------------------|
| `django_handler.py` | Django | `django.*` imports, `manage.py`, `settings.py`, `urls.py`, `wsgi.py` |
| `fastapi_handler.py` | FastAPI | `fastapi.*` imports, `@app.get()`, `APIRouter`, `uvicorn`, `main.py` with `FastAPI()` |
| `flask_handler.py` | Flask | `flask.*` imports, `@app.route()`, `Flask(__name__)`, blueprints |
| `universal_handler.py` | Any Python | Always matches. Extracts functions, classes, async patterns |

### What each handler provides

| Handler | `can_handle()` | `generate_framework_specific_tests()` | Key fixtures |
|---------|---------------|---------------------------------------|-------------|
| Django | Checks imports + files | Import smoke tests, model tests, URL tests, serializer tests | `client`, `rf`, `user`, `admin_user`, `db` |
| FastAPI | Checks imports + patterns | Per-route async tests, OpenAPI probe, lifespan smoke | `client` (TestClient), `async_client`, `event_loop` |
| Flask | Checks imports + decorators | Per-route tests, blueprint smoke tests | `app`, `client`, `reset_app_state` |
| Universal | Always true | Function-level tests, class instantiation, module import smoke | `sample_data`, `mock_request` |

---

## 7. Phase 5: AI Test Generation

This is the core of the pipeline. Here's exactly what happens when tests are generated.

### Entry Point: `src/gen/__main__.py`

**Command**: `python -m src.gen --target PYTHON_ROOT --outdir tests/generated --force`

```python
# __main__.py simply wraps enhanced_generate.main() with error handling
from .enhanced_generate import main
main()  # Parses CLI args, runs generation
```

### Main Orchestrator: `src/gen/enhanced_generate.py`

This is the largest and most important file. Here's its execution flow:

```
main()
  ├── Parse CLI arguments:
  │   ├── --target: Path to Python project
  │   ├── --outdir: Output directory (tests/generated)
  │   ├── --coverage-mode: normal/maximum/gap-focused
  │   └── --force: Force regeneration
  │
  ├── Call analyzer.analyze_python_tree(target_root)
  │   └── Returns: full analysis dict
  │
  └── Call generate_all(analysis, outdir)
```

### `generate_all()` - The Core Generation Loop

```
generate_all(analysis, outdir)
  │
  ├── STEP 1: Export analysis
  │   └── Save ast_full_analysis.json
  │
  ├── STEP 2: Gap-Aware Filtering (if GAP_FOCUSED_MODE=true)
  │   ├── gap_aware_analysis.py loads coverage_gaps.json
  │   ├── Filters analysis to ONLY uncovered functions/classes
  │   ├── If coverage >= 90% → skip_generation=true → return
  │   └── Save ast_gap_analysis.json
  │
  ├── STEP 3: Detect Framework
  │   └── _detect_framework(analysis) → "flask"|"fastapi"|"django"|"python"|"universal"
  │       Static detection only (no dynamic imports):
  │       - Checks import list for framework names
  │       - Checks for web framework patterns
  │       - No web framework found → "python"
  │
  ├── STEP 4: Generate Conftest
  │   └── conftest_text.py generates framework-specific conftest.py
  │       Written to: outdir/conftest.py
  │
  ├── STEP 5: Determine Test Kinds
  │   ├── Plain Python → ["unit"] only
  │   └── Web frameworks → ["unit", "integ", "e2e"]
  │
  ├── STEP 6: For EACH test kind:
  │   ├── Calculate targets_count(compact, kind, framework)
  │   │   ├── Plain Python: ALL functions + classes + methods
  │   │   ├── Flask unit: pure functions only (routes → integration)
  │   │   ├── FastAPI unit: pure functions + routes (TestClient!)
  │   │   └── Integration/E2E: routes + complex functions
  │   │
  │   ├── Calculate files_per_kind(compact, kind, framework)
  │   │   ├── Plain Python: one test file per source file
  │   │   └── Web: targets / 50 for unit, targets / 30 for integ
  │   │
  │   └── For EACH file shard (1..N):
  │       │
  │       ├── Focus Selection
  │       │   └── focus_for(compact, kind, shard_idx, total_shards)
  │       │       Returns: target function/class names for this file
  │       │
  │       ├── Context Gathering
  │       │   └── _gather_universal_context(target_root, analysis, focus_names)
  │       │       Reads actual source files (up to 120KB)
  │       │       Includes: imports, functions, classes, routes
  │       │
  │       ├── Prompt Building
  │       │   └── enhanced_prompt.py::build_prompt(kind, compact, focus, context, framework)
  │       │       See "Prompt Building" section below
  │       │
  │       ├── LLM Generation
  │       │   └── _generate_with_universal_retry(messages, max_attempts=5)
  │       │       ├── Create Azure OpenAI client (openai_client.py)
  │       │       ├── Call chat completion with retry
  │       │       │   Backoff delays: [0, 2, 4, 8, 16] seconds
  │       │       │   Handles: RateLimitError, Timeout, ConnectionError
  │       │       ├── Extract Python code from response
  │       │       ├── Validate with ast.parse()
  │       │       └── Optimize for coverage
  │       │
  │       ├── Import Fixing
  │       │   └── _fix_imports_for_universal_compatibility(code, target_root, analysis)
  │       │       ├── Scan target dir for available modules
  │       │       ├── Find actual app module (main.py vs app.py)
  │       │       ├── Fix: "from app import app" → "from main import app"
  │       │       └── Add sys.path.insert(0, target_root)
  │       │
  │       ├── Post-Processing
  │       │   ├── _sanitize_parametrize_signature_mismatches(code)
  │       │   └── postprocess.py::cleanup_generated_test(code, framework)
  │       │       ├── Remove fake app definitions (app = Flask())
  │       │       ├── Remove HTML assertions
  │       │       ├── Remove duplicate fixtures (already in conftest)
  │       │       ├── Remove wrong framework markers
  │       │       ├── Fix FastAPI error codes (400 → 422)
  │       │       └── Fix global state patterns
  │       │
  │       └── File Writing
  │           └── writer.py::write_text(file_path, content)
  │               ├── Normalize whitespace
  │               ├── Validate/fix syntax
  │               ├── Add professional header
  │               └── Write to disk (UTF-8)
  │
  ├── STEP 7: Post-Process All Files
  │   └── postprocess_all_tests(outdir, framework)
  │
  └── STEP 8: Update Manifest
      └── writer.py::update_manifest() → _manifest.json
```

### Prompt Building: `src/gen/enhanced_prompt.py`

This file constructs the messages sent to the LLM. The prompt includes:

```
SYSTEM MESSAGE:
  └── Framework-specific system prompt
      ├── Flask: "Use Flask's test_client()..."
      ├── Django: "Use Django's test Client or RequestFactory..."
      ├── FastAPI: "Use FastAPI's TestClient..."
      ├── Python: "Generate BRANCH-AWARE pytest test code..."
      └── Universal: "Generate comprehensive pytest test code..."

USER MESSAGE (assembled in order):
  ├── 1. Framework designation ("Framework: fastapi")
  ├── 2. Observable behavior rules (what to assert, what not to)
  ├── 3. Strict test categories (unit vs integ vs e2e definitions)
  ├── 4. Test category guidance (framework-specific for this kind)
  ├── 5. Never-do rules (absolute prohibitions)
  ├── 6. Framework-specific rules
  ├── 7. Anti-patterns (what to avoid)
  ├── 8. Framework scaffold (example test code)
  ├── 9. Gap-focused context (if enabled - which lines are uncovered)
  ├── 10. Project analysis (compact JSON of all targets)
  └── 11. Additional context (actual source code, up to 120KB)
```

**Plain Python special additions**:
- `BRANCH_AWARE_RULES`: Test BOTH branches of every if/else
- `PLAIN_PYTHON_MOCKING_RULES`: NO mocking of pure functions
- `EXCEPTION_TEST_RULES`: pytest.raises() for every exception path

**Key constants**:
```
OBSERVABLE_BEHAVIOR_RULES:
  CAN assert: status codes, JSON structure, field presence, list lengths
  NEVER assert: HTML content, titles, emojis, CSS, exact error text

NEVER_DO_RULES:
  1. NEVER create fake apps (app = Flask(__name__))
  2. NEVER assert HTML content
  3. NEVER manipulate global state directly
  4. NEVER assume test execution order
  5. NEVER redefine conftest fixtures
  6. NEVER use wrong framework markers
```

### LLM Client: `src/gen/openai_client.py`

**Creates**: Azure OpenAI client with:
```python
AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2023-12-01-preview",
    timeout=120s (connect: 30s),
    max_retries=2
)
```

**Retry logic**: Delays of [1, 3, 6] seconds + one final attempt
- Retries: RateLimitError, Timeout, ConnectionError
- Fails fast: 400/401/403 (auth errors)

### Post-Processing: `src/gen/postprocess.py`

Validates and cleans generated code:

| Function | What it does |
|----------|-------------|
| `validate_code()` | AST parse, check for `def test_`, non-empty |
| `extract_python_only()` | Remove markdown ``` blocks |
| `remove_fake_app_definitions()` | Comment out `app = Flask()` |
| `remove_html_assertions()` | Comment out `assert "<title>"` |
| `remove_duplicate_fixtures()` | Remove fixtures already in conftest |
| `remove_wrong_framework_markers()` | Strip `@pytest.mark.django_db` from Flask |
| `fix_fastapi_error_codes()` | Change `== 400` to `== 422` for validation |
| `detect_over_mocking()` | Warn about excessive MagicMock (plain Python) |
| `validate_imports_resolve()` | Check imports can be resolved (plain Python) |

### Conftest Generation: `src/gen/conftest_text.py`

Generates framework-specific `conftest.py`:

| Framework | Key Fixtures | Special Setup |
|-----------|-------------|---------------|
| Flask | `app`, `client`, `reset_app_state`, `sample_data` | App context, CSRF disabled |
| Django | `client`, `rf`, `user`, `admin_user`, `authenticated_client` | Auto `@pytest.mark.django_db`, SQLite in-memory |
| FastAPI | `app`, `client`, `async_client`, `override_dependencies` | Event loop, dependency injection |
| Python | `sample_data`, `edge_case_inputs` | Branch/exception markers only |
| Universal | Auto-detect framework, `app`, `client`, `mock_request` | Tries Flask then FastAPI |

### Gap-Aware Filtering: `src/gen/gap_aware_analysis.py`

When `GAP_FOCUSED_MODE=true`:
1. Loads `coverage_gaps.json`
2. Filters analysis to only uncovered functions/classes
3. Adds coverage context to prompts:
   ```
   "Current coverage: 45.5% (target: 90%)
    Uncovered: app.py:calculate_score (lines 60-65)
    Uncovered: models.py:User.validate (lines 32-34)"
   ```
4. If coverage >= 90% → returns `skip_generation: true`

### Writer: `src/gen/writer.py`

Writes test files to disk with:
1. Professional header with timestamp
2. Normalized whitespace (tabs → spaces)
3. Validated syntax (AST parse)
4. Organized imports (stdlib → third-party → local)
5. Manifest tracking (`_manifest.json`)

---

## 8. Phase 6: Auto-Fixer

**When called**: After pytest run if any tests fail
**Command**: `python run_auto_fixer.py --test-dir tests/generated --project-root PYTHON_ROOT --max-iterations 3`

### Entry Point: `run_auto_fixer.py`

```python
from src.auto_fixer import AutoTestFixerOrchestrator
orchestrator = AutoTestFixerOrchestrator(test_dir, project_root, max_iterations)
result = orchestrator.run()
# Exit 0 if all fixed, exit 1 if code bugs or unfixable
```

### Auto-Fixer Flow: `src/auto_fixer/orchestrator.py`

```
Iteration Loop (max 3):
  │
  ├── Run pytest with JSON report
  │   └── failure_parser.py parses failures
  │
  ├── For EACH failure:
  │   │
  │   ├── CLASSIFY: What went wrong?
  │   │   ├── rule_classifier.py (fast, pattern-based)
  │   │   │   Checks 20+ patterns:
  │   │   │   - ImportError → test_mistake
  │   │   │   - fixture not found → test_mistake
  │   │   │   - AttributeError → test_mistake
  │   │   │   - SyntaxError → test_mistake
  │   │   │   - Unknown → pass to LLM
  │   │   │
  │   │   └── llm_classifier.py (if rule-based says "unknown")
  │   │       Sends test code + source code + error to LLM
  │   │       Returns: "test_mistake" or "code_bug" + reason
  │   │
  │   ├── IF code_bug: Skip (can't auto-fix real bugs)
  │   │
  │   ├── IF test_mistake:
  │   │   │
  │   │   ├── EXTRACT CONTEXT:
  │   │   │   └── ast_context_extractor.py
  │   │   │       ├── Parse test file imports
  │   │   │       ├── Resolve imports → source files
  │   │   │       ├── Build source map (all definitions with line numbers)
  │   │   │       ├── Parse error traceback
  │   │   │       ├── Find HTTP endpoints in test code
  │   │   │       └── Extract relevant code (up to 200 lines per file)
  │   │   │
  │   │   ├── GENERATE FIX (up to 3 attempts):
  │   │   │   └── llm_fixer.py
  │   │   │       Attempt 1: Send test + error + source context → LLM
  │   │   │       Attempt 2: Send above + "previous fix failed because: ..."
  │   │   │       Attempt 3: Send above + "try a completely different approach"
  │   │   │
  │   │   └── APPLY FIX:
  │   │       └── ast_patcher.py
  │   │           ├── Find function in AST
  │   │           ├── Replace function body (preserve indentation)
  │   │           ├── Remove duplicate decorators
  │   │           ├── Validate syntax
  │   │           ├── Write to temp file → run pytest → check
  │   │           ├── IF passes: commit fix
  │   │           └── IF fails: restore original, return feedback
  │   │
  │   └── Track result in fix_history
  │
  ├── Check: any fixes applied this iteration?
  │   ├── YES → continue to next iteration
  │   └── NO → stop (no progress possible)
  │
  └── Generate summary: auto_fixer_report.json
```

### Auto-Fixer Files Summary

| File | Role |
|------|------|
| `orchestrator.py` | Coordinates the fix loop |
| `failure_parser.py` | Parses pytest output into structured failures |
| `rule_classifier.py` | Fast pattern-based classification (20+ rules) |
| `llm_classifier.py` | LLM-based classification for ambiguous cases |
| `ast_context_extractor.py` | Extracts relevant source code for LLM context |
| `embedding_context_extractor.py` | Semantic similarity-based context extraction |
| `llm_fixer.py` | Generates fixed test code using LLM |
| `ast_patcher.py` | Applies fixes safely with validation and rollback |
| `codebase_indexer.py` | Indexes codebase for fast lookup |
| `semantic_code_retriever.py` | Retrieves similar code patterns |

---

## 9. Phase 7: Multi-Iteration Orchestrator

**File**: `multi_iteration_orchestrator.py`

**When called**: When manual tests exist but coverage < 90%
**Command**: `python multi_iteration_orchestrator.py --target PYTHON_ROOT --iterations 3 --target-coverage 90 --outdir tests/generated`

### Iteration Flow

```
Initial State:
  coverage_gaps.json exists (from coverage_gap_analyzer.py)
  Manual tests in tests/manual/
  No generated tests yet

ITERATION 1:
  ├── Read current coverage from coverage_gaps.json → e.g., 45%
  ├── Analyze gaps (what's uncovered)
  ├── Generate AI tests: python -m src.gen --coverage-mode gap-focused
  │   └── AI focuses on uncovered code only
  ├── Run ALL tests: pytest tests/manual tests/generated --cov
  │   └── Produces new coverage.xml
  ├── Re-analyze coverage: coverage_gap_analyzer.py
  │   └── Updates coverage_gaps.json with new coverage
  └── Coverage now: 65% (gained 20%)

ITERATION 2:
  ├── Read current coverage: 65%
  ├── Analyze remaining gaps
  ├── Generate MORE AI tests (targeting remaining uncovered code)
  ├── Run ALL tests (manual + iter1 + iter2)
  ├── Re-analyze coverage
  └── Coverage now: 82% (gained 17%)

ITERATION 3:
  ├── Read current coverage: 82%
  ├── Analyze remaining gaps
  ├── Generate MORE AI tests
  ├── Run ALL tests (manual + iter1 + iter2 + iter3)
  ├── Re-analyze coverage
  └── Coverage now: 91% (gained 9%) → TARGET REACHED!

Output: iteration_report.json
```

### Stopping Conditions

1. **Target reached**: Coverage >= 90% → stop, success
2. **Max iterations**: Completed 3 iterations → stop
3. **No improvement**: Coverage gain <= 0 in an iteration → continue but warn

---

## 10. Phase 8: Reporting and Deployment

### GitHub Actions Post-Pipeline (back in workflow)

#### Metrics Extraction
```yaml
# Parse final coverage
COVERAGE_AFTER=$(python3 -c "parse coverage.xml → line-rate")
COVERAGE_DELTA=$((COVERAGE_AFTER - COVERAGE_BEFORE))

# Count test files
TESTS_GENERATED=$(find tests/generated -name 'test_*.py' | wc -l)

# Parse pytest JSON for pass/fail
TESTS_PASSED=$(python3 -c "parse .pytest_*.json → passed count")
TESTS_FAILED=$(python3 -c "parse .pytest_*.json → failed count")
```

#### Artifacts Uploaded
| Artifact | Contents |
|----------|----------|
| `coverage-report` | `coverage.xml`, `htmlcov/` |
| `test-results` | `test-results.xml`, `.pytest_*.json` |
| `generated-tests` | `tests/generated/*.py` |
| `analysis-files` | `ast_*.json`, `coverage_gaps.json`, `iteration_report.json` |

#### GitHub Pages Deployment
```
gh-pages branch:
  └── coverage/
      ├── run-1/   (htmlcov from run #1)
      ├── run-2/   (htmlcov from run #2)
      └── run-N/   (latest)
```

URL: `https://{owner}.github.io/{repo}/coverage/run-{number}/`

#### PR Comment (if triggered by PR)
```markdown
## AI Test Generation Results

| Metric | Value |
|--------|-------|
| Coverage Before | 45.50% |
| Coverage After | 91.20% |
| Coverage Delta | +45.70% |
| Tests Generated | 12 |
| Tests Passed | 11 |
| Tests Failed | 1 |

### Threshold Check
| Check | Result |
|-------|--------|
| Coverage >= 90% | PASS |
| Delta >= 0% | PASS |
```

#### SonarQube Upload (if configured)
```bash
sonar-scanner \
  -Dsonar.projectKey=... \
  -Dsonar.sources=$TARGET_DIR \
  -Dsonar.tests=$TARGET_DIR/tests/ \
  -Dsonar.python.coverage.reportPaths=coverage.xml
```

---

## 11. File Reference Map

### Complete file list with execution order

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `.github/workflows/ai-test-pipeline-v2.yml` | CI/CD | GitHub Actions workflow, triggers everything |
| 2 | `pipeline_runner.sh` | Orchestration | Main bash orchestrator, runs all phases |
| 3 | `src/detect_manual_tests.py` | Analysis | Finds existing test files in target repo |
| 4 | `src/analyzer.py` | Analysis | AST-based code analysis of target project |
| 5 | `src/coverage_gap_analyzer.py` | Analysis | Parses coverage.xml, identifies uncovered code |
| 6 | `src/framework_handlers/manager.py` | Detection | Framework detection orchestrator |
| 7 | `src/framework_handlers/base_handler.py` | Detection | Abstract base class for handlers |
| 8 | `src/framework_handlers/django_handler.py` | Detection | Django-specific detection and test templates |
| 9 | `src/framework_handlers/fastapi_handler.py` | Detection | FastAPI-specific detection and test templates |
| 10 | `src/framework_handlers/flask_handler.py` | Detection | Flask-specific detection and test templates |
| 11 | `src/framework_handlers/universal_handler.py` | Detection | Fallback handler for any Python project |
| 12 | `src/gen/__init__.py` | Generation | Module exports: `generate_all()`, `main()` |
| 13 | `src/gen/__main__.py` | Generation | CLI entry: `python -m src.gen` |
| 14 | `src/gen/env.py` | Generation | Environment configuration and defaults |
| 15 | `src/gen/enhanced_generate.py` | Generation | **Core**: orchestrates test generation |
| 16 | `src/gen/enhanced_prompt.py` | Generation | Builds framework-aware LLM prompts |
| 17 | `src/gen/openai_client.py` | Generation | Azure OpenAI API client with retry |
| 18 | `src/gen/gap_aware_analysis.py` | Generation | Filters analysis to uncovered code only |
| 19 | `src/gen/postprocess.py` | Generation | Validates and cleans generated test code |
| 20 | `src/gen/conftest_text.py` | Generation | Generates framework-specific conftest.py |
| 21 | `src/gen/writer.py` | Generation | Writes test files to disk with formatting |
| 22 | `multi_iteration_orchestrator.py` | Iteration | Runs up to 3 generation cycles for 90%+ |
| 23 | `run_auto_fixer.py` | Fixing | CLI wrapper for auto-fixer |
| 24 | `src/auto_fixer/orchestrator.py` | Fixing | Coordinates test failure repair |
| 25 | `src/auto_fixer/failure_parser.py` | Fixing | Parses pytest failures into structured data |
| 26 | `src/auto_fixer/rule_classifier.py` | Fixing | Fast pattern-based failure classification |
| 27 | `src/auto_fixer/llm_classifier.py` | Fixing | LLM-based failure classification |
| 28 | `src/auto_fixer/ast_context_extractor.py` | Fixing | Extracts source code context for fixes |
| 29 | `src/auto_fixer/llm_fixer.py` | Fixing | Generates fixed test code via LLM |
| 30 | `src/auto_fixer/ast_patcher.py` | Fixing | Applies fixes with validation + rollback |
| 31 | `src/auto_fixer/codebase_indexer.py` | Fixing | Indexes codebase for fast lookup |
| 32 | `src/auto_fixer/semantic_code_retriever.py` | Fixing | Semantic similarity code search |
| 33 | `src/auto_fixer/embedding_context_extractor.py` | Fixing | Embedding-based context extraction |
| 34 | `src/test_generation/orchestrator.py` | Generation | Alternative orchestrator (modular path) |
| 35 | `src/test_generation/import_resolver.py` | Generation | Import classification and dependency install |
| 36 | `src/test_generation/coverage_optimizer.py` | Generation | Prioritizes targets by coverage potential |
| 37 | `pytest.ini` | Config | Pytest configuration |
| 38 | `requirements.txt` | Config | Pipeline Python dependencies |
| 39 | `sonar-project.properties` | Config | SonarQube integration settings |
| 40 | `.env.example` | Config | Environment variable template |

---

## 12. Data Flow Between Files

```
                    ┌──────────────────┐
                    │  Target Repo     │
                    │  (consumer code) │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ detect_manual_   │──── manual_test_result.json
                    │ tests.py         │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  pytest          │──── coverage.xml
                    │  (manual tests)  │──── .pytest_manual.json
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ coverage_gap_    │──── coverage_gaps.json
                    │ analyzer.py      │
                    └────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │  multi_iteration_           │──── iteration_report.json
              │  orchestrator.py            │
              └──────────────┬──────────────┘
                             │ (calls per iteration)
                             │
                    ┌────────▼─────────┐
                    │  python -m       │
                    │  src.gen         │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────────┐
              │              │                  │
     ┌────────▼──────┐ ┌────▼──────┐ ┌─────────▼────────┐
     │ analyzer.py   │ │ framework │ │ gap_aware_        │
     │               │ │ manager   │ │ analysis.py       │
     └───────┬───────┘ └─────┬─────┘ └─────────┬────────┘
             │               │                  │
             │ analysis      │ framework        │ filtered
             │ dict          │ name             │ analysis
             │               │                  │
     ┌───────▼───────────────▼──────────────────▼────────┐
     │            enhanced_generate.py                    │
     │            (core generation loop)                  │
     └───┬───────────┬───────────┬──────────┬────────────┘
         │           │           │          │
  ┌──────▼──────┐ ┌──▼────────┐ │   ┌──────▼──────┐
  │ enhanced_   │ │ openai_   │ │   │ postprocess │
  │ prompt.py   │ │ client.py │ │   │ .py         │
  │ (build      │ │ (call     │ │   │ (validate   │
  │  prompts)   │ │  LLM)     │ │   │  + clean)   │
  └─────────────┘ └───────────┘ │   └─────────────┘
                                │
                    ┌───────────▼──────────┐
                    │  writer.py           │
                    │  (write test files)  │
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  tests/generated/    │──── test_*.py files
                    │  + conftest.py       │──── _manifest.json
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  pytest              │──── coverage.xml (updated)
                    │  (run all tests)     │──── .pytest_combined.json
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  run_auto_fixer.py   │──── auto_fixer_report.json
                    │  (if tests fail)     │
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Final Output        │
                    │  ├── coverage.xml    │
                    │  ├── htmlcov/        │
                    │  ├── test-results.xml│
                    │  └── tests/generated/│
                    └──────────────────────┘
```

---

## 13. Environment Variables Reference

### Required (for AI generation)

| Variable | Example | Used By |
|----------|---------|---------|
| `AZURE_OPENAI_API_KEY` | `sk-abc123...` | `openai_client.py` |
| `AZURE_OPENAI_ENDPOINT` | `https://my-resource.openai.azure.com/` | `openai_client.py` |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4o` | `openai_client.py` |

### Optional

| Variable | Default | Used By |
|----------|---------|---------|
| `AZURE_OPENAI_API_VERSION` | `2023-12-01-preview` | `openai_client.py` |
| `TARGET_ROOT` | `target_repo/` | `pipeline_runner.sh`, all Python modules |
| `PYTHONPATH` | Same as TARGET_ROOT | Python import resolution |
| `MIN_COVERAGE_THRESHOLD` | `90` | `pipeline_runner.sh` |
| `GENERATION_TIMEOUT` | `600` (10 min) | `pipeline_runner.sh` |
| `OPENAI_TIMEOUT` | `120` (2 min) | `openai_client.py` |
| `OPENAI_CONNECT_TIMEOUT` | `30` | `openai_client.py` |
| `OPENAI_REQUEST_TIMEOUT` | `180` (3 min) | `openai_client.py` |
| `GAP_FOCUSED_MODE` | `false` | `gap_aware_analysis.py` |
| `COVERAGE_GAPS_FILE` | `coverage_gaps.json` | `gap_aware_analysis.py` |
| `TESTGEN_FORCE` | `false` | `enhanced_generate.py` |
| `TESTGEN_DEBUG` | `0` | `env.py` |
| `TESTGEN_MAX_RETRIES` | `3` | `env.py` |
| `TESTGEN_PROMPT_STYLE` | `professional` | `env.py` |
| `SONAR_HOST_URL` | (empty) | `pipeline_runner.sh` |
| `SONAR_TOKEN` | (empty) | `pipeline_runner.sh` |
| `SONAR_PROJECT_KEY` | (empty) | `pipeline_runner.sh` |
| `GIT_PUSH_TOKEN` | (empty) | `pipeline_runner.sh` |
| `GIT_USER_EMAIL` | `ci-bot@example.com` | `pipeline_runner.sh` |
| `GIT_USER_NAME` | `CI Bot` | `pipeline_runner.sh` |

---

## 14. Output Artifacts

### Files Generated During Pipeline Run

| File | Created By | Purpose |
|------|-----------|---------|
| `manual_test_result.json` | `detect_manual_tests.py` | List of found test files |
| `ast_full_analysis.json` | `enhanced_generate.py` | Complete project analysis |
| `ast_gap_analysis.json` | `enhanced_generate.py` | Gap-filtered analysis |
| `coverage_gaps.json` | `coverage_gap_analyzer.py` | Uncovered code details |
| `coverage.xml` | `pytest --cov` | Cobertura coverage report |
| `htmlcov/` | `pytest --cov-report=html` | HTML coverage report |
| `test-results.xml` | `pytest --junitxml` | JUnit test results |
| `.pytest_manual.json` | `pytest --json-report` | Manual test results |
| `.pytest_combined.json` | `pytest --json-report` | Combined test results |
| `.pytest_generated.json` | `pytest --json-report` | Generated test results |
| `iteration_report.json` | `multi_iteration_orchestrator.py` | Multi-iteration summary |
| `auto_fixer_report.json` | `run_auto_fixer.py` | Auto-fix summary |
| `tests/conftest.py` | `pipeline_runner.sh` | Root conftest (sys.path) |
| `tests/generated/conftest.py` | `enhanced_generate.py` | Framework-specific conftest |
| `tests/generated/test_*.py` | `enhanced_generate.py` | AI-generated test files |
| `tests/generated/_manifest.json` | `writer.py` | Generation metadata |

### Final Deliverables

```
Consumer Repo (after pipeline):
  ├── tests/
  │   └── generated/
  │       ├── conftest.py          ← Framework-specific test configuration
  │       ├── test_unit_*.py       ← Unit tests
  │       ├── test_integ_*.py      ← Integration tests
  │       ├── test_e2e_*.py        ← End-to-end tests
  │       └── _manifest.json       ← Generation metadata
  │
  GitHub Pages:
  │   └── coverage/run-N/index.html ← HTML coverage report
  │
  GitHub Actions:
  │   ├── Job Summary              ← Coverage metrics table
  │   ├── PR Comment               ← Coverage delta + thresholds
  │   └── Artifacts                ← All reports downloadable
  │
  SonarQube (optional):
      └── Quality metrics uploaded
```

---

## Summary: One-Line Per File

| File | One-line description |
|------|---------------------|
| `ai-test-pipeline-v2.yml` | GitHub Actions workflow that clones repos, runs pipeline, reports results |
| `pipeline_runner.sh` | Bash orchestrator: detect project → find tests → measure coverage → generate → fix → report |
| `detect_manual_tests.py` | Scans target repo for existing test files |
| `analyzer.py` | AST-based Python code analyzer (functions, classes, routes, imports) |
| `coverage_gap_analyzer.py` | Parses coverage.xml to find uncovered functions/lines |
| `manager.py` | Detects framework (Django/FastAPI/Flask/Universal) using evidence-based matching |
| `django_handler.py` | Django-specific detection and test template generation |
| `fastapi_handler.py` | FastAPI-specific detection and async test generation |
| `flask_handler.py` | Flask-specific detection and test client generation |
| `universal_handler.py` | Fallback handler for any Python project |
| `__main__.py` | CLI entry point: `python -m src.gen` |
| `enhanced_generate.py` | Core engine: analyzes code → builds prompts → calls LLM → post-processes → writes tests |
| `enhanced_prompt.py` | Constructs framework-aware prompts with rules, examples, and anti-patterns |
| `openai_client.py` | Azure OpenAI API client with timeout config and retry logic |
| `gap_aware_analysis.py` | Filters analysis to target only uncovered code for gap-focused generation |
| `postprocess.py` | Validates and cleans generated code (removes fake apps, wrong markers, bad assertions) |
| `conftest_text.py` | Generates framework-specific pytest conftest.py with appropriate fixtures |
| `writer.py` | Writes test files with professional headers, syntax validation, import organization |
| `env.py` | Environment variable management and pipeline configuration |
| `multi_iteration_orchestrator.py` | Runs up to 3 generate→test→analyze cycles to reach 90% coverage |
| `run_auto_fixer.py` | CLI wrapper that triggers intelligent test failure repair |
| `orchestrator.py` (auto_fixer) | Coordinates: parse failures → classify → extract context → generate fix → apply |
| `failure_parser.py` | Parses pytest JSON/text output into structured failure objects |
| `rule_classifier.py` | Fast pattern matching to classify failures as test_mistake or unknown |
| `llm_classifier.py` | LLM-based classification for ambiguous test failures |
| `ast_context_extractor.py` | Extracts relevant source code for LLM context using AST + traceback |
| `llm_fixer.py` | Generates fixed test code using LLM with learning from previous attempts |
| `ast_patcher.py` | Safely applies code fixes with AST manipulation, validation, and rollback |
| `codebase_indexer.py` | Indexes all functions/classes in codebase for fast lookup |
| `semantic_code_retriever.py` | Finds similar code patterns using embeddings |
| `coverage_optimizer.py` | Prioritizes test targets by coverage potential and complexity |
| `import_resolver.py` | Classifies imports (stdlib/local/external) and installs dependencies |
