# src/gen/enhanced_prompt.py - FRAMEWORK-AWARE TEST GENERATION

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple
from .gap_aware_analysis import get_coverage_context_for_prompts, is_gap_focused_mode

# Framework-specific system prompts
SYSTEM_PROMPTS = {
    "flask": (
        "Generate comprehensive pytest test code for a FLASK application.\n"
        "FLASK-SPECIFIC REQUIREMENTS:\n"
        " - Use Flask's test_client() for all HTTP testing\n"
        " - Use app.test_client() from the 'client' fixture\n"
        " - Do NOT use @pytest.mark.django_db - this is Flask, not Django\n"
        " - Do NOT use Django's RequestFactory or QueryDict\n"
        " - For JSON APIs, use client.post('/path', json={...})\n"
        " - Check response.status_code and response.get_json()\n"
        " - Use response.get_data(as_text=True) for HTML responses\n"
        " - Return ONLY Python code, no markdown\n"
    ),
    "django": (
        "Generate comprehensive pytest test code for a DJANGO application.\n"
        "DJANGO-SPECIFIC REQUIREMENTS:\n"
        " - Use Django's test Client or RequestFactory\n"
        " - Mark database tests with @pytest.mark.django_db\n"
        " - Use QueryDict for POST/GET data when needed\n"
        " - Use django.test.Client for view testing\n"
        " - Use RequestFactory for unit testing views\n"
        " - Return ONLY Python code, no markdown\n"
    ),
    "fastapi": (
        "Generate comprehensive pytest test code for a FASTAPI application.\n"
        "FASTAPI-SPECIFIC REQUIREMENTS:\n"
        " - Use FastAPI's TestClient from fastapi.testclient\n"
        " - Do NOT use @pytest.mark.django_db - this is FastAPI, not Django\n"
        " - For async endpoints, use pytest-asyncio or TestClient\n"
        " - Use client.post('/path', json={...}) for JSON APIs\n"
        " - Check response.status_code and response.json()\n"
        " - Return ONLY Python code, no markdown\n"
    ),
    "python": (
        "Generate BRANCH-AWARE pytest test code for PLAIN PYTHON code.\n"
        "PLAIN PYTHON REQUIREMENTS (CRITICAL FOR COVERAGE):\n"
        " - Generate ONE TEST PER BRANCH, not just one test per function\n"
        " - For each if/else: test BOTH branches (true and false conditions)\n"
        " - For each try/except: test BOTH success AND exception paths\n"
        " - For each raise: test with pytest.raises() to cover exception path\n"
        " - For loops: test with empty AND non-empty iterables\n"
        " - For validation functions: test valid AND invalid inputs\n"
        " - Use REAL imports - NO MagicMock unless testing IO/network/time\n"
        " - NEVER mock pure Python logic - execute it directly\n"
        " - Return ONLY Python code, no markdown\n"
    ),
    "universal": (
        "Generate comprehensive pytest test code that works with ANY Python project structure.\n"
        "UNIVERSAL TESTING REQUIREMENTS:\n"
        " - Use REAL imports and REAL code execution whenever possible\n"
        " - Test both success paths AND error conditions\n"
        " - Include edge cases: empty inputs, None values, invalid data\n"
        " - Test ALL public methods, properties, and class attributes\n"
        " - Generate multiple test methods per class/function for maximum coverage\n"
        " - Return ONLY Python code, no markdown\n"
        " - Be completely framework-agnostic and project-structure-agnostic\n"
    )
}

# ============================================================================
# PLAIN PYTHON SPECIFIC CONSTANTS
# ============================================================================

# Branch-aware test generation rules for plain Python
BRANCH_AWARE_RULES = """
🎯 BRANCH-AWARE TEST GENERATION (CRITICAL FOR COVERAGE)

Coverage failures in plain Python are BRANCH failures, not function failures.
Generate tests for EACH BRANCH, not just each function.

BRANCH DETECTION PATTERNS:
┌─────────────────┬───────────────────────────────────────┬─────────────┐
│ Code Pattern    │ What to Test                          │ Min Tests   │
├─────────────────┼───────────────────────────────────────┼─────────────┤
│ if/else         │ Both true AND false conditions        │ 2           │
│ if/elif/else    │ Each branch + else                    │ 3+          │
│ try/except      │ Success path + each except handler    │ 2+          │
│ for/while       │ Empty iterable + non-empty iterable   │ 2           │
│ raise           │ Use pytest.raises() to cover          │ 1           │
│ return early    │ Each return path                      │ 2+          │
│ bool condition  │ True case + False case                │ 2           │
└─────────────────┴───────────────────────────────────────┴─────────────┘

EXAMPLE - ARITHMETIC FUNCTION:
```python
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```
REQUIRED TESTS (minimum 2):
```python
def test_divide_success():
    assert divide(10, 2) == 5.0

def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(10, 0)
```

EXAMPLE - VALIDATION FUNCTION:
```python
def validate_email(email):
    if not email:
        return False
    if "@" not in email:
        return False
    return True
```
REQUIRED TESTS (minimum 3):
```python
def test_validate_email_valid():
    assert validate_email("test@example.com") is True

def test_validate_email_empty():
    assert validate_email("") is False
    assert validate_email(None) is False

def test_validate_email_no_at():
    assert validate_email("invalid") is False
```

EXAMPLE - CLASS WITH CONDITIONAL LOGIC:
```python
class ShoppingCart:
    def __init__(self):
        self.items = []

    def checkout(self):
        if not self.items:
            raise Exception("Cart is empty")
        return sum(item.price for item in self.items)
```
REQUIRED TESTS (minimum 2):
```python
def test_checkout_with_items():
    cart = ShoppingCart()
    cart.items = [Item(price=10), Item(price=20)]
    assert cart.checkout() == 30

def test_checkout_empty_cart():
    cart = ShoppingCart()
    with pytest.raises(Exception):
        cart.checkout()
```
"""

# Anti-mocking rules for plain Python
PLAIN_PYTHON_MOCKING_RULES = """
🚫 MOCKING RULES FOR PLAIN PYTHON (CRITICAL)

DO NOT MOCK unless the function involves:
❌ Network I/O (requests, HTTP calls, APIs)
❌ File system I/O (reading/writing files)
❌ Database connections
❌ Time-dependent operations (datetime.now(), time.sleep())
❌ Random number generation (when determinism needed)
❌ External services or third-party APIs

ALWAYS EXECUTE DIRECTLY (no mocking):
✅ Pure functions (input → output with no side effects)
✅ Data validation functions
✅ Mathematical calculations
✅ String manipulations
✅ List/dict operations
✅ Class instantiation and method calls
✅ Business logic without I/O

WHY: Mocked code does NOT count for coverage!

BAD (coverage killer):
```python
mock = MagicMock()
mock.calculate.return_value = 42  # This line IS NOT TESTED!
```

GOOD (real coverage):
```python
result = calculator.calculate(10, 5)  # This line IS TESTED!
assert result == 15
```
"""

# Exception testing rules
EXCEPTION_TEST_RULES = """
⚠️ EXCEPTION TESTING IS MANDATORY (NOT OPTIONAL)

If a function can raise an exception, you MUST test it:

RULE: Use pytest.raises() for exception paths
RULE: NEVER mix success and exception assertions in the same test

CORRECT PATTERN:
```python
def test_function_success():
    \"\"\"Test normal execution path.\"\"\"
    result = my_function(valid_input)
    assert result == expected_value

def test_function_raises_on_invalid():
    \"\"\"Test exception path.\"\"\"
    with pytest.raises(ValueError):
        my_function(invalid_input)
```

COMMON EXCEPTION TRIGGERS TO TEST:
- Empty input: my_function("")
- None input: my_function(None)
- Zero division: divide(x, 0)
- Index out of range: list[999]
- Key not found: dict["missing_key"]
- Type mismatch: int_function("string")
- Negative values: price(-10)
- Invalid format: parse_date("not-a-date")
"""

# Default for backward compatibility
SYSTEM_MIN = SYSTEM_PROMPTS["universal"]

# ============================================================================
# FASTAPI-SPECIFIC CONSTANTS
# ============================================================================

# FastAPI route decorators for detection
FASTAPI_ROUTE_DECORATORS = {"get", "post", "put", "delete", "patch", "options", "head", "trace"}

# ============================================================================
# STRICT TEST GENERATION RULES (NON-NEGOTIABLE)
# ============================================================================

OBSERVABLE_BEHAVIOR_RULES = """
🔒 OBSERVABLE BEHAVIOR ONLY - NON-NEGOTIABLE RULES

You may ONLY assert things that are GUARANTEED by the backend code:
✅ ALLOWED TO ASSERT:
   - HTTP status codes (200, 201, 400, 404, 500, etc.)
   - JSON response structure (keys that exist in the code)
   - Required fields and their types
   - Error message keys (not exact text unless hardcoded)
   - Content-Type headers
   - List lengths when deterministic
   - ID presence (not specific values)

❌ NEVER ASSERT (will cause false failures):
   - HTML content, titles, or structure (e.g., '<title>Task Manager</title>')
   - Emojis or decorative text (e.g., '✨ Task Manager')
   - Specific UI text that isn't in backend code
   - CSS classes or styling
   - Exact error message wording (unless hardcoded in backend)
   - Timestamps or auto-generated IDs (use 'is not None' instead)
   - Order of items unless explicitly sorted in backend

RULE: If it's not EXPLICITLY in the backend code, don't assert it.
"""

STRICT_TEST_CATEGORIES = """
🎯 STRICT TEST CATEGORIES - WHAT EACH TYPE MUST TEST

UNIT TESTS (test_unit_*.py):
   ✅ Pure functions with no side effects
   ✅ Utility functions
   ✅ Data validation functions
   ✅ Helper methods
   ✅ Class methods that don't require HTTP
   ✅ FastAPI routes (TestClient makes them unit-testable!)
   ❌ Flask routes (those require integration tests)
   ❌ Django views (those require integration tests)

INTEGRATION TESTS (test_integ_*.py):
   ✅ API endpoints via test client
   ✅ Database operations
   ✅ Service interactions
   ✅ Real HTTP request/response cycles
   ✅ Flask routes (must use test_client())
   ❌ NOT creating fake apps - use REAL app

END-TO-END TESTS (test_e2e_*.py):
   ✅ Complete user workflows via API
   ✅ Multi-step operations (create → read → update → delete)
   ✅ Full request/response cycles
   ✅ State built via API calls (not direct manipulation)
   ❌ NOT HTML/UI assertions
   ❌ NOT creating fake apps

⚠️ FRAMEWORK-SPECIFIC:
   - Flask routes → INTEGRATION tests only
   - FastAPI routes → Can be UNIT tested (TestClient)
   - Django views → INTEGRATION tests with @pytest.mark.django_db
"""

NEVER_DO_RULES = """
🚫 ABSOLUTE PROHIBITIONS - NEVER DO THESE

1. NEVER CREATE FAKE APPS:
   ❌ app = Flask(__name__)  # WRONG - creates fake app
   ❌ app = FastAPI()        # WRONG - creates fake app
   ✅ from app import app    # CORRECT - import real app

2. NEVER ASSUME HTML CONTENT:
   ❌ assert '<title>' in response  # WRONG - assumes HTML
   ❌ assert '✨' in response       # WRONG - assumes emoji
   ✅ assert response.status_code == 200  # CORRECT

3. NEVER MANIPULATE GLOBAL STATE DIRECTLY:
   ❌ tasks = []           # WRONG - direct global manipulation
   ❌ tasks.clear()        # WRONG - bypasses app logic
   ✅ client.delete('/api/tasks/1')  # CORRECT - use API

4. NEVER ASSUME EXECUTION ORDER:
   ❌ Test that depends on previous test's state
   ✅ Each test sets up its own state via API calls

5. NEVER REDEFINE CONFTEST FIXTURES:
   ❌ @pytest.fixture def client(): ...  # WRONG - already in conftest
   ✅ def test_foo(client): ...          # CORRECT - use existing

6. NEVER USE WRONG FRAMEWORK MARKERS:
   ❌ @pytest.mark.django_db  # WRONG for Flask/FastAPI
   ✅ Use markers matching detected framework only
"""

# FastAPI-specific error handling rules
FASTAPI_ERROR_RULES = """
⚠️ FASTAPI VALIDATION ERRORS - CRITICAL DIFFERENCE

FastAPI uses Pydantic for automatic validation. This means:

VALIDATION ERRORS → 422 (NOT 400!)
   ❌ assert response.status_code == 400  # WRONG for FastAPI
   ✅ assert response.status_code == 422  # CORRECT for FastAPI validation

ERROR STATUS CODES IN FASTAPI:
   - Missing required field → 422 Unprocessable Entity
   - Wrong type (string instead of int) → 422 Unprocessable Entity
   - Validation constraint failed → 422 Unprocessable Entity
   - Route not found → 404 Not Found
   - Unhandled exception → 500 Internal Server Error
   - Manual HTTPException(400) → 400 Bad Request (only if explicit)

EXAMPLE FASTAPI VALIDATION TEST:
```python
def test_create_item_validation(client):
    # Missing required field - FastAPI returns 422, NOT 400
    res = client.post("/items", json={})
    assert res.status_code == 422  # Pydantic validation error

def test_create_item_wrong_type(client):
    # Wrong type - FastAPI returns 422
    res = client.post("/items", json={"price": "not_a_number"})
    assert res.status_code == 422
```

DO NOT use Flask error patterns in FastAPI tests!
"""

# Universal test templates for any project
UNIT_ENHANCED = (
    "Generate COMPREHENSIVE UNIT tests for ANY Python code:\n"
    "- Test EVERY public method in classes/functions\n"
    "- Test constructor/initialization with various parameters\n"
    "- Test property getters and setters\n"
    "- Test validation methods with valid AND invalid inputs\n"
    "- Test string representations (__str__, __repr__)\n"
    "- Test equality operations (__eq__, __hash__ if present)\n"
    "- Test exception handling and error conditions\n"
    "- Use parametrized tests for multiple input scenarios\n"
    "- Target minimum 80% line coverage per file\n"
    "- Use REAL imports, avoid mocking unless absolutely necessary\n"
)

INTEG_ENHANCED = (
    "Generate COMPREHENSIVE INTEGRATION tests for ANY project:\n"
    "- Test component interactions with REAL implementations\n"
    "- Test complete workflows between modules\n"
    "- Test data flow between different parts of the system\n"
    "- Use real imports and actual code execution\n"
    "- Test both happy paths and error scenarios\n"
    "- Verify integration points work correctly\n"
    "- Avoid mocking internal project components\n"
)

E2E_ENHANCED = (
    "Generate COMPREHENSIVE END-TO-END tests for ANY application:\n"
    "- Test complete user workflows\n"
    "- Test API endpoints with real request/response cycles\n"
    "- Test file operations with temporary files\n"
    "- Test database interactions with test databases\n"
    "- Include both success and failure scenarios\n"
    "- Test response formats, headers, and status codes\n"
    "- Use real application setup and teardown\n"
)

MAX_TEST_FILES = {"unit": 4, "integ": 4, "e2e": 2}  

# Universal scaffold for any Python project
UNIVERSAL_SCAFFOLD = '''
"""
Universal test suite - works with ANY Python project structure.
REAL IMPORTS ONLY - Minimal mocking for maximum coverage.
"""

import pytest
import sys
import os
from unittest.mock import patch, Mock, MagicMock
from typing import Any, Dict, List, Optional

os.environ['COVERAGE_OMIT_PATTERNS'] = 'tests/*,*/wsgi.py,*/asgi.py'

# UNIVERSAL IMPORT SETUP - Works with any project structure
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Universal test utilities
def safe_import(module_path):
    """Safely import any module with comprehensive error handling."""
    try:
        import importlib
        return importlib.import_module(module_path)
    except ImportError as e:
        pytest.skip(f"Module {module_path} not available: {e}")
    except Exception as e:
        pytest.skip(f"Could not import {module_path}: {e}")

def dynamic_import(module_name, class_name=None):
    """Dynamically import modules/classes from ANY project structure."""
    try:
        module = safe_import(module_name)
        if class_name:
            return getattr(module, class_name)
        return module
    except AttributeError:
        pytest.skip(f"Class {class_name} not found in {module_name}")

def create_minimal_stub(**attrs):
    """Create minimal stub only when absolutely necessary."""
    stub = Mock()
    for key, value in attrs.items():
        setattr(stub, key, value)
    return stub

# Universal fixtures for any project
@pytest.fixture
def universal_sample_data():
    """Universal sample data for any Python project."""
    return {
        "string_data": "test value",
        "number_data": 42,
        "list_data": [1, 2, 3],
        "dict_data": {"key": "value"},
        "none_data": None,
        "empty_string": "",
        "empty_list": [],
        "empty_dict": {},
        "boolean_true": True,
        "boolean_false": False,
    }

@pytest.fixture
def temp_file_fixture():
    """Universal temporary file fixture."""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
        f.write('test content')
        temp_path = f.name
    yield temp_path
    # Cleanup
    try:
        os.unlink(temp_path)
    except:
        pass

@pytest.fixture
def mock_external_apis():
    """ONLY mock external APIs, never internal project code."""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post:
        # Setup default responses for external APIs
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {'status': 'ok'}
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'result': 'success'}
        yield {'get': mock_get, 'post': mock_post}

# Async support for any project
@pytest.fixture
def event_loop():
    """Universal event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

# Universal test patterns
def test_with_real_imports(test_function):
    """Decorator to ensure tests use real imports."""
    def wrapper(*args, **kwargs):
        try:
            return test_function(*args, **kwargs)
        except ImportError as e:
            pytest.skip(f"Required import not available: {e}")
    return wrapper

def parametrized_test_cases():
    """Universal parametrized test cases for any project."""
    return [
        ("normal_case", "test_value", True),
        ("empty_case", "", False),
        ("none_case", None, False),
        ("numeric_case", 123, True),
        ("list_case", [1, 2, 3], True),
    ]
'''

def is_fastapi_route(func: Dict[str, Any]) -> bool:
    """
    Detect if a function is a FastAPI route handler via decorators.

    FastAPI routes are identified by decorators like:
    - @app.get("/path")
    - @app.post("/path")
    - @router.get("/path")
    - @router.post("/path")

    Returns:
        True if the function is a FastAPI route handler
    """
    decorators = func.get("decorators", [])
    for dec in decorators:
        dec_str = str(dec).lower()
        # Check for FastAPI-style decorators
        for method in FASTAPI_ROUTE_DECORATORS:
            if f".{method}(" in dec_str or f"@{method}(" in dec_str:
                return True
        # Also check for router patterns
        if "@router." in dec_str or "@app." in dec_str:
            for method in FASTAPI_ROUTE_DECORATORS:
                if method in dec_str:
                    return True
    return False


def is_flask_route(func: Dict[str, Any]) -> bool:
    """
    Detect if a function is a Flask route handler via decorators.

    Flask routes are identified by:
    - @app.route("/path")
    - @blueprint.route("/path")

    Returns:
        True if the function is a Flask route handler
    """
    decorators = func.get("decorators", [])
    for dec in decorators:
        dec_str = str(dec).lower()
        if ".route(" in dec_str or "@route(" in dec_str:
            return True
    return False


def has_dependency_injection(func: Dict[str, Any]) -> bool:
    """
    Check if a function uses FastAPI Depends() for dependency injection.

    Args:
        func: Function dict from analysis

    Returns:
        True if the function uses Depends()
    """
    # Check function signature/parameters for Depends
    params = func.get("parameters", []) or func.get("args", [])
    for param in params:
        param_str = str(param).lower()
        if "depends(" in param_str:
            return True

    # Check decorators
    decorators = func.get("decorators", [])
    for dec in decorators:
        if "depends" in str(dec).lower():
            return True

    return False


def get_pure_functions(compact: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get functions that are pure (not route handlers).

    Pure functions are suitable for unit testing.
    Route handlers should be tested via integration tests (Flask) or unit tests (FastAPI).
    """
    functions = compact.get("functions", [])
    routes = compact.get("routes", [])

    # Get all route handler names
    route_handlers = set()
    for route in routes:
        handler = route.get("handler") or route.get("function") or route.get("name")
        if handler:
            route_handlers.add(handler)

    # Filter out route handlers from functions
    pure_functions = []
    for func in functions:
        func_name = func.get("name")
        if func_name and func_name not in route_handlers:
            # Also filter out common Flask/Django/FastAPI patterns
            if not _is_route_like_function(func_name, func):
                pure_functions.append(func)

    return pure_functions


def get_fastapi_routes(compact: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get FastAPI route functions from the compact analysis.

    FastAPI routes ARE unit-testable (unlike Flask routes).

    Returns:
        List of FastAPI route functions
    """
    functions = compact.get("functions", [])
    routes = compact.get("routes", [])
    fastapi_routes = compact.get("fastapi_routes", [])

    # Combine explicit fastapi_routes with detected ones
    all_routes = list(fastapi_routes)

    # Check functions for FastAPI decorators
    for func in functions:
        if is_fastapi_route(func):
            if func not in all_routes:
                all_routes.append(func)

    # Also check routes list
    for route in routes:
        if is_fastapi_route(route):
            if route not in all_routes:
                all_routes.append(route)

    return all_routes


def _is_route_like_function(name: str, func: Dict[str, Any]) -> bool:
    """
    Check if a function looks like a route handler based on naming/decorators.
    """
    # Common route handler names
    route_like_names = {
        'index', 'home', 'get', 'post', 'put', 'delete', 'patch',
        'create', 'update', 'destroy', 'show', 'edit', 'new',
        'list', 'detail', 'api', 'view', 'handler'
    }

    # Check if name contains route-like patterns
    name_lower = name.lower()
    for pattern in route_like_names:
        if pattern in name_lower:
            # Check if it's decorated with route decorators
            decorators = func.get("decorators", [])
            if decorators:
                for dec in decorators:
                    dec_str = str(dec).lower()
                    if any(x in dec_str for x in ['route', 'get', 'post', 'put', 'delete', 'api']):
                        return True

    # Check decorators directly
    decorators = func.get("decorators", [])
    for dec in decorators:
        dec_str = str(dec).lower()
        if any(x in dec_str for x in ['@app.route', '@router', '@api_view', '@get', '@post']):
            return True

    return False


def has_pure_functions(compact: Dict[str, Any]) -> bool:
    """
    Check if the codebase has any pure functions suitable for unit testing.
    """
    pure_funcs = get_pure_functions(compact)
    classes = compact.get("classes", [])
    methods = compact.get("methods", [])

    # Also check for utility classes/methods (not views/handlers)
    utility_classes = [c for c in classes if not _is_view_class(c)]
    utility_methods = [m for m in methods if not _is_route_like_function(m.get("name", ""), m)]

    return len(pure_funcs) > 0 or len(utility_classes) > 0 or len(utility_methods) > 0


def _is_view_class(cls: Dict[str, Any]) -> bool:
    """Check if a class is a view/handler class."""
    name = cls.get("name", "").lower()
    bases = cls.get("bases", [])

    # Common view class patterns
    view_patterns = ['view', 'handler', 'api', 'resource', 'viewset', 'controller']

    for pattern in view_patterns:
        if pattern in name:
            return True

    # Check base classes
    for base in bases:
        base_str = str(base).lower()
        if any(x in base_str for x in view_patterns):
            return True

    return False


def targets_count(compact: Dict[str, Any], kind: str, framework: str = "universal") -> int:
    """
    Count testable targets for a given test kind.

    CRITICAL DIFFERENCE BY FRAMEWORK:
    - Flask: routes → integration tests ONLY
    - FastAPI: routes → unit OR integration tests (routes are unit-testable!)
    - Python: ALL code → unit tests ONLY (no integration/e2e)
    """
    functions = compact.get("functions", [])
    classes = compact.get("classes", [])
    methods = compact.get("methods", [])
    routes = compact.get("routes", [])

    # PLAIN PYTHON: Only unit tests, everything is testable
    if framework == "python":
        if kind == "unit":
            # For plain Python, ALL functions/classes/methods are unit testable
            return len(functions) + len(classes) + len(methods)
        else:
            # No integration or e2e tests for plain Python
            return 0

    if kind == "unit":
        # Pure functions for unit tests
        pure_funcs = get_pure_functions(compact)
        utility_classes = [c for c in classes if not _is_view_class(c)]
        utility_methods = [m for m in methods if not _is_route_like_function(m.get("name", ""), m)]

        base_count = len(pure_funcs) + len(utility_classes) + len(utility_methods)

        # FASTAPI DIFFERENCE: Routes ARE unit-testable!
        if framework == "fastapi":
            fastapi_routes = get_fastapi_routes(compact)
            # FastAPI routes can be tested with TestClient in unit tests
            base_count += len(fastapi_routes)

        return base_count

    if kind == "e2e":
        return len(routes)

    # Integration tests
    return max(len(functions) + len(classes) + len(methods), len(routes))


def files_per_kind(compact: Dict[str, Any], kind: str, framework: str = "universal") -> int:
    """
    Distribute ALL targets across appropriate number of files.

    CRITICAL: Framework affects what can be unit tested:
    - Flask: routes → integration only (no unit test for routes)
    - FastAPI: routes → unit testable (TestClient makes routes unit-testable)
    - Python: ONLY unit tests, one file per source module
    """
    total_targets = targets_count(compact, kind, framework)

    # PLAIN PYTHON: Only generate unit tests
    if framework == "python":
        if kind != "unit":
            # No integration or e2e tests for plain Python
            return 0
        if total_targets == 0:
            print("   ℹ️  No testable targets found")
            return 0
        # For plain Python: aim for one test file per source file
        # Count unique source files
        source_files = set()
        for func in compact.get("functions", []):
            source_file = func.get("file", "")
            if source_file:
                source_files.add(source_file)
        for cls in compact.get("classes", []):
            source_file = cls.get("file", "")
            if source_file:
                source_files.add(source_file)
        for method in compact.get("methods", []):
            source_file = method.get("file", "")
            if source_file:
                source_files.add(source_file)
        # One test file per source file (or at least 1)
        return max(1, len(source_files))

    # For unit tests: check based on framework
    if kind == "unit":
        if total_targets == 0:
            if framework == "fastapi":
                print("   ℹ️  No pure functions or FastAPI routes found - skipping unit test generation")
            else:
                print("   ℹ️  No pure functions found - skipping unit test generation")
                print("   ℹ️  Routes will be tested via integration tests instead")
            return 0
        return max(1, (total_targets + 49) // 50)

    if total_targets == 0:
        return 0

    if kind == "e2e":
        return max(1, (total_targets + 19) // 20)
    else:
        return max(1, (total_targets + 29) // 30)

def create_strategic_groups(targets: List[Dict[str, Any]], num_groups: int) -> List[List[Dict[str, Any]]]:
    if not targets or num_groups <= 0:
        return []
    
    if len(targets) <= num_groups:
        return [[t] for t in targets]
    
    file_groups = {}
    for target in targets:
        file_path = target.get("file", "unknown")
        if file_path not in file_groups:
            file_groups[file_path] = []
        file_groups[file_path].append(target)
    
    groups = [[] for _ in range(num_groups)]
    group_index = 0
    
    for file_targets in file_groups.values():
        for target in file_targets:
            groups[group_index].append(target)
            group_index = (group_index + 1) % num_groups
    
    return [g for g in groups if g]

def focus_for(compact: Dict[str, Any], kind: str, shard_idx: int, total_shards: int) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    functions = compact.get("functions", [])
    classes = compact.get("classes", [])
    methods = compact.get("methods", [])
    routes = compact.get("routes", [])
    
    if kind == "unit":
        target_list = functions + classes + methods
    elif kind == "e2e":
        target_list = routes
    else:
        target_list = routes if routes else (functions + classes + methods)
    
    groups = create_strategic_groups(target_list, total_shards)
    shard_targets = groups[shard_idx] if 0 <= shard_idx < len(groups) else []
    
    target_names: List[str] = []
    for t in shard_targets:
        name = t.get("name") or t.get("handler")
        if name:
            target_names.append(name)
    
    focus_label = ", ".join(target_names) if target_names else "(none)"
    return focus_label, target_names, shard_targets


def build_prompt(kind: str, compact_json: str, focus_label: str, shard: int, total: int,
                 compact: Dict[str, Any], context: str = "", framework: str = "universal") -> List[Dict[str, str]]:
    """
    Build framework-aware test generation prompt.

    Args:
        kind: Test type (unit, integ, e2e)
        compact_json: JSON representation of code analysis
        focus_label: Label for focus targets
        shard: Current shard index
        total: Total number of shards
        compact: Compact analysis dict
        context: Additional context
        framework: Detected framework (flask, django, fastapi, universal)
    """
    # Select framework-specific system prompt
    framework = framework.lower().strip() if framework else "universal"
    SYSTEM_MIN_LOCAL = SYSTEM_PROMPTS.get(framework, SYSTEM_PROMPTS["universal"])

    test_instructions = {"unit": UNIT_ENHANCED, "integ": INTEG_ENHANCED, "e2e": E2E_ENHANCED}
    dev_instructions = test_instructions.get(kind, UNIT_ENHANCED)
    max_ctx = 60000
    trimmed_context = context[:max_ctx] if context else ""
    merged_rules = _get_framework_rules(framework)

    # === ADD GAP-FOCUSED CONTEXT ===
    gap_context = ""
    if is_gap_focused_mode():
        gap_context = get_coverage_context_for_prompts()
        if gap_context:
            print(f"   Added {len(gap_context)} chars of gap-focused context to prompt")
            print(f"   Targeting uncovered code lines")
            print(f"   Preview: {gap_context[:500]}...")

    # Framework-specific instructions to AVOID wrong patterns
    anti_patterns = _get_anti_patterns(framework)

    # Get test category rules specific to this test kind
    test_category_guidance = _get_test_category_guidance(kind, framework)

    # Add FastAPI-specific error rules if applicable
    fastapi_rules = FASTAPI_ERROR_RULES if framework == "fastapi" else ""

    user_content = f"""
{framework.upper()} {kind.upper()} TEST GENERATION - FILE {shard + 1}/{total}

DETECTED FRAMEWORK: {framework.upper()}

{OBSERVABLE_BEHAVIOR_RULES}

{STRICT_TEST_CATEGORIES}

{test_category_guidance}

{NEVER_DO_RULES}

{fastapi_rules}

{dev_instructions}
{merged_rules}

{gap_context}

{anti_patterns}

CRITICAL REQUIREMENTS:
- ALWAYS import the REAL app (from app import app) - NEVER create fake apps
- Do NOT define fixtures that already exist in conftest.py (client, app, sample_data)
- Do NOT assert on HTML content, titles, emojis, or UI text
- Build test state via API calls, NOT direct global manipulation
- Each test must be isolated - do NOT depend on other tests
- Every @pytest.mark.parametrize name MUST appear in the function signature
- Generate syntactically valid Python

FOCUS TARGETS: {focus_label}
PROJECT ANALYSIS: {compact_json}
ADDITIONAL CONTEXT (TRIMMED): {trimmed_context}

{_get_framework_scaffold(framework)}
""".strip()

    return [
        {"role": "system", "content": SYSTEM_MIN_LOCAL},
        {"role": "user", "content": user_content},
    ]


def _get_framework_rules(framework: str) -> str:
    """Get framework-specific rules."""
    base_rules = (
        "UNIVERSAL REQUIREMENTS:\n"
        "1) Use REAL imports and execution; no stubs.\n"
        "2) Test success, failure, and edge cases (None/empty/invalid).\n"
        "3) Multiple test methods per target; aim high coverage.\n"
        "4) Only output runnable Python (no markdown).\n"
        "5) @pytest.mark.parametrize: EVERY name listed MUST appear in the test "
        "   function signature. Do NOT parametrize unused names.\n"
    )

    if framework == "flask":
        return base_rules + (
            "\nFLASK RULES:\n"
            "- Use client fixture (Flask test client) - do NOT redefine it\n"
            "- Use client.get(), client.post(), client.put(), client.delete()\n"
            "- For JSON: client.post('/path', json={...})\n"
            "- Check: response.status_code, response.get_json(), response.get_data(as_text=True)\n"
            "- Do NOT use global state directly; rely on reset_app_state fixture\n"
        )
    elif framework == "django":
        return base_rules + (
            "\nDJANGO RULES:\n"
            "- Use client fixture (Django test Client) - do NOT redefine it\n"
            "- Use RequestFactory for unit testing views\n"
            "- Use QueryDict for POST/GET data so .getlist works\n"
            "- Mark DB tests with @pytest.mark.django_db\n"
            "- Prefer substring assertions for HTML content\n"
        )
    elif framework == "fastapi":
        return base_rules + (
            "\nFASTAPI RULES:\n"
            "- Use client fixture (TestClient) - do NOT redefine it\n"
            "- Use client.get(), client.post(), etc.\n"
            "- For JSON: client.post('/path', json={...})\n"
            "- Check: response.status_code, response.json()\n"
            "- For async: use pytest-asyncio markers\n"
        )
    elif framework == "python":
        return (
            "PLAIN PYTHON REQUIREMENTS (BRANCH-AWARE):\n"
            "1) Generate ONE TEST PER BRANCH, not one test per function\n"
            "2) Use REAL imports - execute actual code, no mocking unless I/O\n"
            "3) Test BOTH branches of every if/else statement\n"
            "4) Use pytest.raises() for EVERY function that can raise\n"
            "5) Test with edge cases: None, empty, zero, negative, boundary values\n"
            "6) @pytest.mark.parametrize for multiple input combinations\n"
            "7) Only output runnable Python (no markdown)\n"
            "\nPLAIN PYTHON RULES:\n"
            "- NO MagicMock for pure functions - call them directly\n"
            "- NO mocking of class instantiation - create real objects\n"
            "- NO mocking of methods without I/O - execute them\n"
            "- YES pytest.raises() for exception testing (mandatory)\n"
            "- YES parametrize for testing multiple inputs\n"
            "- YES edge case testing: None, '', 0, -1, [], {}\n"
            "\nIMPORT VALIDATION:\n"
            "- ONLY import modules that exist in the project\n"
            "- VERIFY all symbols are defined before using\n"
            "- If import fails, skip test with pytest.importorskip()\n"
        )
    else:
        return base_rules


def _get_anti_patterns(framework: str) -> str:
    """Get instructions for what NOT to do based on framework."""
    # Universal anti-patterns for ALL frameworks
    universal_anti = (
        "🚫 UNIVERSAL ANTI-PATTERNS (NEVER DO THESE):\n"
        "- NEVER create a fake app (e.g., app = Flask(__name__) or app = FastAPI())\n"
        "- ALWAYS import the REAL app: from app import app\n"
        "- NEVER assert HTML content, titles, or emojis\n"
        "- NEVER manipulate global state directly (e.g., tasks = [], tasks.clear())\n"
        "- NEVER assume test execution order - each test must be isolated\n"
        "- NEVER redefine fixtures that exist in conftest.py (client, app, sample_data)\n"
        "- NEVER assert on timestamps or auto-generated IDs with exact values\n"
        "\n"
    )

    if framework == "flask":
        return universal_anti + (
            "FLASK-SPECIFIC ANTI-PATTERNS:\n"
            "- Do NOT use @pytest.mark.django_db - this is Flask\n"
            "- Do NOT use Django's RequestFactory or QueryDict\n"
            "- Do NOT use Django's Client - use Flask's test_client()\n"
            "- Do NOT import from django.* modules\n"
            "- Do NOT write: app = Flask(__name__) - IMPORT the real app instead\n"
        )
    elif framework == "django":
        return universal_anti + (
            "DJANGO-SPECIFIC ANTI-PATTERNS:\n"
            "- Do NOT use Flask's test_client()\n"
            "- Do NOT import from flask.* modules\n"
            "- Do NOT use FastAPI's TestClient\n"
            "- Do NOT create fake Django apps - use the real one\n"
        )
    elif framework == "fastapi":
        return universal_anti + (
            "FASTAPI-SPECIFIC ANTI-PATTERNS:\n"
            "- Do NOT use @pytest.mark.django_db - this is FastAPI\n"
            "- Do NOT use Django's RequestFactory or QueryDict\n"
            "- Do NOT use Flask's test_client() - use TestClient from fastapi.testclient\n"
            "- Do NOT write: app = FastAPI() - IMPORT the real app instead\n"
            "- Do NOT expect 400 for validation errors - FastAPI returns 422!\n"
            "- Do NOT mock dependencies unless they fail without mocking\n"
            "\n"
            "FASTAPI ERROR CODES (CRITICAL):\n"
            "- Validation error (missing/wrong field) → 422 (NOT 400!)\n"
            "- Route not found → 404\n"
            "- Explicit HTTPException(400) → 400\n"
            "- Unhandled exception → 500\n"
        )
    elif framework == "python":
        # Plain Python - different anti-patterns focused on mocking
        return (
            "🚫 PLAIN PYTHON ANTI-PATTERNS (CRITICAL FOR COVERAGE):\n"
            "\n"
            "MOCKING ANTI-PATTERNS (COVERAGE KILLERS):\n"
            "- NEVER use MagicMock for pure functions - execute them directly\n"
            "- NEVER mock class instantiation - create real objects\n"
            "- NEVER mock methods that have no I/O - call them directly\n"
            "- NEVER use mock.return_value for testable logic\n"
            "- Mocked code does NOT count for coverage!\n"
            "\n"
            "WHEN TO USE MOCKING (ONLY THESE CASES):\n"
            "✅ Network calls (requests.get, urllib)\n"
            "✅ File I/O (open, read, write)\n"
            "✅ Database connections\n"
            "✅ Time-dependent functions (datetime.now, time.sleep)\n"
            "✅ External APIs and services\n"
            "\n"
            "TESTING ANTI-PATTERNS:\n"
            "- NEVER write only happy-path tests - test ALL branches\n"
            "- NEVER skip exception testing - use pytest.raises()\n"
            "- NEVER test just one case for if/else - test BOTH branches\n"
            "- NEVER assume single test per function is enough\n"
            "\n"
            "IMPORT ANTI-PATTERNS:\n"
            "- NEVER import non-existent modules\n"
            "- NEVER use undefined symbols\n"
            "- ALWAYS verify imports resolve before generating tests\n"
        )
    else:
        return universal_anti


def _get_test_category_guidance(kind: str, framework: str) -> str:
    """Get specific guidance for each test category to prevent misclassification."""
    if kind == "unit":
        # Plain Python - branch-aware unit tests
        if framework == "python":
            return f"""
🎯 UNIT TEST SPECIFIC GUIDANCE (PLAIN PYTHON - BRANCH-AWARE):

FOR THIS UNIT TEST FILE, YOU MUST:
✅ Generate ONE TEST PER BRANCH, not one test per function
✅ Test BOTH branches of every if/else
✅ Test BOTH success AND exception paths for try/except
✅ Test with empty AND non-empty inputs for loops
✅ Use pytest.raises() for EVERY function that can raise
✅ Execute REAL code - no mocking of pure logic
✅ Test edge cases: None, empty string, zero, negative numbers

BRANCH-AWARE TESTING PATTERN:
```python
# For a function with if/else:
def test_function_when_condition_true():
    result = my_function(valid_input)
    assert result == expected_true_result

def test_function_when_condition_false():
    result = my_function(invalid_input)
    assert result == expected_false_result

# For a function that can raise:
def test_function_success():
    result = my_function(valid_input)
    assert result == expected

def test_function_raises():
    with pytest.raises(ExpectedException):
        my_function(invalid_input)
```

FOR THIS UNIT TEST FILE, YOU MUST NOT:
❌ Mock pure Python functions - execute them directly
❌ Write only happy-path tests - cover ALL branches
❌ Skip exception testing - it's MANDATORY
❌ Use MagicMock for testable logic
❌ Import modules that don't exist

{BRANCH_AWARE_RULES}

{PLAIN_PYTHON_MOCKING_RULES}

{EXCEPTION_TEST_RULES}
"""
        # FastAPI routes ARE unit-testable - this is the key difference!
        elif framework == "fastapi":
            return """
🎯 UNIT TEST SPECIFIC GUIDANCE (FASTAPI):

FOR THIS UNIT TEST FILE, YOU MUST:
✅ Test pure functions and utility methods
✅ Test data validation functions
✅ Test FastAPI routes using TestClient (routes ARE unit-testable!)
✅ Test Pydantic model validation
✅ Test helper methods and utilities

FASTAPI ROUTES ARE UNIT-TESTABLE:
```python
def test_create_item(client):
    res = client.post("/items", json={"name": "Test", "price": 10})
    assert res.status_code == 201
    assert res.json()["name"] == "Test"

def test_create_item_validation(client):
    res = client.post("/items", json={})
    assert res.status_code == 422  # NOT 400!
```

FOR THIS UNIT TEST FILE, YOU MUST NOT:
❌ Create fake apps (app = FastAPI() is WRONG)
❌ Expect 400 for validation errors (FastAPI returns 422!)
❌ Mock dependencies unless absolutely necessary

REMEMBER: FastAPI routes ARE unit-testable with TestClient!
"""
        else:
            return f"""
🎯 UNIT TEST SPECIFIC GUIDANCE ({framework.upper()}):

FOR THIS UNIT TEST FILE, YOU MUST:
✅ Test ONLY pure functions and utility methods
✅ Test data validation functions
✅ Test helper methods that don't require HTTP
✅ Test class methods with direct calls (not via HTTP)

FOR THIS UNIT TEST FILE, YOU MUST NOT:
❌ Test routes/endpoints - those belong in integration tests
❌ Use client.get(), client.post() - that's integration testing
❌ Create any Flask/Django app instances
❌ Test HTTP request/response - that's integration testing

If the codebase only has routes/views and no pure functions,
generate tests for utility logic WITHIN those functions,
or generate minimal placeholder tests.

REMEMBER: Routes ≠ Unit Tests. Routes = Integration Tests.
"""
    elif kind == "integ":
        if framework == "fastapi":
            return """
🎯 INTEGRATION TEST SPECIFIC GUIDANCE (FASTAPI):

FOR THIS INTEGRATION TEST FILE, YOU MUST:
✅ Test complete API workflows
✅ Test database interactions (if applicable)
✅ Test service integrations
✅ Use the client fixture from conftest.py
✅ Test error handling with correct status codes

FASTAPI ERROR CODES (CRITICAL):
- Validation error → 422 (NOT 400!)
- Not found → 404
- Auth failure → 401/403
- Server error → 500

FOR THIS INTEGRATION TEST FILE, YOU MUST NOT:
❌ Create fake apps (app = FastAPI() is WRONG)
❌ Expect 400 for Pydantic validation errors
❌ Define routes inside the test file
❌ Redefine the client fixture

CRITICAL: Import the REAL app, don't create a fake one!
"""
        else:
            return f"""
🎯 INTEGRATION TEST SPECIFIC GUIDANCE ({framework.upper()}):

FOR THIS INTEGRATION TEST FILE, YOU MUST:
✅ Test API endpoints via the client fixture
✅ Use client.get(), client.post(), client.put(), client.delete()
✅ Import the REAL app: from app import app
✅ Test request/response cycles
✅ Assert on status codes and JSON structure

FOR THIS INTEGRATION TEST FILE, YOU MUST NOT:
❌ Create fake apps (app = Flask(__name__) is WRONG)
❌ Define your own routes inside the test file
❌ Assert on HTML content, titles, or emojis
❌ Redefine the client fixture - it's in conftest.py

CRITICAL: Import the REAL app, don't create a fake one!
"""
    elif kind == "e2e":
        return f"""
🎯 END-TO-END TEST SPECIFIC GUIDANCE ({framework.upper()}):

FOR THIS E2E TEST FILE, YOU MUST:
✅ Test complete user workflows via API
✅ Chain operations: create → read → update → delete
✅ Build state via API calls (POST to create data)
✅ Clean up via API calls (DELETE) or rely on fixture reset
✅ Use the client fixture from conftest.py

FOR THIS E2E TEST FILE, YOU MUST NOT:
❌ Create fake apps - use the REAL app
❌ Assert on HTML/UI content
❌ Manipulate global state directly (tasks = [])
❌ Depend on other tests running first - each test is isolated

STATE MANAGEMENT: Build state via API calls, not direct manipulation.
Example: POST /api/tasks to create, then GET /api/tasks to verify.
"""
    else:
        return ""


def _get_framework_scaffold(framework: str) -> str:
    """Get framework-specific test scaffold."""
    if framework == "flask":
        return '''
# Flask test template - use existing fixtures from conftest.py
import pytest

def test_example_endpoint(client):
    """Example Flask test using client fixture."""
    response = client.get('/')
    assert response.status_code == 200

def test_example_json_api(client):
    """Example Flask JSON API test."""
    response = client.post('/api/endpoint', json={"key": "value"})
    assert response.status_code in [200, 201]
    data = response.get_json()
    assert data is not None
'''
    elif framework == "django":
        return '''
# Django test template - use existing fixtures from conftest.py
import pytest

@pytest.mark.django_db
def test_example_view(client):
    """Example Django test using client fixture."""
    response = client.get('/')
    assert response.status_code == 200

@pytest.mark.django_db
def test_example_with_user(authenticated_client):
    """Example Django test with authenticated user."""
    response = authenticated_client.get('/profile/')
    assert response.status_code == 200
'''
    elif framework == "fastapi":
        return '''
# FastAPI test template - use existing fixtures from conftest.py
import pytest

def test_example_endpoint(client):
    """Example FastAPI test using client fixture."""
    response = client.get('/')
    assert response.status_code == 200

def test_example_json_api(client):
    """Example FastAPI JSON API test."""
    response = client.post('/api/endpoint', json={"key": "value"})
    assert response.status_code in [200, 201]
    data = response.json()
    assert data is not None

def test_validation_error(client):
    """FastAPI returns 422 for validation errors, NOT 400!"""
    response = client.post('/api/endpoint', json={})  # Missing required fields
    assert response.status_code == 422  # CRITICAL: FastAPI uses 422!

def test_wrong_type_validation(client):
    """FastAPI validates types via Pydantic."""
    response = client.post('/api/endpoint', json={"count": "not_a_number"})
    assert response.status_code == 422  # Type validation error

def test_auth_required(client):
    """Test endpoint that requires authentication."""
    response = client.get('/api/protected')
    assert response.status_code in (200, 401)  # Don't assume auth behavior
'''
    elif framework == "python":
        return '''
# Plain Python test template - BRANCH-AWARE for maximum coverage
# NO MOCKING of pure logic - execute real code
import pytest

# ============== BRANCH-AWARE TEST EXAMPLES ==============

# Example 1: Testing a function with if/else (test BOTH branches)
def test_function_when_condition_true():
    """Test the TRUE branch of conditional logic."""
    # Call function with input that triggers TRUE branch
    result = my_function(valid_input)
    assert result == expected_true_result

def test_function_when_condition_false():
    """Test the FALSE branch of conditional logic."""
    # Call function with input that triggers FALSE branch
    result = my_function(invalid_input)
    assert result == expected_false_result

# Example 2: Testing exception paths (MANDATORY for functions that can raise)
def test_function_success_path():
    """Test normal execution - no exception."""
    result = risky_function(valid_input)
    assert result is not None

def test_function_exception_path():
    """Test exception path - use pytest.raises()."""
    with pytest.raises(ValueError):
        risky_function(invalid_input)

# Example 3: Testing with edge cases
@pytest.mark.parametrize("input_val,expected", [
    ("valid", True),      # Normal case
    ("", False),          # Empty string
    (None, False),        # None value
    ("  ", False),        # Whitespace only
])
def test_validation_multiple_cases(input_val, expected):
    """Test validation with multiple edge cases."""
    result = validate(input_val)
    assert result == expected

# Example 4: Testing loops (empty vs non-empty)
def test_process_items_with_items():
    """Test loop with non-empty list."""
    items = [1, 2, 3]
    result = process_items(items)
    assert result == 6  # sum

def test_process_items_empty():
    """Test loop with empty list."""
    result = process_items([])
    assert result == 0

# Example 5: Testing class methods (real instantiation, no mocking)
class TestMyClass:
    def test_init(self):
        """Test class instantiation."""
        obj = MyClass(param1, param2)
        assert obj.attribute == expected_value

    def test_method_success(self):
        """Test method with valid input."""
        obj = MyClass(param1, param2)
        result = obj.method(valid_input)
        assert result == expected

    def test_method_raises(self):
        """Test method exception path."""
        obj = MyClass(param1, param2)
        with pytest.raises(ExpectedException):
            obj.method(invalid_input)
'''
    else:
        return UNIVERSAL_SCAFFOLD



def _merge_universal_text():
    """Combines the strongest parts of all requirement variants including gap-focused."""
    base_text = (
        "UNIVERSAL REQUIREMENTS:\n"
        "1) Use REAL imports and execution; no stubs.\n"
        "2) Test success, failure, and edge cases (None/empty/invalid).\n"
        "3) Multiple test methods per target; aim high coverage.\n"
        "4) Only output runnable Python (no markdown).\n"
        "5) @pytest.mark.parametrize: EVERY name listed MUST appear in the test "
        "   function signature. Do NOT parametrize unused names.\n"
        "6) Call-safety: Never repeat the same keyword in a single call "
        "   (e.g., Mock(name=...) only once). Ensure all calls are valid Python.\n"
        "\n"
        "DJANGO RULES (when Django is present):\n"
        "- Build requests with RequestFactory (not SimpleNamespace/DummyRequest).\n"
        "- Use QueryDict (or helper) for request.POST/GET so .getlist works.\n"
        "- If touching models/querysets, mark with pytest.mark.django_db.\n"
        "- Prefer substring assertions (response/content), avoid strict HTML equality.\n"
        "- Don't fabricate .object_list as raw lists; use queryset-like objects "
        "  (supporting .order_by/.all) when needed.\n"
    )
    
    # Add gap-focused specific guidance if in that mode
    if is_gap_focused_mode():
        gap_guidance = (
            "\n"
            "GAP-FOCUSED MODE REQUIREMENTS:\n"
            "- PRIORITY: Generate tests that hit the specific UNCOVERED lines listed above\n"
            "- Do NOT test already-covered code paths\n"
            "- Focus each test on covering multiple uncovered lines when possible\n"
            "- Target the specific functions/classes/methods marked as uncovered\n"
            "- Design tests to cover multiple uncovered lines per test when possible\n"
            "- Each test should directly exercise the uncovered code sections\n"
            "- Use the line numbers provided to guide your test design\n"
            "- Prioritize tests that will increase coverage percentage most\n"
        )
        base_text += gap_guidance
    
    return base_text

# def build_prompt(kind: str, compact_json: str, focus_label: str, shard: int, total: int,
#                  compact: Dict[str, Any], context: str = "") -> List[Dict[str, str]]:
#     """
#     Final, unified override (append-only) with GAP-FOCUSED support.
#     This merges: (a) parametrize-safety, (b) call-safety, and (c) Django-aware guidance,
#     and (d) gap-focused coverage targeting.
#     The LAST definition in the file is the one Python will use.
#     """
#     SYSTEM_MIN_LOCAL = SYSTEM_MIN
#     test_instructions = {
#         "unit": UNIT_ENHANCED,
#         "integ": INTEG_ENHANCED,
#         "e2e": E2E_ENHANCED
#     }
#     dev_instructions = test_instructions.get(kind, UNIT_ENHANCED)
#     max_ctx = 60000
#     trimmed_context = context[:max_ctx] if context else ""
#     merged_rules = _merge_universal_text()

#     # === ADD GAP-FOCUSED CONTEXT ===
#     gap_context = ""
#     if is_gap_focused_mode():
#         gap_context = get_coverage_context_for_prompts()
#         print(f"   Added {len(gap_context)} chars of gap-focused context to prompt")

#     user_content = f"""
# UNIVERSAL {kind.upper()} TEST GENERATION - FILE {shard + 1}/{total}

# {dev_instructions}

# {merged_rules}

# FOCUS TARGETS: {focus_label}
# PROJECT ANALYSIS: {compact_json}
# ADDITIONAL CONTEXT (TRIMMED): {trimmed_context}

# {UNIVERSAL_SCAFFOLD}
# """.strip()

#     return [
#         {"role": "system", "content": SYSTEM_MIN_LOCAL},
#         {"role": "user", "content": user_content},
#     ]
