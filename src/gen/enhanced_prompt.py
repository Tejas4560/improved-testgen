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

# Default for backward compatibility
SYSTEM_MIN = SYSTEM_PROMPTS["universal"]

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

def targets_count(compact: Dict[str, Any], kind: str) -> int:
    functions = compact.get("functions", [])
    classes = compact.get("classes", [])
    methods = compact.get("methods", [])
    routes = compact.get("routes", [])
    
    if kind == "unit":
        return len(functions) + len(classes) + len(methods)
    if kind == "e2e":
        return len(routes)
    return max(len(functions) + len(classes) + len(methods), len(routes))

def files_per_kind(compact: Dict[str, Any], kind: str) -> int:
    """Distribute ALL targets across appropriate number of files."""
    
    total_targets = targets_count(compact, kind)
    if total_targets == 0:
        return 0
    
    targets_per_file = 50
    
    if kind == "unit":
        return max(1, (total_targets + targets_per_file - 1) // targets_per_file)
    elif kind == "e2e":
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

    user_content = f"""
{framework.upper()} {kind.upper()} TEST GENERATION - FILE {shard + 1}/{total}

DETECTED FRAMEWORK: {framework.upper()}

{dev_instructions}
{merged_rules}

{gap_context}

{anti_patterns}

CRITICAL REQUIREMENTS:
- Do NOT define fixtures that already exist in conftest.py (client, app, sample_data)
- Do NOT create duplicate fixture definitions in test files
- Every @pytest.mark.parametrize name MUST appear in the function signature
- Never repeat the same keyword in a call (e.g., Mock(name=...) only once)
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
    else:
        return base_rules


def _get_anti_patterns(framework: str) -> str:
    """Get instructions for what NOT to do based on framework."""
    if framework == "flask":
        return (
            "FLASK ANTI-PATTERNS (DO NOT USE):\n"
            "- Do NOT use @pytest.mark.django_db - this is Flask\n"
            "- Do NOT use Django's RequestFactory or QueryDict\n"
            "- Do NOT use Django's Client - use Flask's test_client()\n"
            "- Do NOT import from django.* modules\n"
            "- Do NOT define @pytest.fixture def client() - it's in conftest.py\n"
        )
    elif framework == "django":
        return (
            "DJANGO ANTI-PATTERNS (DO NOT USE):\n"
            "- Do NOT use Flask's test_client()\n"
            "- Do NOT import from flask.* modules\n"
            "- Do NOT use FastAPI's TestClient\n"
            "- Do NOT define @pytest.fixture def client() - it's in conftest.py\n"
        )
    elif framework == "fastapi":
        return (
            "FASTAPI ANTI-PATTERNS (DO NOT USE):\n"
            "- Do NOT use @pytest.mark.django_db - this is FastAPI\n"
            "- Do NOT use Django's RequestFactory or QueryDict\n"
            "- Do NOT use Flask's test_client()\n"
            "- Do NOT define @pytest.fixture def client() - it's in conftest.py\n"
        )
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
