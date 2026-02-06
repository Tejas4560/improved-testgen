# src/gen/enhanced_generate.py - UNIVERSAL VERSION for any project structure
import argparse
import ast
import datetime
import json
import os
import pathlib
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

# Import the enhanced modules

from .gap_aware_analysis import (
    apply_gap_aware_filtering,
    is_gap_focused_mode,
    get_coverage_context_for_prompts
)

# Import framework handlers and orchestrator
from ..framework_handlers.manager import FrameworkManager
from ..test_generation.orchestrator import TestGenerationOrchestrator

__all__ = ["generate_all", "main"]

try:
    from .postprocess import extract_python_only, massage, validate_code
except Exception as _e:
    print(f"Warning: postprocess import failed: {_e}; using fallbacks")
    import ast as _ast
    import re as _re
    
    def extract_python_only(text: str) -> str:
        if "```" in text:
            blocks = _re.findall(r"```(?:python)?\s*(.*?)```", text, flags=_re.IGNORECASE|_re.DOTALL)
            return "\n\n".join(blocks) if blocks else text.replace("```","")
        return text
    
    def validate_code(code: str):
        if not code.strip(): return False, "empty output"
        if not _re.search(r"def test_", code): return False, "no test functions"
        try: _ast.parse(code); return True, ""
        except SyntaxError as e: return False, f"syntax error: {e}"
    
    def massage(code: str): return code

def _detect_framework(analysis: Dict[str, Any], target_root: pathlib.Path) -> str:
    """
    Detect the framework from analysis or by inspecting the target directory.

    CRITICAL: Detect statically, do NOT import user code.

    Detection priority:
    1. FastAPI (from fastapi import FastAPI, import fastapi)
    2. Flask (from flask import Flask, import flask)
    3. Django (from django..., import django)
    4. Python (plain Python - no web framework)
    5. Universal (fallback for unknown)
    """
    # Check if framework is already in analysis
    if "framework" in analysis:
        return analysis["framework"].lower()

    # Build imports text for exact matching
    imports_text = ""
    imports = analysis.get("imports", [])
    for imp in imports:
        if isinstance(imp, dict):
            imports_text += f" {imp.get('module', '')} {imp.get('name', '')} "
        elif isinstance(imp, str):
            imports_text += f" {imp} "

    imports_text = imports_text.lower()

    # STATIC DETECTION - Do NOT import user code
    # Check for FastAPI FIRST (more specific than Flask)
    if "from fastapi import fastapi" in imports_text or "import fastapi" in imports_text:
        return "fastapi"
    if "from fastapi" in imports_text:
        return "fastapi"

    # Check for Flask
    if "from flask import flask" in imports_text or "import flask" in imports_text:
        return "flask"
    if "from flask" in imports_text:
        return "flask"

    # Check for Django
    if "from django" in imports_text or "import django" in imports_text:
        return "django"

    # Check files in target directory for more thorough detection
    has_web_framework = False
    has_python_files = False
    try:
        for py_file in target_root.glob("**/*.py"):
            py_str = str(py_file)
            # Skip non-project directories (venv, node_modules, etc.)
            if any(skip in py_str for skip in [
                "venv", "site-packages", "node_modules", ".venv",
                "__pycache__", ".git", "frontend", "dist", "build"
            ]):
                continue
            if "test" in py_str.lower():
                continue  # Skip test files

            has_python_files = True
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')

                # FastAPI detection (check first - more specific)
                if "from fastapi import FastAPI" in content:
                    return "fastapi"
                if "import fastapi" in content:
                    return "fastapi"

                # Flask detection
                if "from flask import Flask" in content:
                    return "flask"
                if "import flask" in content:
                    return "flask"

                # Django detection
                if "from django" in content or "import django" in content:
                    return "django"

                # Check for any web framework indicators
                web_indicators = [
                    "import flask", "import fastapi", "import django",
                    "from flask", "from fastapi", "from django",
                    "import tornado", "import aiohttp", "import bottle",
                    "import pyramid", "import cherrypy", "import falcon"
                ]
                for indicator in web_indicators:
                    if indicator in content.lower():
                        has_web_framework = True
                        break
            except Exception:
                continue
    except Exception:
        pass

    # PLAIN PYTHON DETECTION:
    # If we have Python files but NO web framework indicators, it's plain Python
    if has_python_files and not has_web_framework:
        print("   Detected PLAIN PYTHON project (no web framework)")
        return "python"

    return "universal"


def _get_source_to_test_mapping(target_root: pathlib.Path, analysis: Dict[str, Any]) -> Dict[str, str]:
    """
    Create a mapping from source files to test file names.

    For plain Python projects: 1 test file per source file.
    This prevents duplicate coverage and test explosion.

    Example:
        ecommerce.py -> test_unit_ecommerce.py
        app.py -> test_unit_app.py
    """
    mapping = {}

    try:
        for py_file in target_root.glob("**/*.py"):
            py_str = str(py_file)
            if any(skip in py_str for skip in [
                "venv", "site-packages", "node_modules", ".venv",
                "__pycache__", ".git", "frontend", "dist", "build"
            ]):
                continue
            if "test" in py_str.lower():
                continue
            if py_file.name.startswith("_"):
                continue  # Skip private modules

            # Get relative path from target root
            rel_path = py_file.relative_to(target_root)
            source_stem = py_file.stem

            # Create test file name
            test_filename = f"test_unit_{source_stem}.py"

            mapping[str(rel_path)] = test_filename
    except Exception as e:
        print(f"Warning: Error creating source-to-test mapping: {e}")

    return mapping


def _count_branches_in_function(func: Dict[str, Any]) -> int:
    """
    Count the number of branches in a function for branch-aware testing.

    Branch detection heuristic:
    - if/else: +1 branch per if
    - try/except: +1 branch
    - for/while with potential empty: +1 branch
    - raise: +1 branch (exception path)
    - boolean conditions in return: +1 branch

    Returns minimum number of tests needed for branch coverage.
    """
    branches = 1  # Always have at least happy path

    # Get function body or source if available
    body = func.get("body", "") or func.get("source", "") or ""
    body_lower = body.lower()

    # Count if statements
    if_count = body_lower.count("if ") + body_lower.count("elif ")
    branches += if_count

    # Count try/except blocks
    try_count = body_lower.count("try:")
    branches += try_count

    # Count raise statements (exception paths)
    raise_count = body_lower.count("raise ")
    branches += raise_count

    # Count for/while loops (empty vs non-empty)
    loop_count = body_lower.count("for ") + body_lower.count("while ")
    branches += loop_count

    # Count early returns (multiple exit points)
    return_count = body_lower.count("return ")
    if return_count > 1:
        branches += return_count - 1

    return branches


def _can_raise_exception(func: Dict[str, Any]) -> bool:
    """
    Check if a function can raise an exception.

    Used to ensure we generate pytest.raises tests for functions that raise.
    """
    body = func.get("body", "") or func.get("source", "") or ""

    # Direct raise statements
    if "raise " in body:
        return True

    # Common exception-raising patterns
    exception_patterns = [
        "ValueError", "TypeError", "KeyError", "IndexError",
        "RuntimeError", "AttributeError", "ZeroDivisionError",
        "FileNotFoundError", "PermissionError", "Exception"
    ]

    for pattern in exception_patterns:
        if pattern in body:
            return True

    return False


def _estimate_branch_coverage(compact: Dict[str, Any]) -> Dict[str, int]:
    """
    Estimate total branches and calculate tests needed.

    Returns dict with:
    - total_branches: Total branches across all functions
    - tests_needed: Estimated tests needed for coverage
    - functions_with_raises: Count of functions that can raise
    """
    total_branches = 0
    functions_with_raises = 0

    for func in compact.get("functions", []):
        total_branches += _count_branches_in_function(func)
        if _can_raise_exception(func):
            functions_with_raises += 1

    for method in compact.get("methods", []):
        total_branches += _count_branches_in_function(method)
        if _can_raise_exception(method):
            functions_with_raises += 1

    # For classes, count branches in __init__ and other methods
    for cls in compact.get("classes", []):
        methods = cls.get("methods", [])
        for method in methods:
            total_branches += _count_branches_in_function(method)
            if _can_raise_exception(method):
                functions_with_raises += 1

    return {
        "total_branches": total_branches,
        "tests_needed": total_branches,  # One test per branch
        "functions_with_raises": functions_with_raises
    }


def _create_universal_conftest(outdir: pathlib.Path, target_root: pathlib.Path,
                                framework: str = "universal") -> str:
    """Create framework-specific conftest.py."""
    from .conftest_text import conftest_text
    from .writer import write_text

    conftest_path = outdir / "conftest.py"

    # Generate framework-specific conftest
    base_conftest = conftest_text(framework=framework, target_root=str(target_root))
    
    # Enhanced universal conftest
    universal_conftest = base_conftest + f'''

# UNIVERSAL fixtures for any project structure
import sys
import os
import pathlib  # needed for _setup_detected_frameworks()

# Add project root to Python path for universal imports
PROJECT_ROOT = r"{target_root}"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

@pytest.fixture(scope="session", autouse=True)
def universal_coverage_setup():
    """UNIVERSAL setup for maximum coverage with real code execution."""
    # Set coverage optimization environment
    os.environ['COVERAGE_OPTIMIZATION'] = 'universal'
    os.environ['REAL_IMPORTS_ONLY'] = 'true'
    os.environ['TESTING_MAX_COVERAGE'] = 'true'
    
    # Universal framework auto-detection
    _setup_detected_frameworks()
    
    yield
    
    # Cleanup
    os.environ.pop('COVERAGE_OPTIMIZATION', None)
    os.environ.pop('REAL_IMPORTS_ONLY', None)

def _setup_detected_frameworks():
    """Auto-detect and setup frameworks for any project structure."""
    # Try to detect and import common project modules
    project_modules = ['app', 'main', 'application', 'server', 'api', 'backend', 'core', 'project']
    
    # Also try to detect project-specific modules from the structure
    try:
        # Look for Python files in project root to detect main modules
        for py_file in pathlib.Path(PROJECT_ROOT).glob("*.py"):
            module_name = py_file.stem
            if module_name not in project_modules and not module_name.startswith('_'):
                project_modules.append(module_name)
    except Exception:
        pass
    
    for module_name in project_modules:
        try:
            __import__(module_name)
            print(f"Detected and imported: {{module_name}}")
        except ImportError:
            continue

@pytest.fixture
def universal_sample_data():
    """UNIVERSAL sample data for comprehensive testing."""
    return {{
        'user': {{
            'username': 'testuser_universal',
            'email': 'universal_test@example.com',
            'password': 'UniversalPassword123!',
        }},
        'api_payloads': {{
            'create_user': {{
                'user': {{
                    'username': 'api_test_user',
                    'email': 'api_test@example.com',
                    'password': 'ApiTestPass123!',
                }}
            }},
            'login': {{
                'user': {{
                    'email': 'api_test@example.com',
                    'password': 'ApiTestPass123!',
                }}
            }},
        }},
        'edge_cases': {{
            'empty_string': '',
            'none_value': None,
            'zero': 0,
            'negative': -1,
            'large_number': 999999999999,
            'special_chars': r'!@#$%^&*()_+-=[]{{}}|;:,.<>?/\\~`',
            'unicode': '测试数据 🚀 émojis ñoños café ☕',
            'long_string': 'x' * 1000,
            'whitespace': '   ',
        }}
    }}

# UNIVERSAL test utilities
class UniversalTestUtils:
    """Universal utilities for achieving maximum coverage."""
    
    @staticmethod
    def setup_universal_imports():
        """Setup universal imports for any project structure."""
        print("UNIVERSAL: Setting up imports for any project structure")
    
    @staticmethod
    def generate_comprehensive_test_cases(target_name, target_type):
        """Generate comprehensive test cases for any target."""
        base_cases = [
            f"test_{{target_name}}_basic_functionality",
            f"test_{{target_name}}_edge_cases", 
            f"test_{{target_name}}_error_conditions",
            f"test_{{target_name}}_validation",
        ]
        
        if target_type in ['model', 'class']:
            base_cases.extend([
                f"test_{{target_name}}_creation",
                f"test_{{target_name}}_methods",
                f"test_{{target_name}}_properties",
            ])
        
        if target_type in ['api', 'route']:
            base_cases.extend([
                f"test_{{target_name}}_get",
                f"test_{{target_name}}_post", 
                f"test_{{target_name}}_put",
                f"test_{{target_name}}_delete",
            ])
        
        return base_cases
'''
    
    write_text(conftest_path, universal_conftest)
    return str(conftest_path)

def _generate_with_universal_retry(messages: List[Dict], max_attempts: int = 5) -> str:
    """Generate test code with UNIVERSAL retry logic and timeout handling."""
    from .openai_client import (APIError, APITimeoutError, RateLimitError,
                                create_chat_completion, create_client,
                                get_deployment_name)
    import httpx

    client = create_client()
    deployment = get_deployment_name()
    last_error = "unknown error"
    backoff_delays = [0, 2, 4, 8, 16]  # Exponential backoff

    print(f"Starting test generation with {max_attempts} max attempts...")
    
    for attempt in range(max_attempts):
        try:
            # Apply backoff delay
            if backoff_delays[attempt] > 0:
                print(f"Retrying generation in {backoff_delays[attempt]} seconds...")
                time.sleep(backoff_delays[attempt])
            
            # UNIVERSAL prompt enhancement for coverage on retry
            if attempt > 0:
                coverage_reminder = {
                    "role": "user",
                    "content": f"UNIVERSAL RETRY {attempt + 1}/{max_attempts}: Previous attempt failed. "
                               "CRITICAL: Generate tests for maximum COVERAGE that work with ANY PROJECT STRUCTURE. "
                               "Requirements:\n"
                               "1. Use absolute imports or proper relative imports\n"
                               "2. Handle any project structure (flat, nested, packages)\n"
                               "3. Generate 5-8 test methods per major target\n"
                               "4. Test ALL code paths: success, failure, edge cases\n"
                               "5. Include proper Python path setup\n"
                               "6. Test ALL public methods and properties\n"
                               "7. Ensure imports work regardless of project layout\n"
                }
                messages.append(coverage_reminder)
            
            # Make API call
            response_content = create_chat_completion(client, deployment, messages)
            
            if not response_content.strip():
                last_error = "Empty response from API"
                continue
            
            # Extract and validate Python code
            extracted_code = extract_python_only(response_content)
            if not extracted_code.strip():
                last_error = "No Python code found in response"
                continue
            
            # UNIVERSAL validation for coverage-focused code
            is_valid, validation_error = validate_code(extracted_code)
            if not is_valid:
                last_error = f"Code validation failed: {validation_error}"
                
                # Enhanced feedback based on error type
                if "no test functions" in validation_error.lower():
                    feedback_msg = {
                        "role": "user",
                        "content": "CRITICAL: Generated code lacks test functions. "
                                   "Generate 5-8 test methods per major target using 'def test_*' format. "
                                   "Each class should have tests for: creation, all methods, properties, "
                                   "edge cases, and error conditions."
                    }
                elif "syntax error" in validation_error.lower():
                    feedback_msg = {
                        "role": "user", 
                        "content": "Syntax error in generated code. Ensure:\n"
                                   "- Proper Python indentation\n"
                                   "- Variables declared before use\n" 
                                   "- Valid Python syntax in all test methods\n"
                                   "- All functions have proper docstrings\n"
                    }
                else:
                    feedback_msg = {
                        "role": "user",
                        "content": f"Code validation failed: {validation_error}. "
                                   "Generate valid Python code with maximum coverage target."
                    }
                
                messages.append(feedback_msg)
                continue
            
            # UNIVERSAL post-processing for coverage
            try:
                processed_code = massage(extracted_code)
                
                # Enhanced post-processing: Add universal optimization
                processed_code = _optimize_for_universal_coverage(processed_code)
                
                # Final validation
                final_valid, final_error = validate_code(processed_code)
                if final_valid:
                    return processed_code
                else:
                    print(f"Post-processing validation failed: {final_error}, using original")
                    return extracted_code
                    
            except Exception as process_error:
                print(f"Post-processing error: {process_error}, using original code")
                return extracted_code
                
        except RateLimitError as e:
            last_error = f"Rate limit exceeded: {e}"
            print(f"⚠️ Rate limit hit on attempt {attempt + 1}, backing off...")

        except APITimeoutError as e:
            last_error = f"API timeout: {e}"
            print(f"⚠️ API timeout on attempt {attempt + 1}, will retry with longer timeout...")

        except httpx.TimeoutException as e:
            last_error = f"HTTP timeout: {e}"
            print(f"⚠️ HTTP timeout on attempt {attempt + 1}, will retry...")

        except httpx.ConnectError as e:
            last_error = f"Connection error: {e}"
            print(f"⚠️ Connection failed on attempt {attempt + 1}, will retry...")

        except APIError as e:
            last_error = f"API error: {e}"
            if hasattr(e, 'status_code') and e.status_code in [400, 401, 403]:
                print(f"❌ Non-retryable API error: {e}")
                break
            print(f"⚠️ API error on attempt {attempt + 1}, retrying...")

        except Exception as e:
            last_error = f"API call failed: {type(e).__name__}: {e}"
            print(f"⚠️ Generation attempt {attempt + 1} failed: {type(e).__name__}: {e}")
    
    # If all attempts failed, raise with detailed error
    print(f"❌ All {max_attempts} generation attempts failed")
    print(f"Last error: {last_error}")
    print("Troubleshooting tips:")
    print("  - Check Azure OpenAI credentials and endpoint")
    print("  - Increase OPENAI_TIMEOUT environment variable")
    print("  - Check network connectivity to Azure")
    print("  - Verify deployment name is correct")
    raise RuntimeError(f"UNIVERSAL test generation failed after {max_attempts} attempts. Last error: {last_error}")

def _optimize_for_universal_coverage(code: str) -> str:
    """Optimize generated code for universal project compatibility."""
    lines = code.splitlines()
    optimized_lines = []
    
    # Add universal optimization header
    optimized_lines.append('"""')
    optimized_lines.append('UNIVERSAL test suite - works with any project structure')
    optimized_lines.append('REAL IMPORTS ONLY - No stubs')
    optimized_lines.append('Generated for maximum compatibility and coverage')
    optimized_lines.append('"""')
    optimized_lines.append('')
    
    for line in lines:
        # Skip existing headers
        if line.strip().startswith('"""') and 'universal' in line.lower():
            continue
            
        optimized_lines.append(line)
        
        # Add universal optimization for test methods
        if 'def test_' in line and not line.strip().startswith('#'):
            # Add universal test method docstring
            test_name = line.split('def ')[1].split('(')[0]
            optimized_lines.append('    """UNIVERSAL test for maximum coverage."""')
    
    return '\n'.join(optimized_lines)

def _fix_imports_for_universal_compatibility(code: str, target_root: pathlib.Path, analysis: Dict[str, Any]) -> str:
    """Fix imports for universal project compatibility."""
    lines = code.splitlines()
    fixed_lines = []
    
    # Add universal import setup at the top
    fixed_lines.append('# UNIVERSAL IMPORT SETUP - Works with any project structure')
    fixed_lines.append('import sys')
    fixed_lines.append('import os')
    fixed_lines.append(f'sys.path.insert(0, r"{target_root}")')
    fixed_lines.append('')
    
    # Get project structure info (currently not altering imports further)
    for line in lines:
        if any(keyword in line for keyword in ['sys.path.insert', 'UNIVERSAL IMPORT SETUP']):
            continue
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

# ------------------------ RESILIENT CONTEXT GATHERING ------------------------
def _gen__normalize_imports(imports: List[Any]) -> List[Dict[str, Any]]:
    """
    Accept heterogeneous imports (str OR dict) and return a list of dicts:
    - str  -> {"type":"import","modules":[str], "file":""}
    - dict -> kept as-is (ensuring type/file keys)
    - other -> coerced to str in the same wrapper
    """
    norm: List[Dict[str, Any]] = []
    for imp in imports or []:
        if isinstance(imp, dict):
            if "type" not in imp and ("module" in imp or "modules" in imp):
                imp = {"type": "import" if "modules" in imp else "import_from", **imp}
            if "file" not in imp:
                imp["file"] = ""
            norm.append(imp)
        elif isinstance(imp, str):
            norm.append({"type": "import", "modules": [imp], "file": ""})
        else:
            norm.append({"type": "import", "modules": [str(imp)], "file": ""})
    return norm

def _gen__read_text_safe(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _gen__maybe_under_root(target_root: pathlib.Path, rel_or_abs: str) -> pathlib.Path:
    p = pathlib.Path(rel_or_abs)
    if p.is_file():
        return p
    q = (target_root / rel_or_abs).resolve()
    return q if q.is_file() else p

def _gen__index(items: List[Dict[str, Any]], name_key: str) -> Dict[str, Tuple[str, int, int]]:
    idx: Dict[str, Tuple[str, int, int]] = {}
    for it in items or []:
        if not isinstance(it, dict):
            continue
        name = it.get(name_key)
        if not name:
            continue
        cls = it.get("class")
        if cls:
            name = f"{cls}.{name}"
        file_path = it.get("file", "") or ""
        start = int(it.get("lineno", 1) or 1)
        end = int(it.get("end_lineno", start) or start)
        idx[name] = (file_path, start, end)
    return idx

def _gather_universal_context(target_root: pathlib.Path, analysis: Dict[str, Any],
                              focus_names: List[str], max_bytes: int = 120000) -> str:
    """Gather COMPLETE code context (resilient to string/dict imports)."""
    imports_norm = _gen__normalize_imports(analysis.get("imports", []))

    # Build indices
    function_index = _gen__index(analysis.get("functions", []), "name")
    class_index    = _gen__index(analysis.get("classes", []), "name")
    method_index   = _gen__index(analysis.get("methods", []), "name")
    route_index    = _gen__index(analysis.get("routes", []), "handler")

    # Collect files containing requested targets
    relevant_files: Set[str] = set()
    for target_name in focus_names or []:
        for idx in (function_index, class_index, method_index, route_index):
            if target_name in idx:
                file_rel, _, _ = idx[target_name]
                if file_rel:
                    relevant_files.add(file_rel)
                break

    # Also include files that import those targets
    for imp in imports_norm:
        imp_file = imp.get("file", "") or ""
        modules = imp.get("modules", [])
        if isinstance(modules, str):
            modules_text = modules
        else:
            try:
                modules_text = ", ".join(m for m in modules if isinstance(m, str))
            except Exception:
                modules_text = str(modules)
        if imp_file and any(target_file in modules_text for target_file in relevant_files):
            relevant_files.add(imp_file)

    # If nothing matched, fall back to all files present in analysis
    if not relevant_files:
        for coll in ("functions", "classes", "methods", "routes"):
            for it in analysis.get(coll, []) or []:
                if isinstance(it, dict):
                    fp = it.get("file")
                    if fp:
                        relevant_files.add(fp)

    # Assemble the context
    context_parts: List[str] = []

    # Header with structure info
    structure = analysis.get("project_structure", {}) or {}
    header = (
        "# UNIVERSAL PROJECT STRUCTURE\n"
        f"# Root: {structure.get('root', 'Unknown')}\n"
        f"# Detected Packages: {', '.join(structure.get('package_names', []))}\n"
        f"# Total Modules: {len(structure.get('module_paths', {}) or {})}\n"
        f"# Relevant Files: {len(relevant_files)}\n\n"
    )
    context_parts.append(header)
    current_size = len(header)

    # Dump file contents until cap
    for file_rel in sorted(relevant_files):
        path = _gen__maybe_under_root(target_root, file_rel)
        content = _gen__read_text_safe(path)
        if not content:
            continue
        snippet = (
            f"# FILE: {file_rel}\n"
            f"# FULL CONTENT FOR UNIVERSAL COMPATIBILITY\n"
            f"{content}\n\n{'=' * 80}\n\n"
        )
        if current_size + len(snippet) > max_bytes:
            head = content[:1000]
            fallback = (
                f"# FILE: {file_rel}\n"
                f"# FIRST 1000 CHARACTERS\n{head}...\n\n{'=' * 80}\n\n"
            )
            if current_size + len(fallback) > max_bytes:
                break
            context_parts.append(fallback)
            current_size += len(fallback)
            break
        context_parts.append(snippet)
        current_size += len(snippet)

    full_context = "".join(context_parts)

    coverage_header = (
        "\n# UNIVERSAL CODE CONTEXT FOR ANY PROJECT STRUCTURE\n"
        f"# Targets: {len(focus_names or [])} functions/classes/methods\n"
        f"# Files: {len(relevant_files)} source files\n"
        "# Strategy: Test ALL code paths with REAL IMPORTS\n"
        "# Goal: Maximum line coverage and branch coverage\n"
        "# UNIVERSAL COMPATIBILITY: Works with flat, nested, or package structures\n\n"
    )
    full_context = coverage_header + full_context

    # Add gap-focused context if applicable
    if is_gap_focused_mode():
        gap_context = get_coverage_context_for_prompts()
        if gap_context:
            full_context = gap_context + "\n\n" + full_context

    if len(full_context) > max_bytes:
        full_context = full_context[:max_bytes] + "\n# ... (truncated for context limits)"
    return full_context

# -------------------- AUTO-SANITIZER FOR PARAMETRIZE MISMATCHES --------------------
def _sanitize_parametrize_signature_mismatches(code: str) -> str:
    """
    Align any @pytest.mark.parametrize names with the test function signature
    to prevent collection errors like:
      'function uses no argument <name>'
    """
    deco_pattern = re.compile(
        r'(@pytest\.mark\.parametrize\(\s*([rRuU]?[\'"].*?[\'"]|\[.*?\]|\(.*?\))\s*,.*?\))',
        re.DOTALL
    )
    def_block = re.compile(r'(?:^\s*@.*\n)*^\s*def\s+(test_[A-Za-z0-9_]+)\s*\((.*?)\)\s*:', re.MULTILINE|re.DOTALL)

    def extract_param_names(deco: str) -> List[str]:
        m = re.search(r'@pytest\.mark\.parametrize\(\s*([rRuU]?[\'"].*?[\'"]|\[.*?\]|\(.*?\))\s*,', deco, re.DOTALL)
        if not m:
            return []
        first = m.group(1).strip()
        if first.startswith(("'", '"', "r'", 'r"', "u'", 'u"', "R'", 'R"', "U'", 'U"')):
            return [n.strip() for n in first.strip('rRuU')[1:-1].split(",") if n.strip()]
        return re.findall(r'[\'"]([^\'"]+)[\'"]', first)

    out = code
    search_pos = 0
    while True:
        mdef = def_block.search(out, search_pos)
        if not mdef:
            break
        def_start = mdef.start()
        # look back up to 20 lines to gather nearby decorators
        lookback_start = out.rfind("\n", 0, def_start)
        lookback_start = max(0, out.rfind("\n", 0, lookback_start) if lookback_start != -1 else 0)
        window = out[lookback_start:def_start]
        decos = deco_pattern.findall(window)
        needed: List[str] = []
        for full, _ in decos:
            needed += extract_param_names(full)
        needed = list(dict.fromkeys([n for n in needed if n]))  # dedupe

        if needed:
            full_def = mdef.group(0)
            args_str = mdef.group(2).strip()
            arg_list = [a.strip() for a in args_str.split(",")] if args_str else []
            present = {a.split("=",1)[0].strip() for a in arg_list if a}
            to_add = [n for n in needed if n not in present]
            if to_add:
                new_args = ", ".join([a for a in arg_list if a] + to_add)
                replaced = full_def.replace(f"({args_str})", f"({new_args})")
                out = out[:mdef.start()] + replaced + out[mdef.end():]
                search_pos = mdef.start() + len(replaced)
                continue
        search_pos = mdef.end()
    return out
# ------------------------------------------------------------------------------------

def generate_all(analysis: Dict[str, Any], outdir: str = "tests/generated",
                focus_files: Optional[List[str]] = None):
    """Generate framework-aware test suite for any project structure."""
    from .enhanced_analysis_utils import (compact_analysis, enhance_coverage_targeting,
                                          filter_by_files, infer_required_packages,
                                          pip_install, prune_unavailable_targets)
    from .enhanced_prompt import build_prompt, files_per_kind, focus_for
    from .writer import update_manifest, write_text
    from .postprocess import cleanup_generated_test, postprocess_all_tests

    print("FRAMEWORK-AWARE test generation...")
    
    # Export full code AST (before any filtering)
    try:
        with open("ast_full_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        print("Full code AST exported: ast_full_analysis.json")
    except Exception as e:
        print(f"Warning: Failed to export full AST: {e}")

    # Apply gap-aware filtering if in gap-focused mode
    if is_gap_focused_mode():
        print("\n GAP-FOCUSED MODE: Analyzing coverage gaps...")
        analysis = apply_gap_aware_filtering(analysis)
        
        # Check if generation should be skipped
        if analysis.get("skip_generation"):
            print(" Coverage is adequate, skipping generation")
            return []
        
        print(f"Gap analysis complete: Targeting {len(analysis.get('functions', []))} uncovered functions, "
              f"{len(analysis.get('classes', []))} uncovered classes, "
              f"{len(analysis.get('methods', []))} uncovered methods")

        # Export gap-focused AST (after filtering - uncovered code only)
        try:
            with open("ast_gap_analysis.json", 'w') as f:
                json.dump(analysis, f, indent=2)
            print("Gap-focused AST exported: ast_gap_analysis.json")
        except Exception as e:
            print(f"Warning: Failed to export gap AST: {e}")

    output_dir = pathlib.Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_root = pathlib.Path(os.environ.get("TARGET_ROOT", "target"))
    if not target_root.exists():
        raise RuntimeError(f"Target directory not found: {target_root}")

    # Ensure target root is in Python path
    if str(target_root) not in sys.path:
        sys.path.insert(0, str(target_root))

    # DETECT FRAMEWORK for framework-aware generation
    detected_framework = _detect_framework(analysis, target_root)
    print(f"Detected framework: {detected_framework.upper()}")

    # Create framework-specific conftest
    conftest_path = _create_universal_conftest(output_dir, target_root, detected_framework)
    print(f"Created {detected_framework} conftest: {conftest_path}")
    
    # Simple change detection - always generate tests
    changed_files = set()
    deleted_files = set()
    print("Change detection disabled - generating all tests")
    
    force_generation = os.getenv("TESTGEN_FORCE", "").lower() in ["true", "1", "yes"]
    
    focus_file_set = set(focus_files or [])
    if not focus_file_set and not force_generation:
        focus_file_set = changed_files if changed_files else set()
    
    filtered_analysis, no_targets = filter_by_files(analysis, focus_file_set if focus_file_set else None)
    if no_targets:
        filtered_analysis = analysis
    
    compact = prune_unavailable_targets(compact_analysis(filtered_analysis))
    compact = enhance_coverage_targeting(compact)
    
    # Install ONLY external packages for real imports
    print("Analyzing required packages...")
    required_packages = infer_required_packages(compact)
    
    if required_packages:
        print(f"Installing {len(required_packages)} external packages...")
        pip_install(required_packages)
    else:
        print("No external packages need installation")
    
    total_targets = sum(len(compact.get(key, [])) 
                       for key in ["functions", "classes", "methods", "routes"])
    
    if total_targets == 0:
        if is_gap_focused_mode():
            print("Gap-focused analysis found no significant gaps to target")
            print("Your manual tests already provide good coverage!")
            return []  # Return empty list, but this is SUCCESS
        else:
            raise RuntimeError("No testable targets found")
    
    print(f"UNIVERSAL COVERAGE TARGETS:")
    print(f"Functions: {len(compact.get('functions', []))}")
    print(f"Classes: {len(compact.get('classes', []))}")
    print(f"Methods: {len(compact.get('methods', []))}")
    print(f"Routes: {len(compact.get('routes', []))}")
    print(f"Total Targets: {total_targets}")
    print(f"Expected Coverage: Maximum")
    print(f"Project Structure: Universal compatibility")
    
    has_routes = bool(compact.get("routes"))

    # PLAIN PYTHON: Only unit tests, no integration/e2e
    if detected_framework == "python":
        test_kinds = ["unit"]
        print("   Plain Python detected: generating unit tests only (branch-aware)")
    else:
        test_kinds = ["unit", "integ"]
        if has_routes:
            test_kinds.append("e2e")

    compact_json = json.dumps(compact, separators=(",", ":"))
    generated_files = []
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # For plain Python: create source-to-test mapping
    source_to_test_map = {}
    if detected_framework == "python":
        source_to_test_map = _get_source_to_test_mapping(target_root, analysis)
        print(f"   Source-to-test mapping: {len(source_to_test_map)} source files")

        # Estimate branch coverage
        branch_estimate = _estimate_branch_coverage(compact)
        print(f"   Branch analysis: {branch_estimate['total_branches']} branches, "
              f"{branch_estimate['functions_with_raises']} functions can raise")

    for test_kind in test_kinds:
        num_files = files_per_kind(compact, test_kind, detected_framework)
        if num_files <= 0:
            continue

        print(f"Generating {num_files} {test_kind.upper()} test files for {detected_framework.upper()}...")

        for file_index in range(num_files):
            try:
                focus_label, focus_names, shard_targets = focus_for(compact, test_kind, file_index, num_files)

                if not focus_names:
                    continue

                print(f"Generating {test_kind} test {file_index + 1}/{num_files} for {len(focus_names)} targets")

                context = _gather_universal_context(target_root, filtered_analysis, focus_names)

                # Pass detected framework to prompt builder
                prompt_messages = build_prompt(test_kind, compact_json, focus_label,
                                              file_index, num_files, compact, context,
                                              framework=detected_framework)

                test_code = _generate_with_universal_retry(prompt_messages, max_attempts=3)

                # Fix imports for project structure
                test_code = _fix_imports_for_universal_compatibility(test_code, target_root, analysis)
                # Sanitize parametrization mismatches
                test_code = _sanitize_parametrize_signature_mismatches(test_code)
                # Clean up framework-specific issues (duplicate fixtures, wrong markers)
                test_code = cleanup_generated_test(test_code, detected_framework)

                filename = f"test_{test_kind}_{timestamp}_{file_index + 1:02d}.py"
                file_path = output_dir / filename

                write_text(file_path, test_code)
                generated_files.append(str(file_path))
                print(f"  {filename} - {len(focus_names)} targets")
                
            except Exception as e:
                print(f"Error generating {test_kind} test {file_index + 1}: {e}")
                traceback.print_exc()
    
    # Final post-processing pass on all generated files
    if generated_files:
        print(f"\nPost-processing {len(generated_files)} generated test files...")
        modified_count = postprocess_all_tests(output_dir, detected_framework)
        if modified_count > 0:
            print(f"  Post-processed {modified_count} files (removed duplicates/wrong markers)")

    change_summary = {
        "added_or_modified": len(changed_files),
        "deleted": len(deleted_files),
        "total_analyzed": len(changed_files) + len(deleted_files),
        "coverage_target": "Maximum",
        "detected_framework": detected_framework,
        "project_structure": analysis.get("project_structure", {}).get("package_names", [])
    }
    from .writer import update_manifest  # re-import here to be safe
    update_manifest(output_dir, generated_files, change_summary)

    if generated_files:
        print(f"\n{'='*50}")
        print(f"GENERATION COMPLETE: {len(generated_files)} test files")
        print(f"{'='*50}")
        print(f"  Framework: {detected_framework.upper()}")
        print(f"  Targets Covered: {total_targets}")
        print(f"  Coverage Target: Maximum with REAL IMPORTS")
        print(f"{'='*50}")

    return generated_files

def _validate_and_fix_test_code(code: str, filename: str) -> str:
    """Validate test code and attempt to fix common issues."""
    from .postprocess import massage, validate_code
    
    # First, try the normal massage process
    try:
        processed_code = massage(code)
        is_valid, error = validate_code(processed_code)
        
        if is_valid:
            return processed_code
        else:
            print(f"Massaged code still invalid for {filename}: {error}")
    except Exception as e:
        print(f"Error during massage for {filename}: {e}")
    
    # If massage failed, try basic fixes
    try:
        # Fix common indentation patterns
        lines = code.splitlines()
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Fix lines that should be indented after colons
            if i > 0 and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                prev_line = lines[i-1].rstrip()
                if prev_line.endswith(':') and not prev_line.startswith('#'):
                    line = '    ' + line
            
            fixed_lines.append(line)
        
        fixed_code = '\n'.join(fixed_lines)
        is_valid, error = validate_code(fixed_code)
        
        if is_valid:
            return fixed_code
        else:
            print(f"Basic fixes failed for {filename}: {error}")
    except Exception as e:
        print(f"Error during basic fixes for {filename}: {e}")
    
    # Last resort: return original code with warning
    print(f"All fixes failed for {filename}, using original code (may have syntax errors)")
    return f"# WARNING: This file may contain syntax errors\n# Generation system could not fix all issues\n\n{code}"

def main():
    """UNIVERSAL main entry point for any project structure."""
    parser = argparse.ArgumentParser(
        description="Generate UNIVERSAL pytest test suites for ANY PROJECT STRUCTURE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
UNIVERSAL COMPATIBILITY EXAMPLES:
  python -m src.gen --target ./my_project
  python -m src.gen --target ./backend --force
  python -m src.gen --target ./app --outdir ./tests
  python -m src.gen --target ./app --coverage-mode gap-focused 
  
SUPPORTED PROJECT STRUCTURES:
  - Flat structure (all files in root)
  - Nested structure (files in subdirectories) 
  - Package structure (with __init__.py files)
  - Mixed structures
  - Any Python project layout

COVERAGE MODES:
  - normal: Full test generation (default)
  - gap-focused: Generate tests only for uncovered code (requires coverage_gaps.json)
  
FEATURES:
  - Automatic project structure detection
  - Universal import handling
  - Real imports only (no stubs)
  - Maximum coverage target
  - Framework auto-detection
  - Gap-focused generation for targeted coverage improvement
"""
    )
    
    parser.add_argument(
        "--target",
        default=os.environ.get("TARGET_ROOT", "target"),
        help="Path to Python project to analyze (default: %(default)s)"
    )
    
    parser.add_argument(
        "--outdir", 
        default="tests/generated",
        help="Output directory for generated tests (default: %(default)s)"
    )
    
    parser.add_argument(
        "--coverage-mode",
        choices=["normal", "maximum", "universal", "gap-focused"],
        default=os.getenv("COVERAGE_MODE", "universal"),
        help="Coverage optimization mode (default: %(default)s)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true", 
        help="Force regeneration"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze code but don't generate tests"
    )
    
    parser.add_argument(
        "--use-orchestrator",
        action="store_true",
        help="Use the new orchestrator for test generation"
    )
    
    args = parser.parse_args()
    # Set UNIVERSAL environment variables
    if args.force:
        os.environ["TESTGEN_FORCE"] = "true"
    os.environ["TARGET_ROOT"] = args.target
    os.environ["COVERAGE_MODE"] = args.coverage_mode
    os.environ["UNIVERSAL_COMPATIBILITY"] = "true"
    
    # === ADD THIS: Enable gap-focused mode if requested ===
    if args.coverage_mode == "gap-focused":
        os.environ["GAP_FOCUSED_MODE"] = "true"
        print("Gap-focused mode enabled via --coverage-mode argument")
        
        # Check if coverage gaps file exists
        gaps_file = os.environ.get("COVERAGE_GAPS_FILE", "coverage_gaps.json")
        if not pathlib.Path(gaps_file).exists():
            print(f"Warning: Gap-focused mode requires {gaps_file}")
            print("Run coverage_gap_analyzer.py first to generate this file")
            print("Falling back to normal mode")
            os.environ["GAP_FOCUSED_MODE"] = "false"

    # Validate target
    target_path = pathlib.Path(args.target)
    if not target_path.exists():
        print(f"Target directory not found: {target_path}")
        return 1
    
    # UNIVERSAL: Find Python files recursively
    python_files = list(target_path.rglob("*.py"))
    if not python_files:
        print(f"No Python files found in: {target_path} (searched recursively)")
        return 1
    
    print(f"Found {len(python_files)} Python files in target directory")
    print(f"Project structure: Universal compatibility enabled")
    
    try:
        if args.use_orchestrator:
            # Use the new orchestrator
            print("Using new test generation orchestrator...")
            orchestrator = TestGenerationOrchestrator(
                target_root=target_path,
                output_dir=pathlib.Path(args.outdir)
            )
            
            result = orchestrator.orchestrate_generation(force_regeneration=args.force)
            
            if result["success"]:
                print("Test generation completed successfully using orchestrator!")
                return 0
            else:
                print(f"Test generation failed: {result['error']}")
                return 1
        
        # Legacy path using original enhanced_generate
        try:
            from src.analyzer import analyze_python_tree
        except ImportError:
            try:
                from analyzer import analyze_python_tree  
            except ImportError:
                print("Could not import analyzer module")
                return 1
        
        print(f"UNIVERSAL analysis for ANY PROJECT STRUCTURE in: {target_path}")
        analysis_result = analyze_python_tree(target_path)
        
        if args.dry_run:
            print("UNIVERSAL DRY RUN - Project analysis complete")
            print(f"UNIVERSAL Analysis Summary:")
            print(f"   Functions: {len(analysis_result.get('functions', []))}")
            print(f"   Classes: {len(analysis_result.get('classes', []))}")
            print(f"   Methods: {len(analysis_result.get('methods', []))}")
            print(f"   Routes: {len(analysis_result.get('routes', []))}")
            print(f"   Modules: {len(analysis_result.get('modules', []))}")
            print(f"   Packages: {len(analysis_result.get('project_structure', {}).get('package_names', []))}")
            print(f"   Coverage Mode: {args.coverage_mode}")
            print(f"   Universal Compatibility: ENABLED")
            return 0
        
        # Generate UNIVERSAL tests
        print(f"Starting UNIVERSAL test generation with {args.coverage_mode} coverage mode...")
        generated_files = generate_all(analysis_result, outdir=args.outdir)
        
        if generated_files:
            print(f"UNIVERSAL TEST GENERATION SUCCESSFUL!")
            print(f"UNIVERSAL Results:")
            print(f"   Generated: {len(generated_files)} test files")
            print(f"   Coverage Mode: {args.coverage_mode}")
            print(f"   Universal Compatibility: ENABLED")
            print(f"   Targets Covered: {sum(len(analysis_result.get(key, [])) for key in ['functions', 'classes', 'methods', 'routes'])}")
            print(f"   Project Structure: Universal handling enabled")
            
            print(f"Run UNIVERSAL Tests:")
            print(f"   Basic: python -m pytest {args.outdir} -v")
            print(f"   Coverage: python -m pytest {args.outdir} --cov=. --cov-report=html")
            print(f"   Universal: PYTHONPATH={args.target} python -m pytest {args.outdir}")
            
            return 0
        else:
            # In gap-focused mode, no targets means coverage is good - this is success!
            if args.coverage_mode == "gap-focused":
                print("No additional tests needed - coverage gaps are minimal")
                print("   This is expected when most code is already covered by manual tests")
                return 0  # Success
            else:
                print("No tests generated - this may indicate an issue with the source code")
                return 1
            
    except Exception as e:
        print(f"UNIVERSAL test generation failed: {e}")
        if os.getenv("TESTGEN_DEBUG", "0").lower() in ("1", "true"):
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

import re as _eg_re
from pathlib import Path as _eg_Path

def _eg__sanitize_duplicate_keyword_args(_code: str) -> str:
    """
    Remove duplicate keyword arguments within common calls to avoid:
      SyntaxError: keyword argument repeated
    Targets Mock(...) / MagicMock(...) and dedupes common kwargs.
    Append-only, safe for re-imports.
    """
    def _dedupe_kwargs_in_args(_args_src: str, kw: str) -> str:
        # keep the first occurrence of "kw=..." and drop subsequent ones
        pattern = _eg_re.compile(rf'(,?\s*{kw}\s*=\s*[^,)\n]+)')
        matches = list(pattern.finditer(_args_src))
        if len(matches) <= 1:
            return _args_src
        # remove later occurrences (including any leading comma)
        out = _args_src
        for i, m in enumerate(matches):
            if i == 0:
                continue
            s, e = m.span()
            seg = _args_src[s:e]
            out = out.replace(seg, '', 1)
        return out

    def _process_calls(func_name: str, text: str) -> str:
        # naive but robust enough for generator output
        pattern = _eg_re.compile(rf'({func_name}\()\s*(.*?)\s*(\))', _eg_re.DOTALL)
        def _repl(m):
            head, args, tail = m.group(1), m.group(2), m.group(3)
            for kw in ('name', 'spec', 'return_value', 'side_effect', 'content_type', 'status_code', 'headers'):
                args = _dedupe_kwargs_in_args(args, kw)
            return head + args + tail
        # local rename to avoid shadowing
        _dedupe_kwargs_in_args = _dedupe_kwargs_in_args
        return pattern.sub(_repl, text)

    _code = _process_calls('Mock', _code)
    _code = _process_calls('MagicMock', _code)
    return _code


def _eg__postprocess_generated_files(_generated_files):
    """
    Append-only sanitation pass over generated test files:
      * align @pytest.mark.parametrize names with function signature
      * remove duplicate keyword args in common calls (Mock, MagicMock)
    """
    if not _generated_files:
        return _generated_files

    # lazy import to reuse your existing validators if available
    try:
        from .postprocess import validate_code as _eg_validate
    except Exception:
        def _eg_validate(code: str):
            try:
                import ast
                ast.parse(code)
                return True, ""
            except SyntaxError as e:
                return False, f"syntax error: {e}"

    for _f in _generated_files:
        try:
            p = _eg_Path(_f)
            if not p.exists():
                continue
            original = p.read_text(encoding="utf-8", errors="ignore")
            updated = original

            # fix parametrize/signature mismatches (uses function defined earlier in your file)
            try:
                updated = _sanitize_parametrize_signature_mismatches(updated)
            except Exception:
                # best effort; keep going
                pass

            # remove duplicate keyword args in Mock/MagicMock
            try:
                updated = _eg__sanitize_duplicate_keyword_args(updated)
            except Exception:
                pass

            # only write if changed and still valid python
            if updated != original:
                ok, err = _eg_validate(updated)
                if ok:
                    p.write_text(updated, encoding="utf-8")
                else:
                    # fall back to original if our sanitation broke syntax
                    # (extremely unlikely, but safe)
                    p.write_text(original, encoding="utf-8")
        except Exception:
            # Never fail the run for sanitation; just continue
            continue

    return _generated_files


# Wrap your existing generate_all with a post-write sanitizer WITHOUT deleting any lines.
# We preserve the original and override the symbol by re-defining it here (append-only).
try:
    _eg__orig_generate_all = generate_all
except NameError:
    _eg__orig_generate_all = None

def generate_all(analysis: Dict[str, Any], outdir: str = "tests/generated",
                 focus_files: Optional[List[str]] = None):
    """
    Append-only wrapper that calls the original generate_all, then post-processes
    the generated test files to prevent collection-time SyntaxErrors.
    """
    if _eg__orig_generate_all is None:
        # If, for some reason, the original isn't present, bail gracefully.
        return []

    _files = _eg__orig_generate_all(analysis, outdir=outdir, focus_files=focus_files)
    return _eg__postprocess_generated_files(_files)

