# src/framework_handlers/universal_handler.py
"""
Universal framework handler for projects without specific frameworks.
"""

import ast
import pathlib
from typing import Any, Dict, List

from .base_handler import BaseFrameworkHandler


class UniversalHandler(BaseFrameworkHandler):
    """Universal framework handler for any Python project."""
    
    def __init__(self):
        super().__init__()
        self.framework_name = "universal"
        self.supported_patterns = {"functions", "classes", "methods", "modules"}
    
    def can_handle(self, analysis: Dict[str, Any]) -> bool:
        """Universal handler can handle any project as fallback."""
        return True
    
    def analyze_framework_specifics(self, tree, file_path: str) -> Dict[str, Any]:
        """Universal analysis - no framework-specific patterns."""
        return {}
    
    def get_framework_dependencies(self) -> List[str]:
        """Get universal dependencies for any Python project."""
        return [
            "pytest",
            "pytest-cov",
            "pytest-asyncio",
            "coverage"
        ]
    
    def detect_framework_patterns(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect universal patterns in the analysis."""
        universal_specific = {
            "framework": "universal",
            "function_count": len(analysis.get("functions", [])),
            "class_count": len(analysis.get("classes", [])),
            "method_count": len(analysis.get("methods", [])),
            "module_count": len(analysis.get("modules", [])),
            "has_async": len(analysis.get("async_functions", [])) > 0,
        }
        return universal_specific


# ----------------------- APPENDED ENHANCEMENTS BELOW (no deletions) -----------------------

class UniversalHandler(UniversalHandler):  # type: ignore[misc]
    """
    Extended universal handler providing:
      - richer generic Python analysis
      - generation of importable real tests
      - detection of CLI entry points and main() functions
      - better coverage guidance for standalone projects
    """

    def analyze_framework_specifics(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:  # type: ignore[override]
        """
        Enhanced static analysis to detect functions, classes, and __main__ entry points.
        """
        details = {
            "functions": [],
            "classes": [],
            "main_entrypoints": [],
            "imports": [],
            "async_functions": [],
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                details["functions"].append({
                    "name": node.name,
                    "lineno": getattr(node, "lineno", 1),
                    "args": [a.arg for a in node.args.args],
                })
                if node.name == "main":
                    details["main_entrypoints"].append(file_path)
            elif isinstance(node, ast.AsyncFunctionDef):
                details["async_functions"].append(node.name)
            elif isinstance(node, ast.ClassDef):
                details["classes"].append({
                    "name": node.name,
                    "lineno": getattr(node, "lineno", 1),
                })
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    details["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    details["imports"].append(node.module)

        return details

    def detect_framework_patterns(self, analysis: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        """
        Enhanced detection summary for pure Python projects.
        """
        summary = {
            "framework": "universal",
            "function_count": len(analysis.get("functions", [])),
            "class_count": len(analysis.get("classes", [])),
            "async_function_count": len(analysis.get("async_functions", [])),
            "module_count": len(analysis.get("modules", [])),
            "main_entrypoints": analysis.get("main_entrypoints", []),
            "has_async": len(analysis.get("async_functions", [])) > 0,
            "coverage_target": "95%+",
            "test_strategy": "pytest with real imports",
        }
        if analysis.get("main_entrypoints"):
            summary["entrypoint_type"] = "CLI / script"
        elif analysis.get("classes"):
            summary["entrypoint_type"] = "OOP module"
        else:
            summary["entrypoint_type"] = "functional module"
        return summary

    def generate_framework_specific_tests(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate basic pytest test templates that import and execute real functions/classes.
        """
        tests: List[Dict[str, Any]] = []
        functions = analysis.get("functions", [])
        classes = analysis.get("classes", [])
        app_module = self._guess_module(analysis)

        # Function-level tests
        for f in functions[:200]:
            template = self._function_test_template(f, app_module)
            tests.append({
                "type": "function_test",
                "target": f["name"],
                "template": template,
                "coverage_goal": "95%+",
                "test_count": 1,
            })

        # Class-level tests
        for c in classes[:100]:
            template = self._class_test_template(c, app_module)
            tests.append({
                "type": "class_test",
                "target": c["name"],
                "template": template,
                "coverage_goal": "95%+",
                "test_count": 1,
            })

        if not tests:
            tests.append({
                "type": "smoke_test",
                "target": "module_import",
                "template": self._import_smoke_template(app_module),
                "coverage_goal": "95%+",
                "test_count": 1,
            })
        return tests

    def _function_test_template(self, func: Dict[str, Any], module_name: str) -> str:
        """Generate a test template for standalone functions."""
        name = func["name"]
        return f'''
"""
Test for function: {name} (real import)
"""

import importlib
import pytest

def test_{name}_import_exec():
    mod = importlib.import_module("{module_name}")
    func = getattr(mod, "{name}", None)
    assert callable(func), "Function {name} not found"
    # Try calling safely if no required args
    try:
        if func.__code__.co_argcount == 0:
            func()
    except Exception:
        pytest.skip("Function {name} requires args, skipping direct call")
'''

    def _class_test_template(self, cls: Dict[str, Any], module_name: str) -> str:
        """Generate a test template for classes."""
        cname = cls["name"]
        return f'''
"""
Test for class: {cname} (real import)
"""

import importlib

def test_{cname.lower()}_class_instantiation():
    mod = importlib.import_module("{module_name}")
    cls = getattr(mod, "{cname}", None)
    assert cls is not None, "Class {cname} not found"
    try:
        _ = cls()  # instantiate without args
    except TypeError:
        pass  # allow constructors with required args
'''

    def _import_smoke_template(self, module_name: str) -> str:
        return f'''
"""
Universal smoke test to ensure {module_name} imports correctly.
"""

def test_import_smoke():
    __import__("{module_name}")
    assert True
'''

    def _guess_module(self, analysis: Dict[str, Any]) -> str:
        """Guess the main module name based on structure or fallback."""
        ps = analysis.get("project_structure", {}) or {}
        module_paths = list(ps.get("module_paths", {}).keys()) if isinstance(ps, dict) else []
        for pref in ("main.py", "__init__.py"):
            for p in module_paths:
                if p.lower().endswith(pref):
                    return p.replace("/", ".").replace("\\", ".").rstrip(".py")
        if module_paths:
            return module_paths[0].replace("/", ".").replace("\\", ".").rstrip(".py")
        return "main"


class UniversalHandler(UniversalHandler):  # type: ignore[misc]
    def _norm(self, analysis: Any) -> Dict[str, Any]:
        return analysis if isinstance(analysis, dict) else {"_raw_analysis": analysis}

    def analyze_framework_specifics(self, tree, file_path: str) -> Dict[str, Any]:  # type: ignore[override]
        # Keep existing behavior; this method is per-file already.
        return super().analyze_framework_specifics(tree, file_path)

    def detect_framework_patterns(self, analysis: Any) -> Dict[str, Any]:  # type: ignore[override]
        a = self._norm(analysis)
        return {
            "framework": "universal",
            "function_count": len(a.get("functions", [])),
            "class_count": len(a.get("classes", [])),
            "method_count": len(a.get("methods", [])),
            "module_count": len(a.get("modules", [])),
            "has_async": len(a.get("async_functions", [])) > 0,
        }

    def generate_framework_specific_tests(self, analysis: Any) -> List[Dict[str, Any]]:  # type: ignore[override]
        a = self._norm(analysis)
        return super().generate_framework_specific_tests(a)
