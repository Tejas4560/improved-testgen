# src/gen/enhanced_analysis_utils.py - UNIVERSAL VERSION
import importlib.util
import json
import os
import pathlib
import random
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from .env import norm_rel

# Import framework manager
from ..framework_handlers.manager import FrameworkManager

# Universal package mappings
COMMON_PKG_ALIASES = {
    "pptx": "python-pptx", "flask": "flask", "flask_bcrypt": "flask-bcrypt",
    "flask_login": "flask-login", "flask_sqlalchemy": "flask-sqlalchemy",
    "flask_wtf": "flask-wtf", "httpx": "httpx", "openai": "openai",
    "python_dotenv": "python-dotenv", "requests": "requests", "wtforms": "wtforms",
    "sqlalchemy": "SQLAlchemy", "django": "Django", "PIL": "Pillow",
    "bs4": "beautifulsoup4", "yaml": "PyYAML", "cv2": "opencv-python",
    "sklearn": "scikit-learn", "pandas": "pandas", "numpy": "numpy",
}


DENY_GENERIC = {"tests", "test", "migrations", "__pycache__", "testing", "test_"}

def _is_local_module(top: str, analysis: Dict[str, Any]) -> bool:
    """UNIVERSAL: Check if a module is local to the project."""
    if 'test' in top.lower():
        return True
    

DENY_TOPS = set(DENY_GENERIC) | {
    "__future__", "__main__", "builtins", "typing", "types", "dataclasses",
}

def _is_stdlib(top: str) -> bool:
    """Check if a module is part of Python standard library."""
    if hasattr(sys, "stdlib_module_names"):
        return top in sys.stdlib_module_names
    
    stdlib_modules = {
        "os", "sys", "re", "json", "pathlib", "math", "itertools",
        "functools", "typing", "subprocess", "datetime", "time",
        "collections", "dataclasses", "ast", "logging", "unittest",
        "argparse", "asyncio", "threading", "sqlite3", "email",
    }
    return top in stdlib_modules

def _is_local_module(top: str, analysis: Dict[str, Any]) -> bool:
    """UNIVERSAL: Check if a module is local to the project."""
    project_structure = analysis.get("project_structure", {})
    package_names = project_structure.get("package_names", [])
    module_paths = project_structure.get("module_paths", {})
    
    # Check if it's a detected package
    if top in package_names:
        return True
    
    # Check if it's in module paths
    if top in module_paths:
        return True
    
    # Check common local module patterns
    local_patterns = [
        top in ['app', 'main', 'application', 'server', 'api', 'backend', 'core', 'project'],
        any(top in module for module in module_paths),
        any(top == pkg for pkg in package_names),
    ]
    
    return any(local_patterns)

def compact_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create analysis with ALL targets - UNIVERSAL approach."""
    
    # Get ALL targets without any filtering
    functions = analysis.get("functions", [])
    classes = analysis.get("classes", [])
    methods = analysis.get("methods", [])
    routes = analysis.get("routes", [])
    nested_functions = analysis.get("nested_functions", [])
    fastapi_routes = analysis.get("fastapi_routes", [])
    
    # Include nested functions in functions list for testing
    all_functions = functions + nested_functions
    all_routes = routes + fastapi_routes
    
    # Sort by file and line number for logical organization
    all_functions = sorted(all_functions, key=lambda x: (x.get("file", ""), x.get("lineno", 0)))
    classes = sorted(classes, key=lambda x: (x.get("file", ""), x.get("lineno", 0)))
    methods = sorted(methods, key=lambda x: (x.get("file", ""), x.get("lineno", 0)))
    all_routes = sorted(all_routes, key=lambda x: (x.get("file", ""), x.get("lineno", 0)))
    
    print(f"UNIVERSAL TARGET INCLUSION:")
    print(f"   Functions: {len(all_functions)} (including {len(nested_functions)} nested)")
    print(f"   Classes: {len(classes)}")
    print(f"   Methods: {len(methods)}")
    print(f"   Routes: {len(all_routes)}")
    print(f"   Project Packages: {len(analysis.get('project_structure', {}).get('package_names', []))}")
    
    return {
        "functions": all_functions,
        "classes": classes,
        "methods": methods,
        "routes": all_routes,
        "modules": sorted(set(analysis.get("modules", []))),
        "django_patterns": analysis.get("django_patterns", {}),
        "imports": analysis.get("imports", []),
        "project_structure": analysis.get("project_structure", {}),  # Include for universal handling
    }

def filter_by_files(analysis: Dict[str, Any], focus_files: Optional[Set[str]]) -> Tuple[Dict[str, Any], bool]:
    """Filter analysis to focus on specific files - UNIVERSAL approach."""
    if not focus_files:
        return analysis, False
    
    focus_normalized = {norm_rel(f) for f in focus_files}
    focus_basenames = {pathlib.Path(f).name for f in focus_normalized}
    
    def should_keep(entry):
        file_path = norm_rel(entry.get("file") or "")
        file_basename = pathlib.Path(file_path).name
        return (file_path in focus_normalized or
                file_basename in focus_basenames or
                any(focus in file_path for focus in focus_normalized))
    
    # Also track imports from focus files
    focus_imports = []
    for imp in analysis.get("imports", []):
        if isinstance(imp, dict):
            if should_keep(imp):
                focus_imports.append(imp)
        else:
            # keep string imports as-is
            focus_imports.append(imp)
    
    filtered = {
        "functions": [item for item in analysis.get("functions", []) if should_keep(item)],
        "classes": [item for item in analysis.get("classes", []) if should_keep(item)],
        "methods": [item for item in analysis.get("methods", []) if should_keep(item)],
        "routes": [item for item in analysis.get("routes", []) if should_keep(item)],
        "modules": analysis.get("modules", []),
        "django_patterns": analysis.get("django_patterns", {}),
        "imports": focus_imports,
        "project_structure": analysis.get("project_structure", {}),  # Keep structure for universal handling
    }
    
    has_targets = any(filtered[key] for key in ["functions", "classes", "methods", "routes"])
    return (filtered, not has_targets)

def enhance_coverage_targeting(compact: Dict[str, Any]) -> Dict[str, Any]:
    """UNIVERSAL: Return targets as-is for maximum coverage."""
    print("Using universal targeting - all targets included")
    return compact

def infer_required_packages(compact: Dict[str, Any]) -> List[str]:
    """UNIVERSAL: Infer required third-party packages, excluding local modules."""
    modules = compact.get("modules", [])
    imports = compact.get("imports", [])
    project_structure = compact.get("project_structure", {})
    required_packages = set()
    
    # Analyze both modules and individual imports
    all_imports = set(modules)
    for imp in imports:
        if imp.get("type") == "import":
            all_imports.update(imp.get("modules", []))
        elif imp.get("type") == "import_from":
            if imp.get("module"):
                all_imports.add(imp["module"])
    
    for module_name in all_imports:
        if not module_name or module_name.startswith('_'):
            continue
            
        top_module = module_name.split(".")[0].strip()
        
        if (not top_module or
            top_module in DENY_TOPS or
            any(char.isupper() for char in top_module)):
            continue
        
        if _is_stdlib(top_module):
            continue
        
        # UNIVERSAL: Skip local modules using project structure detection
        if _is_local_module(top_module, compact):
            print(f"   {top_module}: Local module (skipped pip install)")
            continue
        
        package_name = COMMON_PKG_ALIASES.get(top_module, top_module)
        required_packages.add(package_name)
    
    packages_list = sorted(required_packages, key=str.lower)
    
    print(f"Inferred {len(packages_list)} external packages: {', '.join(packages_list)}")
    return packages_list

def pip_install(packages: List[str]) -> None:
    """Install packages with robust error handling - UNIVERSAL approach."""
    if not packages:
        print("No external packages to install.")
        return
    
    print(f"Installing packages: {', '.join(packages)}")
    
    successful_installs = []
    failed_installs = []
    
    for package in packages:
        if not package or not package.strip():
            continue
        
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--disable-pip-version-check",
            "--no-input",
            "--quiet",
            package
        ]
        
        try:
            subprocess.check_call(cmd,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
            successful_installs.append(package)
            print(f"   {package}")
        except subprocess.CalledProcessError:
            failed_installs.append(package)
            print(f"   {package}: Installation failed")
        except Exception as e:
            failed_installs.append(package)
            print(f"   {package}: {e}")
    
    if successful_installs:
        print(f"Successfully installed {len(successful_installs)} packages.")
    if failed_installs:
        print(f"Failed to install {len(failed_installs)} packages (tests will use available packages)")

# Keep other functions the same as they're already universal
def prune_unavailable_targets(compact: Dict[str, Any]) -> Dict[str, Any]:
    """Remove targets that depend on unavailable heavy dependencies."""
    return compact  # Universal: Don't prune for maximum compatibility

def validate_analysis_quality(analysis: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate that analysis contains sufficient data for test generation."""
    functions = analysis.get("functions", [])
    classes = analysis.get("classes", [])
    methods = analysis.get("methods", [])
    routes = analysis.get("routes", [])
    
    total_targets = len(functions) + len(classes) + len(methods) + len(routes)
    
    if total_targets == 0:
        return False, "No testable targets found"
    
    files_with_targets = set()
    for item in functions + classes + methods + routes:
        if item.get("file"):
            files_with_targets.add(item["file"])
    
    if len(files_with_targets) == 0:
        return False, "No files contain identifiable targets"
    
    status_msg = f"Analysis valid: {total_targets} targets across {len(files_with_targets)} files"
    
    return True, status_msg


def _normalize_imports_list(imports: List[Any]) -> List[Any]:
    """
    Accepts a heterogeneous list (strings OR dicts) and returns the same list,
    but guarantees entries are either str or dict with the expected keys.
    """
    norm: List[Any] = []
    for imp in imports or []:
        if isinstance(imp, (str, bytes)):
            # keep simple module strings as str
            norm.append(imp if isinstance(imp, str) else imp.decode("utf-8", "ignore"))
        elif isinstance(imp, dict):
            norm.append(imp)
        else:
            # Unknown shape; stringify to keep things moving without crashing
            norm.append(str(imp))
    return norm

def infer_required_packages(compact: Dict[str, Any]) -> List[str]:  # type: ignore[override]
    """
    SAFER override:
    - Handles imports as strings *or* dicts.
    - Ignores stdlib and clearly-local modules.
    - Maps common aliases (e.g., PIL -> Pillow).
    """
    modules = compact.get("modules", []) or []
    imports_raw = compact.get("imports", []) or []
    imports = _normalize_imports_list(imports_raw)

    required_packages: Set[str] = set()

    # Collect all imported module names
    all_imports: Set[str] = set()
    # 1) simple modules list
    for m in modules:
        if isinstance(m, str) and m.strip():
            all_imports.add(m.strip())

    # 2) analyzer "imports" (strings or dicts)
    for imp in imports:
        if isinstance(imp, str):
            if imp.strip():
                all_imports.add(imp.strip())
            continue
        if not isinstance(imp, dict):
            # fall back to string form
            all_imports.add(str(imp))
            continue

        typ = imp.get("type")
        if typ == "import":
            mods = imp.get("modules", [])
            for m in mods or []:
                if isinstance(m, str) and m.strip():
                    all_imports.add(m.strip())
        elif typ == "import_from":
            mod = imp.get("module")
            if isinstance(mod, str) and mod.strip():
                all_imports.add(mod.strip())
        else:
            # unknown shape; try best-effort keys
            for k in ("module", "modules"):
                v = imp.get(k)
                if isinstance(v, str) and v.strip():
                    all_imports.add(v.strip())
                elif isinstance(v, list):
                    for s in v:
                        if isinstance(s, str) and s.strip():
                            all_imports.add(s.strip())

    # Decide which need pip installs
    for module_name in sorted(all_imports):
        if not module_name or module_name.startswith("_"):
            continue

        top = module_name.split(".", 1)[0].strip()
        if (not top) or (top in DENY_TOPS) or any(ch.isupper() for ch in top):
            continue
        if _is_stdlib(top):
            continue
        if _is_local_module(top, compact):
            print(f"   {top}: Local module (skipped pip install)")
            continue

        pkg = COMMON_PKG_ALIASES.get(top, top)
        required_packages.add(pkg)

    packages_list = sorted(required_packages, key=str.lower)
    if packages_list:
        print(f"Inferred {len(packages_list)} external packages: {', '.join(packages_list)}")
    else:
        print("Inferred 0 external packages.")
    return packages_list
