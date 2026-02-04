# src/analyzer.py - UNIVERSAL VERSION for any project structure
import argparse
import ast
import json
import os
import pathlib
from typing import Any, Dict, List, Set, Tuple

# Import framework handlers
from .framework_handlers.manager import FrameworkManager

# Minimal skipping - only truly problematic directories
SKIP_DIR_NAMES = {
    ".git", ".github", ".venv", "venv", "env", "node_modules", 
    "__pycache__", ".mypy_cache", ".pytest_cache", "tests/generated", "tests","test"
}

def read_text(p: pathlib.Path) -> str:
    """Safely read file content with comprehensive error handling."""
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Warning: Could not read {p}: {e}")
        return ""


def _should_skip(p: pathlib.Path, root: pathlib.Path) -> bool:
    """Check if path should be skipped - UNIVERSAL skipping logic."""
    try:
        rel = p.relative_to(root)
    except Exception:
        return False
    
    # Enhanced test directory skipping - ANY directory containing 'test'
    rel_parts = str(rel).lower().split('/')
    if any('test' in part for part in rel_parts):
        return True
    
    # Skip generated test directory to avoid circular analysis
    if "tests/generated" in str(rel):
        return True
    
    
    # Check each part of the path
    for part in rel.parts:
        if part in SKIP_DIR_NAMES:
            return True
        # Skip only truly hidden directories
        if part.startswith('.') and part not in {'.', '..'}:
            return True
    
    return False

def _should_skip_file(file_path: pathlib.Path) -> bool:
    """Check if specific file should be skipped."""
    filename = file_path.name.lower()
    return filename in {'wsgi.py', 'asgi.py', 'manage.py'}

def _extract_route_info(dec) -> Dict[str, Any]:
    """Extract route information from ALL web framework decorators."""
    info = {}
    try:
        # Handle @router.get("/path") style (FastAPI, Starlette)
        if hasattr(dec, 'func'):
            func = dec.func
            if hasattr(func, 'attr'):
                if func.attr in {"get", "post", "put", "patch", "delete", "options", "head"}:
                    info["method"] = func.attr
                    
                # Extract path from first argument
                if hasattr(dec, "args") and dec.args:
                    arg0 = dec.args[0]
                    if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                        info["path"] = arg0.value
        
        # Handle @app.route("/path") style (Flask)
        if isinstance(dec, ast.Call):
            if (isinstance(dec.func, ast.Attribute) and 
                dec.func.attr in {"route", "as_view", "add_url_rule"}):
                info["is_view"] = True
                if dec.args:
                    arg0 = dec.args[0]
                    if isinstance(arg0, ast.Constant):
                        info["path"] = arg0.value
        
        # Handle @api.route() style
        if (isinstance(dec, ast.Call) and
            isinstance(dec.func, ast.Attribute) and
            'route' in dec.func.attr.lower()):
            info["is_view"] = True
            if dec.args:
                arg0 = dec.args[0]
                if isinstance(arg0, ast.Constant):
                    info["path"] = arg0.value
                        
    except Exception:
        pass
    return info

def _analyze_class_methods(cls_node: ast.ClassDef, file_path: str) -> List[Dict[str, Any]]:
    """Extract ALL methods from a class including nested classes and complex methods."""
    methods = []
    
    for item in cls_node.body:
        # Regular methods
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_info = {
                "name": item.name,
                "class": cls_node.name,
                "file": file_path,
                "lineno": getattr(item, "lineno", 1),
                "end_lineno": getattr(item, "end_lineno", getattr(item, "lineno", 1)),
                "is_async": isinstance(item, ast.AsyncFunctionDef),
                "is_property": False,
                "is_classmethod": False,
                "is_staticmethod": False,
                "is_private": item.name.startswith("_") and not item.name.startswith("__"),
                "is_special": item.name.startswith("__") and item.name.endswith("__"),
                "args_count": len(item.args.args) if hasattr(item, 'args') else 0,
                "has_decorators": bool(getattr(item, "decorator_list", [])),
            }
            
            # Check decorators
            for dec in getattr(item, "decorator_list", []):
                if isinstance(dec, ast.Name):
                    if dec.id == "property":
                        method_info["is_property"] = True
                    elif dec.id == "classmethod":
                        method_info["is_classmethod"] = True
                    elif dec.id == "staticmethod":
                        method_info["is_staticmethod"] = True
            
            methods.append(method_info)
        
        # Nested classes - recurse into them
        elif isinstance(item, ast.ClassDef):
            nested_methods = _analyze_class_methods(item, file_path)
            for nested_method in nested_methods:
                nested_method["class"] = f"{cls_node.name}.{nested_method['class']}"
            methods.extend(nested_methods)
    
    return methods

def analyze_python_tree(root: pathlib.Path) -> Dict[str, Any]:
    """
    UNIVERSAL analysis of Python codebase for any project structure.
    
    Returns detailed information about ALL code elements.
    """
    # UNIVERSAL: Find ALL Python files recursively
    files: List[pathlib.Path] = [
    p for p in root.rglob("*.py") 
    if p.is_file() and not _should_skip(p, root) and not _should_skip_file(p)
    ]
        
    print(f"Analyzing {len(files)} Python files in project...")
    
    out = {
        "functions": [],
        "classes": [],
        "methods": [],          
        "routes": [],
        "modules": [],
        "django_patterns": {    
            "models": [],
            "serializers": [],
            "views": [],
            "viewsets": [],
            "forms": [],
            "admin": [],
            "urls": [],
            "middleware": [],
        },
        "fastapi_routes": [],
        "properties": [],       
        "async_functions": [],  
        "files_analyzed": [],   
        "imports": [],
        "nested_functions": [],
        "project_structure": {  # NEW: Track project structure for universal handling
            "root": str(root),
            "package_names": set(),
            "module_paths": {}
        }
    }
    
    # Initialize framework manager
    framework_manager = FrameworkManager()
    
    for f in files:
        rel_path = str(f.relative_to(root))
        out["files_analyzed"].append(rel_path)
        
        try:
            code = read_text(f)
            tree = ast.parse(code)
        except Exception as e:
            print(f"Warning: Failed to parse {rel_path}: {e}")
            continue
        
        # Track project structure
        dir_parts = pathlib.Path(rel_path).parts
        if len(dir_parts) > 0:
            # Track potential package names
            first_dir = dir_parts[0]
            if first_dir and not first_dir.startswith('.'):
                out["project_structure"]["package_names"].add(first_dir)
            
            # Track module paths
            module_path = rel_path.replace('/', '.').replace('.py', '')
            out["project_structure"]["module_paths"][module_path] = rel_path
        
        # Track ALL top-level vs nested elements
        top_level_names = {
            node.name for node in tree.body 
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        
        # Use framework handlers for framework-specific analysis
        for handler in framework_manager.get_all_handlers():
            framework_specifics = handler.analyze_framework_specifics(tree, rel_path)
            for key, value in framework_specifics.items():
                if key in out and isinstance(out[key], list):
                    out[key].extend(value)
                elif key in out and isinstance(out[key], dict):
                    for subkey, subvalue in value.items():
                        if subkey in out[key] and isinstance(out[key][subkey], list):
                            out[key][subkey].extend(subvalue)
        
        # Walk AST for ALL nodes
        for node in ast.walk(tree):
            # Functions and async functions (including nested)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_top_level = node.name in top_level_names
                
                func_rec = {
                    "name": node.name,
                    "file": rel_path,
                    "lineno": getattr(node, "lineno", 1),
                    "end_lineno": getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "is_top_level": is_top_level,
                    "is_nested": not is_top_level,
                    "args_count": len(node.args.args) if hasattr(node, 'args') else 0,
                    "has_decorators": bool(getattr(node, "decorator_list", [])),
                }
                
                # Add to appropriate collections
                if is_top_level:
                    out["functions"].append(func_rec)
                    
                    if isinstance(node, ast.AsyncFunctionDef):
                        out["async_functions"].append(func_rec.copy())
                else:
                    out["nested_functions"].append(func_rec)
                
                # Check for route decorators
                for d in getattr(node, "decorator_list", []):
                    route_info = _extract_route_info(d)
                    if route_info and route_info.get("path"):
                        out["routes"].append({
                            "handler": node.name,
                            "file": rel_path,
                            "method": route_info.get("method"),
                            "path": route_info.get("path"),
                            "lineno": func_rec["lineno"],
                            "end_lineno": func_rec["end_lineno"],
                            "framework": "fastapi" if route_info.get("method") else "flask",
                        })
                    
                    # Check for property decorator
                    if isinstance(d, ast.Name) and d.id == "property":
                        out["properties"].append(func_rec.copy())
            
            # Classes (including nested)
            elif isinstance(node, ast.ClassDef):
                is_top_level = node.name in top_level_names
                
                class_rec = {
                    "name": node.name,
                    "file": rel_path,
                    "lineno": getattr(node, "lineno", 1),
                    "end_lineno": getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                    "is_top_level": is_top_level,
                    "bases": [
                        b.id if isinstance(b, ast.Name) else 
                        getattr(b, 'attr', str(b)) 
                        for b in node.bases
                    ],
                    "method_count": len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]),
                }
                
                if is_top_level:
                    out["classes"].append(class_rec)
                
                # Extract ALL methods from the class (including nested)
                methods = _analyze_class_methods(node, rel_path)
                out["methods"].extend(methods)
            
            # Import statements - track ALL imports
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                import_info = {
                    "file": rel_path,
                    "lineno": getattr(node, "lineno", 1),
                    "type": "import" if isinstance(node, ast.Import) else "import_from",
                }
                
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                    out["modules"].extend(names)
                    import_info["modules"] = names
                elif node.module:
                    out["modules"].append(node.module)
                    import_info["module"] = node.module
                    import_info["names"] = [a.name for a in node.names]
                
                out["imports"].append(import_info)
    
    # Deduplicate and sort modules
    out["modules"] = sorted(set(out["modules"]))
    
    # Convert set to list for JSON serialization
    out["project_structure"]["package_names"] = list(out["project_structure"]["package_names"])
    
    # Detect framework and add framework info
    framework = framework_manager.detect_framework(out)
    framework_info = framework_manager.get_framework_analysis(out)
    out["framework_info"] = framework_info
    
    # Print comprehensive summary
    print(f"UNIVERSAL ANALYSIS COMPLETE:")
    print(f"   Files analyzed: {len(out['files_analyzed'])}")
    print(f"   Functions: {len(out['functions'])} (top-level)")
    print(f"   Nested Functions: {len(out['nested_functions'])}")
    print(f"   Classes: {len(out['classes'])}")
    print(f"   Methods: {len(out['methods'])}")
    print(f"   Routes: {len(out['routes'])}")
    print(f"   FastAPI Routes: {len(out['fastapi_routes'])}")
    print(f"   Properties: {len(out['properties'])}")
    print(f"   Async functions: {len(out['async_functions'])}")
    print(f"   Imports tracked: {len(out['imports'])}")
    print(f"   Django models: {len(out['django_patterns']['models'])}")
    print(f"   Serializers: {len(out['django_patterns']['serializers'])}")
    print(f"   Views/ViewSets: {len(out['django_patterns']['views']) + len(out['django_patterns']['viewsets'])}")
    print(f"   Forms: {len(out['django_patterns']['forms'])}")
    print(f"   Admin: {len(out['django_patterns']['admin'])}")
    print(f"   Project packages: {len(out['project_structure']['package_names'])}")
    print(f"   Detected Framework: {framework}")
    
    return out

def main():
    ap = argparse.ArgumentParser(
        description="UNIVERSAL Python code analyzer for any project structure"
    )
    ap.add_argument(
        "--root", 
        default=os.getenv("ANALYZE_ROOT", "."), 
        help="root directory to analyze (default: current directory)"
    )
    ap.add_argument(
        "--output",
        help="output JSON file (default: print to stdout)"
    )
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="only print summary statistics"
    )
    
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    
    if not root.exists():
        print(f"Error: Directory does not exist: {root}")
        return 1
    
    analysis = analyze_python_tree(root)
    
    if args.summary_only:
        return 0
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to: {args.output}")
    else:
        print(json.dumps(analysis, indent=2))
    
    return 0

if __name__ == "__main__":
    exit(main())