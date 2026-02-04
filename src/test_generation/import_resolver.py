# src/test_generation/import_resolver.py
"""
Import resolver for handling project imports and dependencies.
"""

import importlib
import subprocess
import sys
from typing import Any, Dict, List, Set, Tuple


class ImportResolver:
    """Resolves imports and manages dependencies for test generation."""
    
    def __init__(self):
        self.resolved_imports = set()
        self.failed_imports = set()
        self.external_dependencies = set()
    
    def resolve_imports(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve all imports found in the analysis."""
        modules = analysis.get("modules", [])
        imports = analysis.get("imports", [])
        
        all_imports = set(modules)
        for imp in imports:
            if imp.get("type") == "import":
                all_imports.update(imp.get("modules", []))
            elif imp.get("type") == "import_from":
                if imp.get("module"):
                    all_imports.add(imp["module"])
        
        resolved = {
            "resolved_imports": [],
            "unresolved_imports": [],
            "external_dependencies": [],
            "local_modules": []
        }
        
        for module_name in all_imports:
            if not module_name or module_name.startswith('_'):
                continue
            
            if self._is_stdlib_module(module_name):
                resolved["resolved_imports"].append(module_name)
            elif self._is_local_module(module_name, analysis):
                resolved["local_modules"].append(module_name)
            else:
                # External dependency
                dependency = self._module_to_package(module_name)
                if dependency:
                    resolved["external_dependencies"].append(dependency)
                    self.external_dependencies.add(dependency)
        
        return resolved
    
    def _is_stdlib_module(self, module_name: str) -> bool:
        """Check if a module is part of Python standard library."""
        if hasattr(sys, "stdlib_module_names"):
            return module_name in sys.stdlib_module_names
        
        stdlib_modules = {
            "os", "sys", "re", "json", "pathlib", "math", "itertools",
            "functools", "typing", "subprocess", "datetime", "time",
            "collections", "dataclasses", "ast", "logging", "unittest",
            "argparse", "asyncio", "threading", "sqlite3", "email",
        }
        return module_name in stdlib_modules
    
    def _is_local_module(self, module_name: str, analysis: Dict[str, Any]) -> bool:
        """Check if a module is local to the project."""
        project_structure = analysis.get("project_structure", {})
        package_names = project_structure.get("package_names", [])
        module_paths = project_structure.get("module_paths", {})
        
        # Check if it's a detected package
        if module_name in package_names:
            return True
        
        # Check if it's in module paths
        if module_name in module_paths:
            return True
        
        # Check common local module patterns
        local_patterns = [
            module_name in ['app', 'main', 'application', 'server', 'api', 'backend', 'core', 'project'],
            any(module_name in module for module in module_paths),
            any(module_name == pkg for pkg in package_names),
        ]
        
        return any(local_patterns)
    
    def _module_to_package(self, module_name: str) -> str:
        """Convert module name to package name for installation."""
        common_packages = {
            "pptx": "python-pptx", "flask": "flask", "flask_bcrypt": "flask-bcrypt",
            "flask_login": "flask-login", "flask_sqlalchemy": "flask-sqlalchemy",
            "flask_wtf": "flask-wtf", "httpx": "httpx", "openai": "openai",
            "python_dotenv": "python-dotenv", "requests": "requests", "wtforms": "wtforms",
            "sqlalchemy": "SQLAlchemy", "django": "Django", "PIL": "Pillow",
            "bs4": "beautifulsoup4", "yaml": "PyYAML", "cv2": "opencv-python",
            "sklearn": "scikit-learn", "pandas": "pandas", "numpy": "numpy",
        }
        
        top_module = module_name.split(".")[0]
        return common_packages.get(top_module, top_module)
    
    def install_dependencies(self, dependencies: List[str]) -> bool:
        """Install required dependencies."""
        if not dependencies:
            print("No external dependencies to install.")
            return True
        
        print(f"Installing {len(dependencies)} external dependencies...")
        
        successful_installs = []
        failed_installs = []
        
        for package in dependencies:
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
                print(f"Successfully installed: {package}")
            except subprocess.CalledProcessError:
                failed_installs.append(package)
                print(f"Failed to install: {package}")
            except Exception as e:
                failed_installs.append(package)
                print(f"Error installing {package}: {e}")
        
        if successful_installs:
            print(f"Successfully installed {len(successful_installs)} packages.")
        
        if failed_installs:
            print(f"Failed to install {len(failed_installs)} packages.")
            return False
        
        return True