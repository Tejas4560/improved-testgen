"""
AST Context Extractor

Extracts relevant source code context based on test imports.
"""

import ast
import os
import re
from typing import Dict, List, Set, Optional
from pathlib import Path


class ASTContextExtractor:
    """
    Extracts relevant source code based on test file imports.

    Analyzes test imports and extracts:
    - Classes
    - Functions
    - Routes
    - Models
    - Utils

    Advanced features:
    - Targeted extraction (finds functions at any line number)
    - Recursive dependency resolution
    - Error traceback parsing
    - Smart caching for performance
    """

    def __init__(self, project_root: str = ".", verbose: bool = False):
        self.project_root = Path(project_root)
        self.verbose = verbose
        # Max lines to extract from a single source file (prevent token overflow)
        # REDUCED from 300 to 200 to prevent token overflow with error messages
        self.max_source_lines = 200
        # Cache for source maps (performance optimization)
        self._source_map_cache = {}

    def extract_context(
        self,
        test_file_path: str,
        test_function_name: str,
        error_message: str = ""
    ) -> Dict[str, str]:
        """
        Extract relevant source code context for a failing test.

        Args:
            test_file_path: Path to the test file
            test_function_name: Name of the failing test function
            error_message: Error message with traceback (for targeted extraction)

        Returns:
            Dictionary mapping source file paths to their relevant code
        """
        # Read test file
        try:
            with open(test_file_path, 'r') as f:
                test_content = f.read()
        except FileNotFoundError:
            return {}

        # Parse test file AST
        try:
            tree = ast.parse(test_content)
        except SyntaxError:
            return {}

        # Extract imports from test file
        imports = self._extract_imports(tree)

        # Extract the specific test function code
        test_func_code = self._extract_test_function(tree, test_function_name)

        # Analyze imports used in the test function
        test_imports = self._get_function_imports(test_func_code, imports)

        # NEW: Extract HTTP endpoints from test code (for e2e/integration tests)
        http_endpoints = self._extract_http_endpoints(test_func_code)
        if http_endpoints and self.verbose:
            print(f"HTTP endpoints detected: {http_endpoints[:3]}")

        # Resolve import paths to actual files
        source_files = self._resolve_imports_to_files(test_imports)

        # FALLBACK: If HTTP endpoints detected but no source files found,
        # search common locations for files that might contain the endpoints
        if http_endpoints and not source_files:
            if self.verbose:
                print(f"No source files from imports, searching for HTTP endpoint handlers...")
            source_files = self._find_files_with_http_endpoints(http_endpoints)
            if source_files and self.verbose:
                print(f"Found {len(source_files)} file(s) with matching endpoints")

        # Extract relevant code from each source file
        context = {}
        for source_file in source_files:
            # Use targeted extraction if error message provided, otherwise fallback
            if error_message:
                code = self._extract_relevant_code_targeted(
                    source_file=source_file,
                    test_file=test_file_path,
                    error_message=error_message,
                    max_lines=self.max_source_lines,
                    http_endpoints=http_endpoints  # Pass HTTP endpoints for e2e tests
                )
            else:
                code = self._extract_relevant_code(source_file, test_imports)

            if code:
                context[source_file] = code

        if self.verbose:
            if context:
                print(f"Extracted context from {len(context)} source file(s)")
            else:
                print(f"No source code context found")
                print(f"Imports detected: {list(imports.keys())[:5]}")
                print(f"Used in test: {list(test_imports)[:5]}")

        return context

    def _extract_imports(self, tree: ast.AST) -> Dict[str, str]:
        """
        Extract all imports from the AST, including string-based references in patch/monkeypatch.

        Args:
            tree: AST tree

        Returns:
            Dictionary mapping import names to module paths
        """
        imports = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # import app.main as app_main
                    name = alias.asname or alias.name
                    imports[name] = alias.name

                    # Also add the base module path for matching
                    # e.g., "app.main" should match both "app_main" and "app.main"
                    if alias.asname and '.' in alias.name:
                        # Add intermediate paths for multi-part imports
                        parts = alias.name.split('.')
                        for i in range(len(parts)):
                            partial = '.'.join(parts[:i+1])
                            imports[partial] = partial

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    name = alias.asname or alias.name
                    full_path = f"{module}.{alias.name}" if module else alias.name
                    imports[name] = full_path

                    # Also add the module itself
                    if module:
                        imports[module] = module

            # NEW: Detect string-based imports in patch() and monkeypatch calls
            elif isinstance(node, ast.Call):
                # Check for unittest.mock.patch('app.main.model')
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'patch':
                    if node.args and isinstance(node.args[0], (ast.Constant, ast.Str)):
                        # Extract the string argument
                        patch_target = node.args[0].value if isinstance(node.args[0], ast.Constant) else node.args[0].s
                        if isinstance(patch_target, str) and '.' in patch_target:
                            # 'app.main.model' -> extract 'app.main'
                            parts = patch_target.split('.')
                            # Add the module path (everything except the last part)
                            if len(parts) >= 2:
                                module = '.'.join(parts[:-1])
                                imports[module] = module
                                if self.verbose:
                                    print(f"Detected patch target: '{patch_target}' - importing '{module}'")

                # Check for monkeypatch.setattr('app.main.model', ...)
                elif isinstance(node.func, ast.Attribute) and node.func.attr == 'setattr':
                    if node.args and isinstance(node.args[0], (ast.Constant, ast.Str)):
                        # Extract the string argument
                        setattr_target = node.args[0].value if isinstance(node.args[0], ast.Constant) else node.args[0].s
                        if isinstance(setattr_target, str) and '.' in setattr_target:
                            # 'app.main.model' -> extract 'app.main'
                            parts = setattr_target.split('.')
                            if len(parts) >= 2:
                                module = '.'.join(parts[:-1])
                                imports[module] = module
                                if self.verbose:
                                    print(f"Detected monkeypatch target: '{setattr_target}' - importing '{module}'")

                # Check for dynamic import helpers: pytest.importorskip("app.main"), safe_import("app.main"), try_import("app.main")
                elif isinstance(node.func, ast.Attribute):
                    # pytest.importorskip("app.main")
                    if node.func.attr == 'importorskip':
                        if node.args and isinstance(node.args[0], (ast.Constant, ast.Str)):
                            module_path = node.args[0].value if isinstance(node.args[0], ast.Constant) else node.args[0].s
                            if isinstance(module_path, str):
                                imports[module_path] = module_path
                                if self.verbose:
                                    print(f"Detected pytest.importorskip('{module_path}') - importing '{module_path}'")

                # Check for function-based dynamic imports: safe_import("app.main"), try_import("app.main")
                elif isinstance(node.func, ast.Name):
                    if node.func.id in ('safe_import', 'try_import', 'importorskip'):
                        if node.args and isinstance(node.args[0], (ast.Constant, ast.Str)):
                            module_path = node.args[0].value if isinstance(node.args[0], ast.Constant) else node.args[0].s
                            if isinstance(module_path, str):
                                imports[module_path] = module_path
                                if self.verbose:
                                    print(f"Detected {node.func.id}('{module_path}') - importing '{module_path}'")

            # Also check for patch() used as decorator
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        # @patch('app.main.model')
                        if isinstance(decorator.func, ast.Name) and decorator.func.id == 'patch':
                            if decorator.args and isinstance(decorator.args[0], (ast.Constant, ast.Str)):
                                patch_target = decorator.args[0].value if isinstance(decorator.args[0], ast.Constant) else decorator.args[0].s
                                if isinstance(patch_target, str) and '.' in patch_target:
                                    parts = patch_target.split('.')
                                    if len(parts) >= 2:
                                        module = '.'.join(parts[:-1])
                                        imports[module] = module
                                        if self.verbose:
                                            print(f"Detected patch decorator: '@patch({patch_target})' - importing '{module}'")

        return imports

    def _extract_test_function(self, tree: ast.AST, func_name: str) -> str:
        """
        Extract the source code of a specific test function.

        Args:
            tree: AST tree
            func_name: Function name to extract (may include parameters like "test_foo[param]")

        Returns:
            Source code of the function
        """
        # Strip parameter suffix for parameterized tests
        # e.g., "test_foo[param]" → "test_foo"
        base_func_name = func_name.split('[')[0] if '[' in func_name else func_name

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == base_func_name:
                return ast.unparse(node)

        return ""

    def _get_function_imports(
        self,
        function_code: str,
        all_imports: Dict[str, str]
    ) -> Set[str]:
        """
        Identify which imports are used in a specific function.

        Args:
            function_code: Source code of the function
            all_imports: All available imports

        Returns:
            Set of module paths used in the function
        """
        used_imports = set()

        for name, module_path in all_imports.items():
            # Check if the import name is used in the function code
            if name in function_code:
                used_imports.add(module_path)

        return used_imports

    def _extract_http_endpoints(self, test_code: str) -> List[tuple[str, str]]:
        """
        Extract HTTP endpoints from test code (for e2e/integration tests).

        Looks for patterns like:
        - client.get("/health")
        - client.post("/predict")
        - response = await client.get("/model/info")

        Args:
            test_code: Test function code

        Returns:
            List of (method, endpoint) tuples, e.g., [("GET", "/health"), ("POST", "/predict")]
        """
        endpoints = []

        # Patterns for HTTP method calls
        # Match: client.get("/path"), client.post("/path", ...), etc.
        http_methods = ['get', 'post', 'put', 'delete', 'patch', 'head', 'options']

        for method in http_methods:
            # Pattern: client.method("/endpoint")
            pattern = rf'client\.{method}\s*\(\s*["\']([^"\']+)["\']'
            matches = re.findall(pattern, test_code, re.IGNORECASE)

            for endpoint in matches:
                endpoints.append((method.upper(), endpoint))

        return endpoints

    def _map_endpoints_to_handlers(
        self,
        http_endpoints: List[tuple[str, str]],
        source_file: str,
        source_map: Dict[str, Dict]
    ) -> Set[str]:
        """
        Map HTTP endpoints to their FastAPI handler functions.

        Looks for decorators like:
        - @app.get("/health")
        - @app.post("/predict")
        - @router.get("/model/info")

        Args:
            http_endpoints: List of (method, endpoint) tuples
            source_file: Path to source file
            source_map: Source map with function definitions

        Returns:
            Set of handler function names
        """
        handlers = set()

        try:
            with open(source_file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
        except:
            return handlers

        # Build endpoint -> function mapping
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            # Check decorators for route definitions
            for decorator in node.decorator_list:
                route_info = self._parse_route_decorator(decorator)
                if not route_info:
                    continue

                method, endpoint = route_info

                # Match against HTTP endpoints from test
                for test_method, test_endpoint in http_endpoints:
                    if method == test_method and endpoint == test_endpoint:
                        handlers.add(node.name)
                        if self.verbose:
                            print(f"{method} {endpoint} - {node.name}()")

        return handlers

    def _parse_route_decorator(self, decorator: ast.expr) -> Optional[tuple[str, str]]:
        """
        Parse a FastAPI route decorator to extract method and endpoint.

        Examples:
        - @app.get("/health") → ("GET", "/health")
        - @router.post("/predict") → ("POST", "/predict")

        Args:
            decorator: Decorator AST node

        Returns:
            (method, endpoint) tuple or None
        """
        # Pattern: @app.method("/endpoint") or @router.method("/endpoint")
        if not isinstance(decorator, ast.Call):
            return None

        if not isinstance(decorator.func, ast.Attribute):
            return None

        # Get the method name (get, post, put, delete, etc.)
        method_name = decorator.func.attr
        http_methods = ['get', 'post', 'put', 'delete', 'patch', 'head', 'options']

        if method_name not in http_methods:
            return None

        # Get the endpoint path (first argument)
        if not decorator.args:
            return None

        first_arg = decorator.args[0]
        if isinstance(first_arg, ast.Constant):
            endpoint = first_arg.value
        elif isinstance(first_arg, ast.Str):  # Python 3.7 compatibility
            endpoint = first_arg.s
        else:
            return None

        return (method_name.upper(), endpoint)

    def _extract_decorator_dependencies(
        self,
        func_node: ast.AST,
        source_map: Dict[str, Dict]
    ) -> Set[str]:
        """
        Extract dependency functions from route decorators.

        Parses decorators like:
        - @app.get("/path", dependencies=[Depends(verify_api_key)])
        - @app.post("/path", dependencies=[Depends(auth_user), Depends(rate_limit)])

        Args:
            func_node: Function AST node
            source_map: Map of all source definitions

        Returns:
            Set of dependency function names
        """
        dependencies = set()

        if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return dependencies

        # Check each decorator
        for decorator in func_node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue

            # Look for 'dependencies' keyword argument
            # @app.get("/path", dependencies=[...])
            for keyword in decorator.keywords:
                if keyword.arg == 'dependencies':
                    # Parse the list of dependencies
                    self._parse_dependency_list(keyword.value, dependencies, source_map)

        return dependencies

    def _parse_dependency_list(
        self,
        node: ast.AST,
        dependencies: Set[str],
        source_map: Dict[str, Dict]
    ):
        """
        Recursively parse dependency list to extract function names.

        Args:
            node: AST node (could be List, Call, Name, etc.)
            dependencies: Set to add found dependencies to
            source_map: Map of all source definitions
        """
        if isinstance(node, ast.List):
            # dependencies=[Depends(func1), Depends(func2)]
            for element in node.elts:
                self._parse_dependency_list(element, dependencies, source_map)

        elif isinstance(node, ast.Call):
            # Depends(verify_api_key) or Depends(verify_api_key())
            if isinstance(node.func, ast.Name) and node.func.id == 'Depends':
                # First argument is the dependency function
                if node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Name):
                        # Depends(verify_api_key)
                        func_name = arg.id
                        if func_name in source_map:
                            dependencies.add(func_name)
                    elif isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                        # Depends(verify_api_key())
                        func_name = arg.func.id
                        if func_name in source_map:
                            dependencies.add(func_name)

        elif isinstance(node, ast.IfExp):
            # dependencies=[Depends(func)] if condition else []
            self._parse_dependency_list(node.body, dependencies, source_map)
            self._parse_dependency_list(node.orelse, dependencies, source_map)

        elif isinstance(node, ast.Name):
            # dependencies=AUTH_DEPS (variable reference!)
            var_name = node.id
            # Look up variable in source map and parse its value
            if var_name in source_map:
                var_node = source_map[var_name]['node']
                # If it's an assignment, get the assigned value
                if isinstance(var_node, ast.Assign):
                    # Parse the value (could be list, ternary, etc.)
                    # This will recursively find Depends(verify_api_key) etc.
                    self._parse_dependency_list(var_node.value, dependencies, source_map)

    def _find_files_with_http_endpoints(self, http_endpoints: List[tuple[str, str]]) -> List[str]:
        """
        Search project for files containing the specified HTTP endpoints.

        This is a fallback for when imports don't resolve to source files,
        but we know the test makes HTTP requests that must be defined somewhere.

        Args:
            http_endpoints: List of (method, endpoint) tuples

        Returns:
            List of file paths containing matching endpoints
        """
        matching_files = []

        # Common locations for FastAPI apps
        search_paths = [
            'app/main.py',
            'app/__init__.py',
            'main.py',
            'src/main.py',
            'src/app/main.py',
            'api/main.py',
            'server.py',
            'app.py',
        ]

        # Also search for route files
        search_patterns = [
            'app/routes*.py',
            'app/api*.py',
            'routes/*.py',
            'api/*.py',
        ]

        # Collect candidate files
        candidate_files = []

        # Add direct paths that exist
        for path in search_paths:
            full_path = os.path.join(self.project_root, path)
            if os.path.exists(full_path):
                candidate_files.append(full_path)

        # Add files matching patterns
        import glob
        for pattern in search_patterns:
            full_pattern = os.path.join(self.project_root, pattern)
            candidate_files.extend(glob.glob(full_pattern))

        # Check each candidate for matching endpoints
        for file_path in candidate_files:
            if self._file_contains_endpoints(file_path, http_endpoints):
                matching_files.append(file_path)

        return matching_files

    def _file_contains_endpoints(self, file_path: str, http_endpoints: List[tuple[str, str]]) -> bool:
        """
        Check if a file contains any of the specified HTTP endpoints.

        Args:
            file_path: Path to Python file
            http_endpoints: List of (method, endpoint) tuples

        Returns:
            True if file contains at least one matching endpoint
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
        except:
            return False

        # Extract all route decorators from this file
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            for decorator in node.decorator_list:
                route_info = self._parse_route_decorator(decorator)
                if not route_info:
                    continue

                method, endpoint = route_info

                # Check if this matches any of our target endpoints
                for target_method, target_endpoint in http_endpoints:
                    if method == target_method and endpoint == target_endpoint:
                        return True

        return False

    def _resolve_imports_to_files(self, imports: Set[str]) -> List[str]:
        """
        Resolve import module paths to actual file paths.

        Args:
            imports: Set of module paths

        Returns:
            List of resolved file paths
        """
        files = []

        for import_path in imports:
            # Skip standard library and third-party imports
            if self._is_stdlib_or_third_party(import_path):
                continue

            # Convert module path to file path
            file_path = self._module_to_file(import_path)

            if file_path and os.path.exists(file_path):
                files.append(file_path)

        return files

    def _is_stdlib_or_third_party(self, module_path: str) -> bool:
        """
        Check if a module is from stdlib or third-party.

        Args:
            module_path: Module path (e.g., "os.path" or "django.db")

        Returns:
            True if stdlib or third-party
        """
        # Common stdlib modules
        stdlib_modules = {
            'os', 'sys', 'json', 'ast', 're', 'typing', 'pathlib',
            'collections', 'itertools', 'functools', 'datetime',
            'unittest', 'pytest', 'asyncio'
        }

        # Common third-party frameworks
        third_party_prefixes = [
            'django', 'flask', 'fastapi', 'pydantic', 'sqlalchemy',
            'requests', 'httpx', 'pytest', 'unittest', 'mock'
        ]

        # Check top-level module
        top_level = module_path.split('.')[0]

        if top_level in stdlib_modules:
            return True

        for prefix in third_party_prefixes:
            if module_path.startswith(prefix):
                return True

        return False

    def _module_to_file(self, module_path: str) -> Optional[str]:
        """
        Convert module path to file path.

        Args:
            module_path: Module path (e.g., "src.models.user")

        Returns:
            File path or None
        """
        parts = module_path.split('.')

        # Try different variations
        variations = []

        # 1. Direct path: src.models.user -> src/models/user.py
        variations.append(module_path.replace('.', '/') + '.py')

        # 2. Package path: src.models.user -> src/models/user/__init__.py
        variations.append(module_path.replace('.', '/') + '/__init__.py')

        # 3. Without first component: models.user -> models/user.py
        if len(parts) > 1:
            variations.append('/'.join(parts[1:]) + '.py')

        # 4. Single file at root: app.main -> main.py or app.py
        if len(parts) >= 2:
            variations.append(parts[-1] + '.py')  # Last component
            variations.append(parts[0] + '.py')   # First component
        elif len(parts) == 1:
            variations.append(parts[0] + '.py')

        # 5. Common Python app structures
        if len(parts) >= 2:
            # app.something -> app/something.py
            variations.append(f"{parts[0]}/{'/'.join(parts[1:])}.py")
            # something.else -> src/something/else.py
            variations.append(f"src/{module_path.replace('.', '/')}.py")

        # 6. Single file patterns for common entry points
        common_files = ['main.py', 'app.py', 'server.py', 'api.py', '__init__.py']
        variations.extend(common_files)
        variations.extend([f"app/{f}" for f in common_files])
        variations.extend([f"src/{f}" for f in common_files])

        if self.verbose:
            print(f"Trying to resolve module '{module_path}'...")

        # Try each variation
        for var in variations:
            full_path = self.project_root / var
            if full_path.exists():
                if self.verbose:
                    print(f"Found: {full_path}")
                return str(full_path)

        if self.verbose:
            print(f"Not found (tried {len(variations)} variations)")

        return None

    def _extract_relevant_code(
        self,
        source_file: str,
        imports: Set[str]
    ) -> str:
        """
        Extract relevant code elements from a source file.

        Intelligently limits extraction to avoid token overflow.

        Args:
            source_file: Path to source file
            imports: Import paths that reference this file

        Returns:
            Concatenated relevant code (limited to max_source_lines)
        """
        try:
            with open(source_file, 'r') as f:
                content = f.read()
        except (FileNotFoundError, IOError):
            return ""

        lines = content.split('\n')

        # If file is small enough, return all content
        if len(lines) <= self.max_source_lines:
            return content

        # File too large - extract intelligently
        if self.verbose:
            print(f"File too large ({len(lines)} lines), extracting relevant parts only...")

        try:
            tree = ast.parse(content)
        except SyntaxError:
            # If parsing fails, return truncated raw content
            truncated = '\n'.join(lines[:self.max_source_lines])
            return truncated + f"\n\n# ... (file truncated: {len(lines)} total lines)"

        # Extract definitions with size tracking
        extracted_items = []
        current_lines = 0

        # Priority 1: Extract imports and constants (usually at top)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign)):
                code = ast.unparse(node)
                item_lines = len(code.split('\n'))
                if current_lines + item_lines <= self.max_source_lines:
                    extracted_items.append(code)
                    current_lines += item_lines

        # Priority 2: Extract functions and classes (up to limit)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                code = ast.unparse(node)
                item_lines = len(code.split('\n'))

                # If adding this would exceed limit, skip
                if current_lines + item_lines > self.max_source_lines:
                    continue

                extracted_items.append(code)
                current_lines += item_lines

        result = "\n\n".join(extracted_items) if extracted_items else ""

        # Add truncation notice
        if current_lines < len(lines):
            result += f"\n\n# ... (extracted {current_lines}/{len(lines)} lines to fit token limit)"

            if self.verbose:
                print(f"Extracted {current_lines}/{len(lines)} lines")

        return result if result else content[:self.max_source_lines * 80]  # Fallback

    # ========================================================================
    # ADVANCED TARGETED EXTRACTION COMPONENTS
    # ========================================================================

    def _parse_test_imports_detailed(self, test_file: str) -> Dict[str, Set[str]]:
        """
        Parse test file to find what it imports from each module (detailed version).

        Args:
            test_file: Path to test file

        Returns:
            Dict mapping module paths to imported names
            Example: {
                'app.main': {'predict_batch', 'validate_sentence'},
                'app.utils': {'sanitize_input'}
            }
        """
        try:
            with open(test_file, 'r') as f:
                tree = ast.parse(f.read())
        except (FileNotFoundError, SyntaxError):
            return {}

        imports = {}  # module_path -> set of imported names

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # import app.main as app_main
                for alias in node.names:
                    module_path = alias.name  # 'app.main'
                    import_name = alias.asname or alias.name  # 'app_main'

                    if module_path not in imports:
                        imports[module_path] = set()
                    imports[module_path].add(import_name)

            elif isinstance(node, ast.ImportFrom):
                # from app.main import predict_batch, validate_sentence
                module_path = node.module or ""

                if module_path not in imports:
                    imports[module_path] = set()

                for alias in node.names:
                    import_name = alias.name  # 'predict_batch'
                    imports[module_path].add(import_name)

            elif isinstance(node, ast.Call):
                # Check for dynamic import helpers: pytest.importorskip("app.main"), safe_import("app.main"), try_import("app.main")
                module_path = None

                if isinstance(node.func, ast.Attribute):
                    # pytest.importorskip("app.main")
                    if node.func.attr == 'importorskip':
                        if node.args and isinstance(node.args[0], (ast.Constant, ast.Str)):
                            module_path = node.args[0].value if isinstance(node.args[0], ast.Constant) else node.args[0].s

                elif isinstance(node.func, ast.Name):
                    # safe_import("app.main"), try_import("app.main")
                    if node.func.id in ('safe_import', 'try_import', 'importorskip'):
                        if node.args and isinstance(node.args[0], (ast.Constant, ast.Str)):
                            module_path = node.args[0].value if isinstance(node.args[0], ast.Constant) else node.args[0].s

                if module_path and isinstance(module_path, str):
                    if module_path not in imports:
                        imports[module_path] = set()
                    # For dynamic imports, we import the whole module
                    imports[module_path].add('*')
                    if self.verbose:
                        print(f"Detected dynamic import: '{module_path}'")

        if self.verbose and imports:
            print(f"Parsed test imports:")
            # Separate dynamic imports (with '*') from regular imports
            dynamic_imports = {k: v for k, v in imports.items() if '*' in v}
            regular_imports = {k: v for k, v in imports.items() if '*' not in v}

            # Show dynamic imports first
            for module, names in list(dynamic_imports.items())[:3]:
                print(f"{module}: <module> (dynamic import)")

            # Then show regular imports
            for module, names in list(regular_imports.items())[:3]:
                names_str = ', '.join(list(names)[:5])
                if len(names) > 5:
                    names_str += f', ... ({len(names)} total)'
                print(f"{module}: {names_str}")

        return imports

    def _build_source_map(self, source_file: str) -> Dict[str, Dict]:
        """
        Build an index of all definitions in the source file.

        Args:
            source_file: Path to source file

        Returns:
            Dict mapping names to definition info
            Format: {
                'function_name': {
                    'node': ast.FunctionDef,
                    'line_start': int,
                    'line_end': int,
                    'code': str
                }
            }
        """
        # Check cache first
        if source_file in self._source_map_cache:
            return self._source_map_cache[source_file]

        try:
            with open(source_file, 'r') as f:
                content = f.read()
        except (FileNotFoundError, IOError):
            return {}

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {}

        source_map = {}

        # Walk through all top-level definitions
        for node in tree.body:
            name = None

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Function definitions
                name = node.name

            elif isinstance(node, ast.ClassDef):
                # Class definitions
                name = node.name

            elif isinstance(node, ast.Assign):
                # Variable assignments (constants)
                # MODEL = None
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        break

            if name:
                # Store definition info
                try:
                    code = ast.unparse(node)
                except:
                    code = ""  # Fallback if unparsing fails

                source_map[name] = {
                    'node': node,
                    'line_start': node.lineno if hasattr(node, 'lineno') else 0,
                    'line_end': node.end_lineno if hasattr(node, 'end_lineno') else 0,
                    'code': code
                }

        # Cache the result
        self._source_map_cache[source_file] = source_map

        if self.verbose and source_map:
            print(f"Built source map: {len(source_map)} definitions found")

        return source_map

    def _parse_error_traceback(
        self,
        error_message: str,
        source_file: str
    ) -> Set[str]:
        """
        Extract function names from error traceback.

        Args:
            error_message: The full error message with traceback
            source_file: Path to source file (to filter relevant entries)

        Returns:
            Set of function names that appear in the traceback
        """
        functions = set()

        if not error_message:
            return functions

        # Normalize paths for comparison
        try:
            source_file_normalized = os.path.abspath(source_file)
            source_file_name = os.path.basename(source_file)
        except:
            return functions

        # Pattern: File "path/to/file.py", line 123, in function_name
        # Matches both:
        #   File "/home/user/app/main.py", line 520, in predict_batch
        #   File "app/main.py", line 520, in predict_batch
        pattern = r'File\s+"([^"]+)",\s+line\s+(\d+),\s+in\s+(\w+)'

        for match in re.finditer(pattern, error_message):
            file_path = match.group(1)
            line_number = int(match.group(2))
            function_name = match.group(3)

            # Check if this traceback entry is from our source file
            # Match by filename or full path
            try:
                file_path_normalized = os.path.abspath(file_path)
                file_name = os.path.basename(file_path)

                if (file_path_normalized == source_file_normalized or
                    file_name == source_file_name):
                    functions.add(function_name)

                    if self.verbose:
                        print(f"Found in traceback: {function_name} (line {line_number})")
            except:
                # If path normalization fails, try basic string matching
                if source_file_name in file_path:
                    functions.add(function_name)

        return functions

    def _find_dependencies(
        self,
        node: ast.AST,
        source_map: Dict[str, Dict],
        max_depth: int = 3,
        visited: Optional[Set[str]] = None
    ) -> Set[str]:
        """
        Find all functions/variables that a node depends on (recursive).

        Args:
            node: AST node to analyze
            source_map: Map of all available definitions
            max_depth: Maximum recursion depth (prevent infinite loops)
            visited: Set of already visited names (for cycle detection)

        Returns:
            Set of dependency names
        """
        if visited is None:
            visited = set()

        if max_depth <= 0:
            return set()

        dependencies = set()

        # Walk through the function/class body
        for child in ast.walk(node):
            # Find Name nodes (variable/function references)
            if isinstance(child, ast.Name):
                name = child.id

                # Check if this name is defined in our source map
                if name in source_map and name not in visited:
                    dependencies.add(name)
                    visited.add(name)

                    # Recursively find dependencies of this dependency
                    dep_node = source_map[name]['node']
                    try:
                        sub_deps = self._find_dependencies(
                            dep_node,
                            source_map,
                            max_depth - 1,
                            visited
                        )
                        dependencies.update(sub_deps)
                    except RecursionError:
                        # Safety net for deep recursion
                        pass

            # Find function calls
            elif isinstance(child, ast.Call):
                # Direct function call: predict(text)
                if isinstance(child.func, ast.Name):
                    name = child.func.id

                    if name in source_map and name not in visited:
                        dependencies.add(name)
                        visited.add(name)

                        # Recursively find dependencies
                        dep_node = source_map[name]['node']
                        try:
                            sub_deps = self._find_dependencies(
                                dep_node,
                                source_map,
                                max_depth - 1,
                                visited
                            )
                            dependencies.update(sub_deps)
                        except RecursionError:
                            pass

                # Attribute call: obj.method()
                elif isinstance(child.func, ast.Attribute):
                    # MODEL.predict() - the object is 'MODEL'
                    if isinstance(child.func.value, ast.Name):
                        obj_name = child.func.value.id

                        if obj_name in source_map and obj_name not in visited:
                            dependencies.add(obj_name)
                            visited.add(obj_name)

        return dependencies

    def _extract_relevant_code_targeted(
        self,
        source_file: str,
        test_file: str,
        error_message: str,
        max_lines: int = 300,
        http_endpoints: List[tuple[str, str]] = None
    ) -> str:
        """
        Extract only the code relevant to the failing test (TARGETED VERSION).

        Algorithm:
        1. Parse test imports to find what test uses
        2. Build source map to index all definitions
        3. Parse error traceback for additional context
        4. Map HTTP endpoints to handler functions (NEW for e2e tests!)
        5. Find dependencies recursively
        6. Extract targeted code with priority ordering

        Args:
            source_file: Path to source file
            test_file: Path to test file
            error_message: Error message with traceback
            max_lines: Maximum lines to extract
            http_endpoints: List of (method, endpoint) tuples for e2e tests

        Returns:
            Extracted code string
        """
        if http_endpoints is None:
            http_endpoints = []
        try:
            with open(source_file, 'r') as f:
                content = f.read()
        except (FileNotFoundError, IOError):
            return ""

        total_lines = len(content.split('\n'))

        # Always use targeted extraction (only extract functions/classes found, not entire file)
        if self.verbose:
            print(f"Using targeted extraction for {os.path.basename(source_file)} ({total_lines} lines)...")

        # Step 1: Parse test imports
        test_imports = self._parse_test_imports_detailed(test_file)

        # Get imported names from this source file
        imported_names = set()
        source_file_name = Path(source_file).stem  # 'main' from 'app/main.py'

        for module_path, names in test_imports.items():
            # Check if this module corresponds to our source file
            # E.g., 'app.main' matches 'app/main.py'
            if source_file_name in module_path.replace('.', '/'):
                imported_names.update(names)

        # Step 2: Build source map
        source_map = self._build_source_map(source_file)

        if not source_map:
            # Could not parse source file, skip AST (embeddings will handle it)
            if self.verbose:
                print(f"Could not parse source file, skipping AST (will use embeddings only)")
            return ""  # Empty = skip this file, embeddings will handle it

        # Step 3: Parse error traceback
        error_functions = self._parse_error_traceback(error_message, source_file)

        # Step 3.5: Map HTTP endpoints to handler functions (NEW for e2e tests!)
        endpoint_handlers = set()
        decorator_dependencies = set()
        if http_endpoints:
            endpoint_handlers = self._map_endpoints_to_handlers(http_endpoints, source_file, source_map)
            if endpoint_handlers and self.verbose:
                print(f"Mapped endpoints to handlers: {', '.join(list(endpoint_handlers)[:3])}")

            # Extract dependencies from decorators (NEW: for API keys, auth, etc.)
            for handler_name in endpoint_handlers:
                if handler_name in source_map:
                    deps = self._extract_decorator_dependencies(source_map[handler_name]['node'], source_map)
                    decorator_dependencies.update(deps)

            if decorator_dependencies and self.verbose:
                print(f"Found decorator dependencies: {', '.join(list(decorator_dependencies)[:3])}")

        # Step 4: Combine all target names
        target_names = imported_names | error_functions | endpoint_handlers | decorator_dependencies

        # Remove wildcard '*' - it's not a real function name, just indicates "module imported"
        # If we have '*', rely on error_functions to provide the actual targets
        has_wildcard = '*' in target_names
        target_names.discard('*')

        if self.verbose and target_names:
            targets_str = ', '.join(list(target_names)[:5])
            if len(target_names) > 5:
                targets_str += f', ... ({len(target_names)} total)'
            print(f"Target functions: {targets_str}")
        elif self.verbose and has_wildcard:
            print(f"Target functions: * (will extract from error traceback)")

        if not target_names:
            # No specific targets found, skip AST and rely on embeddings
            if self.verbose:
                print(f"No specific targets found, skipping AST (will use embeddings only)")
            return ""  # Empty = skip this file, embeddings will handle it

        # Step 5: Extract with priority ordering
        extracted = []
        extracted_names = set()
        current_lines = 0

        # Priority 1: Imports (always include if space)
        for name, info in source_map.items():
            node = info['node']
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                code = info['code']
                lines = len(code.split('\n'))

                if current_lines + lines <= max_lines:
                    extracted.append(code)
                    extracted_names.add(name)
                    current_lines += lines

        # Priority 2: Constants used by target functions
        all_dependencies = set()
        for target in target_names:
            if target in source_map:
                try:
                    deps = self._find_dependencies(
                        source_map[target]['node'],
                        source_map
                    )
                    all_dependencies.update(deps)
                except:
                    pass  # Skip if dependency finding fails

        for name in all_dependencies:
            if name not in extracted_names and name in source_map:
                node = source_map[name]['node']
                if isinstance(node, ast.Assign):
                    code = source_map[name]['code']
                    lines = len(code.split('\n'))

                    if current_lines + lines <= max_lines:
                        extracted.append(code)
                        extracted_names.add(name)
                        current_lines += lines

        # Priority 3: Target functions (the ones actually used)
        for target in target_names:
            if target not in extracted_names and target in source_map:
                code = source_map[target]['code']
                lines = len(code.split('\n'))

                if current_lines + lines <= max_lines:
                    extracted.append(code)
                    extracted_names.add(target)
                    current_lines += lines

                    if self.verbose:
                        print(f"Extracted: {target} ({lines} lines)")

        # Priority 4: Dependencies of target functions
        for dep in all_dependencies:
            if dep not in extracted_names and dep in source_map:
                code = source_map[dep]['code']
                lines = len(code.split('\n'))

                if current_lines + lines <= max_lines:
                    extracted.append(code)
                    extracted_names.add(dep)
                    current_lines += lines

                    if self.verbose:
                        print(f"Extracted: {dep} ({lines} lines, dependency)")

        # Priority 5: Fill remaining space with other definitions
        if current_lines < max_lines:
            for name, info in source_map.items():
                if name not in extracted_names:
                    node = info['node']
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        code = info['code']
                        lines = len(code.split('\n'))

                        if current_lines + lines <= max_lines:
                            extracted.append(code)
                            extracted_names.add(name)
                            current_lines += lines

        # Build result with markers (for consistent counting with embeddings)
        if not extracted:
            # Fallback if nothing extracted
            return self._extract_relevant_code(source_file, set())

        # Add markers to each extracted element for counting compatibility
        marked_extracted = []
        for name in extracted_names:
            if name in source_map:
                info = source_map[name]
                node = info['node']
                line_start = info.get('line_start', 0)

                # Determine element type
                if isinstance(node, ast.ClassDef):
                    element_type = "class"
                elif name in endpoint_handlers:
                    element_type = "http_endpoint"
                else:
                    element_type = "function"

                # Add marker
                marker = f"# {element_type}: {name} (line {line_start})"
                code = info['code']
                marked_extracted.append(f"{marker}\n{code}")

        result = "\n\n".join(marked_extracted)

        # Add metadata
        result += f"\n\n# ... (extracted {current_lines} targeted lines from {total_lines} total)"
        result += f"\n# Targeted extraction: {len(extracted_names)} definitions"

        if self.verbose:
            print(f"Extracted {current_lines}/{total_lines} lines ({len(extracted_names)} definitions)")

        return result

    def get_full_context_string(
        self,
        test_file_path: str,
        test_function_name: str,
        error_message: str = ""
    ) -> str:
        """
        Get a formatted string with all relevant context.

        Args:
            test_file_path: Path to test file
            test_function_name: Name of failing test
            error_message: Error message with traceback (for targeted extraction)

        Returns:
            Formatted context string
        """
        context = self.extract_context(test_file_path, test_function_name, error_message)

        if not context:
            return "# No relevant source code found"

        output = []
        for file_path, code in context.items():
            output.append(f"# {file_path}")
            output.append(f"```python")
            output.append(code)
            output.append(f"```")
            output.append("")

        return "\n".join(output)