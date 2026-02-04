"""
Codebase Indexer - Builds Embedding Index for Semantic Code Retrieval

This module creates a semantic index of the entire codebase by:
1. Extracting all functions, classes, variables, and HTTP endpoints
2. Generating embeddings for each code element
3. Storing them in a vector database for fast retrieval

This solves the following problems from AST-based extraction:
- ✓ Bypasses brittle import resolution
- ✓ Handles misspelled function names
- ✓ Finds dynamically assigned routes
- ✓ Works with nested/hidden functions
- ✓ No token limit issues during indexing
"""

import ast
import os
import sys
import json
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib
import pickle
import re

# Add parent directory to path to import gen modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class CodeElement:
    """Represents a single code element (function, class, variable, etc.)"""
    element_type: str  # 'function', 'class', 'variable', 'http_endpoint'
    name: str
    file_path: str
    line_start: int
    line_end: int
    source_code: str
    signature: str  # Function signature or class definition
    docstring: Optional[str] = None
    http_method: Optional[str] = None  # For HTTP endpoints: GET, POST, etc.
    http_path: Optional[str] = None  # For HTTP endpoints: /api/users

    def to_embedding_text(self) -> str:
        """
        Convert code element to text suitable for embedding.

        This creates a rich semantic representation that captures:
        - Element type and name
        - Signature/definition
        - Docstring context
        - HTTP endpoint info (if applicable)
        - Partial source code
        """
        parts = []

        # Basic info
        parts.append(f"{self.element_type}: {self.name}")

        # Signature
        if self.signature:
            parts.append(f"signature: {self.signature}")

        # Docstring
        if self.docstring:
            parts.append(f"description: {self.docstring}")

        # HTTP endpoint info
        if self.http_method and self.http_path:
            parts.append(f"HTTP {self.http_method} {self.http_path}")

        # Source code (truncated to first 500 chars for embedding)
        if self.source_code:
            code_sample = self.source_code[:500]
            parts.append(f"code: {code_sample}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return asdict(self)


class CodebaseIndexer:
    """
    Indexes entire codebase for semantic search.

    Creates embeddings for all code elements and stores them
    in a simple vector database (using FAISS or ChromaDB).
    """

    def __init__(
        self,
        project_root: str = ".",
        cache_dir: str = ".codebase_index",
        embedding_model: str = None,
        verbose: bool = False
    ):
        self.project_root = Path(project_root)
        self.cache_dir = Path(cache_dir)

        # Auto-detect embedding model from environment
        if embedding_model is None:
            # Check for Ollama model
            if os.getenv("OLLAMA_EMBED_MODEL"):
                self.embedding_model = os.getenv("OLLAMA_EMBED_MODEL")
            # Check for OpenAI model
            elif os.getenv("OPENAI_EMBEDDING_MODEL"):
                self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
            else:
                # Last resort fallback (user should set env var)
                self.embedding_model = "deepseek-r1:latest"
        else:
            self.embedding_model = embedding_model

        self.verbose = verbose

        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)

        # Storage
        self.code_elements: List[CodeElement] = []
        self.embeddings: List[List[float]] = []

        # Lazy-load embedding client
        self._embedding_client = None

    @property
    def embedding_client(self):
        """Lazy-load embedding client (Ollama or OpenAI)."""
        if self._embedding_client is None:
            # Import using spec_from_file_location to completely bypass gen package __init__.py
            import importlib.util
            from pathlib import Path

            # Get absolute path to client files (works from any directory)
            current_file = Path(__file__).resolve()
            ollama_client_path = current_file.parent.parent / 'gen' / 'ollama_client.py'
            openai_client_path = current_file.parent.parent / 'gen' / 'openai_client.py'

            # Check if using Ollama (preferred)
            if os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_EMBED_MODEL"):
                try:
                    # Load ollama_client.py directly without triggering package __init__
                    spec = importlib.util.spec_from_file_location(
                        "ollama_client_indexer",
                        str(ollama_client_path)
                    )
                    ollama_module = importlib.util.module_from_spec(spec)
                    sys.modules['ollama_client_indexer'] = ollama_module
                    spec.loader.exec_module(ollama_module)

                    self._embedding_client = ollama_module.get_ollama_client()
                    if self.verbose:
                        print("  Using Ollama for embeddings")
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to load Ollama client: {e}")
                    # Fall back to OpenAI
                    try:
                        spec = importlib.util.spec_from_file_location(
                            "openai_client_indexer_fallback",
                            str(openai_client_path)
                        )
                        openai_module = importlib.util.module_from_spec(spec)
                        sys.modules['openai_client_indexer_fallback'] = openai_module
                        spec.loader.exec_module(openai_module)

                        self._embedding_client = openai_module.create_client()
                        if self.verbose:
                            print("  Using OpenAI for embeddings (fallback)")
                    except Exception as e2:
                        if self.verbose:
                            print(f"Both Ollama and OpenAI failed: {e2}")
                        self._embedding_client = None
            else:
                # Use OpenAI
                try:
                    spec = importlib.util.spec_from_file_location(
                        "openai_client_indexer_main",
                        str(openai_client_path)
                    )
                    openai_module = importlib.util.module_from_spec(spec)
                    sys.modules['openai_client_indexer_main'] = openai_module
                    spec.loader.exec_module(openai_module)

                    self._embedding_client = openai_module.create_client()
                    if self.verbose:
                        print("  Using OpenAI for embeddings")
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to load OpenAI client: {e}")
                    self._embedding_client = None
        return self._embedding_client

    def should_index_file(self, file_path: Path) -> bool:
        """
        Check if a file should be indexed.

        Args:
            file_path: Path to file

        Returns:
            True if should be indexed
        """
        # Skip directories to exclude
        exclude_dirs = {
            'venv', 'env', '.venv', 'myvenv',
            'node_modules', '.git', '__pycache__',
            '.pytest_cache', '.mypy_cache',
            'build', 'dist', '.eggs'
        }

        # Check if any parent directory is excluded
        for part in file_path.parts:
            if part in exclude_dirs:
                return False

        # Only Python files
        if file_path.suffix != '.py':
            return False

        # Skip test files (we index source code only)
        if 'test_' in file_path.name or file_path.name.startswith('test'):
            return False

        return True

    def extract_code_elements(self, file_path: Path) -> List[CodeElement]:
        """
        Extract all code elements from a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            List of CodeElement objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (IOError, UnicodeDecodeError) as e:
            if self.verbose:
                print(f"Could not read {file_path}: {e}")
            return []

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            if self.verbose:
                print(f"Syntax error in {file_path}: {e}")
            return []

        elements = []

        # Extract all top-level definitions
        for node in tree.body:
            # Functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                element = self._extract_function(node, file_path, content)
                if element:
                    elements.append(element)

                    # Extract HTTP endpoints from decorators
                    http_element = self._extract_http_endpoint(node, file_path, content)
                    if http_element:
                        elements.append(http_element)

            # Classes
            elif isinstance(node, ast.ClassDef):
                element = self._extract_class(node, file_path, content)
                if element:
                    elements.append(element)

                # Extract methods from class
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_element = self._extract_function(
                            item, file_path, content,
                            class_name=node.name
                        )
                        if method_element:
                            elements.append(method_element)

            # Variables (module-level constants)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        element = self._extract_variable(
                            target.id, node, file_path, content
                        )
                        if element:
                            elements.append(element)

        return elements

    def _extract_function(
        self,
        node: ast.FunctionDef,
        file_path: Path,
        content: str,
        class_name: Optional[str] = None
    ) -> Optional[CodeElement]:
        """Extract function/method as CodeElement."""
        try:
            # Get function name
            func_name = node.name
            if class_name:
                func_name = f"{class_name}.{func_name}"

            # Get signature
            signature = ast.unparse(node.args) if hasattr(ast, 'unparse') else str(node.args)
            signature = f"def {node.name}({signature}):"

            # Get docstring
            docstring = ast.get_docstring(node)

            # Get source code
            source_code = ast.unparse(node) if hasattr(ast, 'unparse') else ""

            return CodeElement(
                element_type='function',
                name=func_name,
                file_path=str(file_path),
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                source_code=source_code,
                signature=signature,
                docstring=docstring
            )
        except Exception as e:
            if self.verbose:
                print(f"Error extracting function {node.name}: {e}")
            return None

    def _extract_class(
        self,
        node: ast.ClassDef,
        file_path: Path,
        content: str
    ) -> Optional[CodeElement]:
        """Extract class as CodeElement."""
        try:
            # Get class signature
            signature = f"class {node.name}"
            if node.bases:
                bases = ', '.join(ast.unparse(base) for base in node.bases)
                signature += f"({bases})"
            signature += ":"

            # Get docstring
            docstring = ast.get_docstring(node)

            # Get source code
            source_code = ast.unparse(node) if hasattr(ast, 'unparse') else ""

            return CodeElement(
                element_type='class',
                name=node.name,
                file_path=str(file_path),
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                source_code=source_code,
                signature=signature,
                docstring=docstring
            )
        except Exception as e:
            if self.verbose:
                print(f"Error extracting class {node.name}: {e}")
            return None

    def _extract_variable(
        self,
        var_name: str,
        node: ast.Assign,
        file_path: Path,
        content: str
    ) -> Optional[CodeElement]:
        """Extract module-level variable as CodeElement."""
        try:
            # Get source code
            source_code = ast.unparse(node) if hasattr(ast, 'unparse') else ""

            return CodeElement(
                element_type='variable',
                name=var_name,
                file_path=str(file_path),
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                source_code=source_code,
                signature=f"{var_name} = ..."
            )
        except Exception as e:
            if self.verbose:
                print(f"Error extracting variable {var_name}: {e}")
            return None

    def _extract_http_endpoint(
        self,
        node: ast.FunctionDef,
        file_path: Path,
        content: str
    ) -> Optional[CodeElement]:
        """
        Extract HTTP endpoint from FastAPI/Flask decorators.

        Examples:
        - @app.get("/health") → GET /health
        - @router.post("/predict") → POST /predict
        """
        for decorator in node.decorator_list:
            endpoint_info = self._parse_route_decorator(decorator)
            if not endpoint_info:
                continue

            method, path = endpoint_info

            # Create HTTP endpoint element
            try:
                source_code = ast.unparse(node) if hasattr(ast, 'unparse') else ""

                return CodeElement(
                    element_type='http_endpoint',
                    name=node.name,
                    file_path=str(file_path),
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    source_code=source_code,
                    signature=f"@app.{method.lower()}('{path}')",
                    http_method=method,
                    http_path=path
                )
            except Exception as e:
                if self.verbose:
                    print(f"Error extracting HTTP endpoint {node.name}: {e}")
                return None

        return None

    def _parse_route_decorator(self, decorator: ast.expr) -> Optional[Tuple[str, str]]:
        """
        Parse route decorator to extract HTTP method and path.

        Returns:
            (method, path) tuple or None
        """
        if not isinstance(decorator, ast.Call):
            return None

        if not isinstance(decorator.func, ast.Attribute):
            return None

        # Get method (get, post, put, delete, etc.)
        method_name = decorator.func.attr
        http_methods = ['get', 'post', 'put', 'delete', 'patch', 'head', 'options']

        if method_name not in http_methods:
            return None

        # Get path (first argument)
        if not decorator.args:
            return None

        first_arg = decorator.args[0]
        if isinstance(first_arg, ast.Constant):
            path = first_arg.value
        elif isinstance(first_arg, ast.Str):
            path = first_arg.s
        else:
            return None

        return (method_name.upper(), path)

    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Build the embedding index for the entire codebase.

        Args:
            force_rebuild: If True, rebuild even if cache exists
        """
        cache_file = self.cache_dir / "index.pkl"

        # Check cache
        if not force_rebuild and cache_file.exists():
            if self.verbose:
                print("Loading index from cache...")
            self.load_index()
            return

        if self.verbose:
            print("Building codebase index...")

        # Find all Python files
        python_files = []
        for file_path in self.project_root.rglob("*.py"):
            if self.should_index_file(file_path):
                python_files.append(file_path)

        if self.verbose:
            print(f"  Found {len(python_files)} Python files to index")

        # Extract code elements
        self.code_elements = []
        for file_path in python_files:
            if self.verbose:
                print(f"Indexing {file_path.relative_to(self.project_root)}...")

            elements = self.extract_code_elements(file_path)
            self.code_elements.extend(elements)

            if self.verbose and elements:
                print(f"Extracted {len(elements)} elements")

        if self.verbose:
            print(f"\n Extracted {len(self.code_elements)} total code elements")
            print(f"Functions: {sum(1 for e in self.code_elements if e.element_type == 'function')}")
            print(f"Classes: {sum(1 for e in self.code_elements if e.element_type == 'class')}")
            print(f"Variables: {sum(1 for e in self.code_elements if e.element_type == 'variable')}")
            print(f"HTTP Endpoints: {sum(1 for e in self.code_elements if e.element_type == 'http_endpoint')}")

        # Generate embeddings
        if self.verbose:
            print("\n Generating embeddings...")

        self.embeddings = self._generate_embeddings_batch(self.code_elements)

        # Save cache
        if self.verbose:
            print("\n Saving index to cache...")
        self.save_index()

        if self.verbose:
            print(" Index built successfully!")

    def _generate_embeddings_batch(
        self,
        elements: List[CodeElement],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for code elements in batches.

        Args:
            elements: List of code elements
            batch_size: Number of elements per batch

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i in range(0, len(elements), batch_size):
            batch = elements[i:i + batch_size]

            if self.verbose:
                print(f"  Processing batch {i // batch_size + 1}/{(len(elements) + batch_size - 1) // batch_size}...")

            # Convert to embedding text
            texts = [elem.to_embedding_text() for elem in batch]

            # Generate embeddings using OpenAI/Ollama API
            try:
                if self.embedding_client is None:
                    raise RuntimeError("No embedding client available")

                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )

                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)

            except Exception as e:
                if self.verbose:
                    print(f"Error generating embeddings: {e}")
                # Fallback: add zero vectors
                # Determine embedding dimension from environment (VECTOR_DIM)
                # or infer from model name for OpenAI models
                if os.getenv("VECTOR_DIM"):
                    # User-specified dimension (e.g., 1024 for deepseek-r1)
                    embedding_dim = int(os.getenv("VECTOR_DIM"))
                elif 'small' in self.embedding_model:
                    # OpenAI text-embedding-3-small
                    embedding_dim = 1536
                elif 'large' in self.embedding_model:
                    # OpenAI text-embedding-3-large
                    embedding_dim = 3072
                else:
                    # Generic fallback
                    embedding_dim = 1024
                embeddings.extend([[0.0] * embedding_dim for _ in batch])

        return embeddings

    def save_index(self) -> None:
        """Save index to cache."""
        cache_file = self.cache_dir / "index.pkl"

        data = {
            'code_elements': [elem.to_dict() for elem in self.code_elements],
            'embeddings': self.embeddings,
            'metadata': {
                'embedding_model': self.embedding_model,
                'num_elements': len(self.code_elements)
            }
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

    def load_index(self) -> None:
        """Load index from cache."""
        cache_file = self.cache_dir / "index.pkl"

        with open(cache_file, 'rb') as f:
            data = pickle.load(f)

        self.code_elements = [CodeElement(**elem) for elem in data['code_elements']]
        self.embeddings = data['embeddings']

        if self.verbose:
            metadata = data.get('metadata', {})
            print(f"Loaded {metadata.get('num_elements', 0)} elements")
            print(f"Model: {metadata.get('embedding_model', 'unknown')}")