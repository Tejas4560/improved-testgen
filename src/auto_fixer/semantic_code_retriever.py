"""
Semantic Code Retriever - Finds Code Using Embedding Similarity

This module performs semantic search over the codebase index to find:
- Functions/classes relevant to test failures
- HTTP endpoints matching requests
- Dependencies and related code
- Handles misspellings, wrong imports, etc.

Key advantages over AST-based extraction:
- ✓ Semantic matching (not just string matching)
- ✓ Finds code even with wrong imports
- ✓ Handles typos and variations
- ✓ Discovers hidden dependencies
- ✓ Works with dynamic code patterns
"""

import numpy as np
import sys
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .codebase_indexer import CodebaseIndexer, CodeElement

# Add parent directory to path to import gen modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class SearchResult:
    """Result from semantic search."""
    code_element: CodeElement
    similarity_score: float
    rank: int

    def __repr__(self) -> str:
        return (
            f"SearchResult(name={self.code_element.name}, "
            f"type={self.code_element.element_type}, "
            f"score={self.similarity_score:.3f}, "
            f"rank={self.rank})"
        )


class SemanticCodeRetriever:
    """
    Retrieves code using semantic similarity search.

    Uses embeddings to find relevant code based on:
    - Test failure messages
    - HTTP endpoint requests
    - Function/class names (even misspelled)
    - Error tracebacks
    """

    def __init__(
        self,
        indexer: CodebaseIndexer,
        verbose: bool = False
    ):
        self.indexer = indexer
        self.verbose = verbose

        # Lazy-load embedding client
        self._embedding_client = None

    @property
    def embedding_client(self):
        """Lazy-load embedding client (Ollama or OpenAI)."""
        if self._embedding_client is None:
            # Import using spec_from_file_location to completely bypass gen package __init__.py
            import importlib.util
            from pathlib import Path

            # Get absolute path to ollama_client.py (works from any directory)
            current_file = Path(__file__).resolve()
            ollama_client_path = current_file.parent.parent / 'gen' / 'ollama_client.py'
            openai_client_path = current_file.parent.parent / 'gen' / 'openai_client.py'

            # Check if using Ollama (preferred)
            if os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_EMBED_MODEL"):
                try:
                    # Load ollama_client.py directly without triggering package __init__
                    spec = importlib.util.spec_from_file_location(
                        "ollama_client_retriever",
                        str(ollama_client_path)
                    )
                    ollama_module = importlib.util.module_from_spec(spec)
                    sys.modules['ollama_client_retriever'] = ollama_module
                    spec.loader.exec_module(ollama_module)

                    self._embedding_client = ollama_module.get_ollama_client()
                    if self.verbose:
                        print("Using Ollama for embeddings")
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to load Ollama client: {e}")
                    # Fall back to OpenAI
                    try:
                        spec = importlib.util.spec_from_file_location(
                            "openai_client_retriever_fallback",
                            str(openai_client_path)
                        )
                        openai_module = importlib.util.module_from_spec(spec)
                        sys.modules['openai_client_retriever_fallback'] = openai_module
                        spec.loader.exec_module(openai_module)

                        self._embedding_client = openai_module.create_client()
                        if self.verbose:
                            print("Using OpenAI for embeddings (fallback)")
                    except Exception as e2:
                        if self.verbose:
                            print(f"Both Ollama and OpenAI failed: {e2}")
                        self._embedding_client = None
            else:
                # Use OpenAI
                try:
                    spec = importlib.util.spec_from_file_location(
                        "openai_client_retriever_main",
                        str(openai_client_path)
                    )
                    openai_module = importlib.util.module_from_spec(spec)
                    sys.modules['openai_client_retriever_main'] = openai_module
                    spec.loader.exec_module(openai_module)

                    self._embedding_client = openai_module.create_client()
                    if self.verbose:
                        print("Using OpenAI for embeddings")
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to load OpenAI client: {e}")
                    self._embedding_client = None
        return self._embedding_client

    def search_by_query(
        self,
        query: str,
        top_k: int = 10,
        filter_type: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for code elements using a text query.

        Args:
            query: Natural language or code query
            top_k: Number of top results to return
            filter_type: Filter by element type ('function', 'class', etc.)
            min_similarity: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of SearchResult objects, sorted by similarity
        """
        if not self.indexer.code_elements:
            if self.verbose:
                print("Index is empty. Run indexer.build_index() first.")
            return []

        # Check if embedding client is available
        if self.embedding_client is None:
            if self.verbose:
                print("No embedding client available")
            return []

        # Generate query embedding
        try:
            response = self.embedding_client.embeddings.create(
                model=self.indexer.embedding_model,
                input=[query]
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            if self.verbose:
                print(f"Error generating query embedding: {e}")
            return []

        # Calculate similarities
        similarities = self._compute_similarities(
            query_embedding,
            self.indexer.embeddings
        )

        # Filter by type if specified
        if filter_type:
            indices = [
                i for i, elem in enumerate(self.indexer.code_elements)
                if elem.element_type == filter_type
            ]
        else:
            indices = list(range(len(self.indexer.code_elements)))

        # Filter by minimum similarity
        indices = [i for i in indices if similarities[i] >= min_similarity]

        # Sort by similarity (descending)
        indices.sort(key=lambda i: similarities[i], reverse=True)

        # Take top K
        indices = indices[:top_k]

        # Build results
        results = []
        for rank, idx in enumerate(indices, 1):
            results.append(SearchResult(
                code_element=self.indexer.code_elements[idx],
                similarity_score=similarities[idx],
                rank=rank
            ))

        return results

    def search_by_test_failure(
        self,
        test_code: str,
        error_message: str,
        traceback: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search for relevant code based on test failure information.

        Args:
            test_code: The failing test code
            error_message: Error message
            traceback: Full traceback
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        # Build comprehensive query from failure info
        query_parts = []

        # Add test code context
        query_parts.append("Test code:")
        query_parts.append(test_code[:500])  # First 500 chars

        # Add error message
        query_parts.append(f"\nError: {error_message}")

        # Add relevant traceback lines
        traceback_lines = traceback.split('\n')
        relevant_traceback = [
            line for line in traceback_lines
            if 'File' in line or 'def ' in line or 'class ' in line
        ]
        if relevant_traceback:
            query_parts.append("\nTraceback:")
            query_parts.append('\n'.join(relevant_traceback[:10]))

        query = '\n'.join(query_parts)

        if self.verbose:
            print(f"Searching for code matching test failure...")
            print(f"Query length: {len(query)} chars")

        return self.search_by_query(query, top_k=top_k)

    def search_by_http_endpoint(
        self,
        method: str,
        path: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Find HTTP endpoint handlers.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Endpoint path (/api/users, etc.)
            top_k: Number of results

        Returns:
            List of SearchResult objects
        """
        query = f"HTTP {method.upper()} {path} endpoint handler function"

        if self.verbose:
            print(f"Searching for {method} {path} endpoint...")

        # First try: filter to http_endpoint type
        results = self.search_by_query(
            query,
            top_k=top_k,
            filter_type='http_endpoint'
        )

        # If no HTTP endpoint elements found, search all functions
        if not results:
            if self.verbose:
                print(f"No HTTP endpoints found, searching all functions...")
            results = self.search_by_query(
                query,
                top_k=top_k,
                filter_type='function'
            )

        return results

    def search_by_function_name(
        self,
        function_name: str,
        top_k: int = 5,
        fuzzy: bool = True
    ) -> List[SearchResult]:
        """
        Find functions by name (supports fuzzy matching).

        Args:
            function_name: Function name to search for
            top_k: Number of results
            fuzzy: If True, uses semantic search; if False, exact match only

        Returns:
            List of SearchResult objects
        """
        if fuzzy:
            # Semantic search (handles typos)
            query = f"function named {function_name}"
            results = self.search_by_query(
                query,
                top_k=top_k,
                filter_type='function'
            )
        else:
            # Exact match
            results = []
            for idx, elem in enumerate(self.indexer.code_elements):
                if elem.element_type == 'function' and elem.name == function_name:
                    results.append(SearchResult(
                        code_element=elem,
                        similarity_score=1.0,
                        rank=1
                    ))
                    break

        return results

    def search_by_import_statement(
        self,
        import_statement: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Find code elements related to an import statement.

        Example:
            "from app.main import predict_batch"
            → Finds predict_batch function

        Args:
            import_statement: Import statement from test
            top_k: Number of results

        Returns:
            List of SearchResult objects
        """
        query = f"code imported as: {import_statement}"

        if self.verbose:
            print(f"Searching for import: {import_statement}")

        return self.search_by_query(query, top_k=top_k)

    def find_missing_target(
        self,
        target_name: str,
        context: str = "",
        threshold: float = 0.55
    ) -> Optional[SearchResult]:
        """
        Detect if a target function/class doesn't exist in codebase.

        Uses similarity threshold: if best match is below threshold,
        the target likely doesn't exist.

        Args:
            target_name: Name of target to find
            context: Additional context (error message, etc.)
            threshold: Similarity threshold (default 0.55)

        Returns:
            Best match if found, None if target doesn't exist
        """
        query = f"function or class named {target_name}"
        if context:
            query += f"\n{context}"

        results = self.search_by_query(query, top_k=1)

        if not results:
            if self.verbose:
                print(f"Target '{target_name}' not found in codebase")
            return None

        best_match = results[0]

        if best_match.similarity_score < threshold:
            if self.verbose:
                print(f"Target '{target_name}' not found (best match: {best_match.code_element.name}, score: {best_match.similarity_score:.3f})")
            return None

        if self.verbose:
            print(f"Found '{target_name}' → {best_match.code_element.name} (score: {best_match.similarity_score:.3f})")

        return best_match

    def verify_ast_extraction(
        self,
        extracted_names: List[str],
        test_context: str,
        threshold: float = 0.65
    ) -> Dict[str, bool]:
        """
        Verify if AST-extracted names are correct.

        Compares AST extraction results with embedding-based search
        to detect extraction errors.

        Args:
            extracted_names: Names extracted by AST parser
            test_context: Test code or error message
            threshold: Similarity threshold for verification

        Returns:
            Dict mapping names to verification status (True = correct)
        """
        verification = {}

        for name in extracted_names:
            result = self.find_missing_target(name, test_context, threshold)
            verification[name] = result is not None

        if self.verbose:
            correct = sum(1 for v in verification.values() if v)
            total = len(verification)
            print(f"AST extraction verification: {correct}/{total} correct")

        return verification

    def _compute_similarities(
        self,
        query_embedding: List[float],
        embeddings: List[List[float]]
    ) -> np.ndarray:
        """
        Compute cosine similarities between query and all embeddings.

        Args:
            query_embedding: Query embedding vector
            embeddings: List of embedding vectors

        Returns:
            Array of similarity scores
        """
        query_vec = np.array(query_embedding)
        embeddings_matrix = np.array(embeddings)

        # Normalize
        query_norm = query_vec / np.linalg.norm(query_vec)
        embeddings_norm = embeddings_matrix / np.linalg.norm(
            embeddings_matrix, axis=1, keepdims=True
        )

        # Cosine similarity
        similarities = np.dot(embeddings_norm, query_norm)

        return similarities

    def get_context_from_results(
        self,
        results: List[SearchResult],
        max_lines: int = 300,
        include_dependencies: bool = True
    ) -> str:
        """
        Extract source code context from search results.

        Args:
            results: List of SearchResult objects
            max_lines: Maximum total lines to extract
            include_dependencies: If True, include related code elements

        Returns:
            Formatted source code context string
        """
        if not results:
            return "# No relevant code found"

        output = []
        current_lines = 0
        seen_files = set()

        for result in results:
            elem = result.code_element

            # Group by file for better organization
            if elem.file_path not in seen_files:
                output.append(f"\n# File: {elem.file_path}")
                output.append("```python")
                seen_files.add(elem.file_path)

            # Add code element
            code_lines = elem.source_code.count('\n') + 1

            if current_lines + code_lines > max_lines:
                break

            output.append(f"\n# {elem.element_type}: {elem.name} (line {elem.line_start})")
            output.append(elem.source_code)

            current_lines += code_lines

        output.append("```")

        return '\n'.join(output)