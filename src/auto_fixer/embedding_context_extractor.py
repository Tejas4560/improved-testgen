"""
Embedding-Enhanced Context Extractor

Hybrid approach that combines:
1. AST-based extraction (fast, precise when it works)
2. Embedding-based retrieval (robust, handles edge cases)

This provides the best of both worlds:
- Uses AST extraction as primary method
- Falls back to embedding search when AST fails
- Validates AST results with embeddings
- Handles all the edge cases AST can't

What this solves:
✓ Target function not found → embedding search finds it
✓ Wrong import paths → semantic search doesn't care
✓ Misspelled names → fuzzy matching finds them
✓ Token limit exceeded → embedding returns only relevant code
✓ Dynamic routes → HTTP endpoint search finds them
✓ Missing source files → comprehensive indexing finds them
"""

import os
from typing import Dict, List, Optional
from pathlib import Path

from .ast_context_extractor import ASTContextExtractor
from .codebase_indexer import CodebaseIndexer
from .semantic_code_retriever import SemanticCodeRetriever


class EmbeddingContextExtractor:
    """
    Hybrid context extractor using both AST and embeddings.

    Workflow:
    1. Try AST-based extraction first (fast)
    2. If AST finds nothing, use embedding search
    3. Validate AST results with embeddings
    4. Combine results intelligently
    """

    def __init__(
        self,
        project_root: str = ".",
        max_source_lines: int = 300,
        use_embeddings: bool = True,
        verbose: bool = False
    ):
        self.project_root = Path(project_root)
        self.max_source_lines = max_source_lines
        self.use_embeddings = use_embeddings
        self.verbose = verbose

        # Initialize AST extractor (always available)
        self.ast_extractor = ASTContextExtractor(
            project_root=str(project_root),
            verbose=verbose
        )

        # Initialize embedding components (lazy load)
        self._indexer = None
        self._retriever = None

        # Control flag from environment
        if os.getenv("DISABLE_EMBEDDINGS", "").lower() in ("true", "1", "yes"):
            self.use_embeddings = False
            if self.verbose:
                print("Embeddings disabled via DISABLE_EMBEDDINGS env var")

    @property
    def indexer(self) -> Optional[CodebaseIndexer]:
        """Lazy-load codebase indexer."""
        if not self.use_embeddings:
            return None

        if self._indexer is None:
            try:
                self._indexer = CodebaseIndexer(
                    project_root=str(self.project_root),
                    verbose=self.verbose
                )
                # Build or load index
                self._indexer.build_index()
            except Exception as e:
                if self.verbose:
                    print(f"Could not initialize indexer: {e}")
                self._indexer = None

        return self._indexer

    @property
    def retriever(self) -> Optional[SemanticCodeRetriever]:
        """Lazy-load semantic retriever."""
        if not self.use_embeddings or self.indexer is None:
            return None

        if self._retriever is None:
            self._retriever = SemanticCodeRetriever(
                indexer=self.indexer,
                verbose=self.verbose
            )

        return self._retriever

    def extract_context(
        self,
        test_file_path: str,
        test_function_name: str,
        error_message: str = ""
    ) -> Dict[str, str]:
        """
        Extract relevant source code context for a failing test.

        Hybrid approach:
        1. Try AST extraction
        2. If empty or insufficient, try embedding search
        3. Validate and combine results

        Args:
            test_file_path: Path to test file
            test_function_name: Name of failing test
            error_message: Error message with traceback

        Returns:
            Dict mapping file paths to code context
        """
        if self.verbose:
            print(f"\n Extracting context for {test_function_name}")

        # Step 1: Try AST extraction
        ast_context = self.ast_extractor.extract_context(
            test_file_path,
            test_function_name,
            error_message
        )

        # If AST extraction succeeded and embeddings disabled, return
        if ast_context and not self.use_embeddings:
            return ast_context

        # Step 2: Use embeddings if needed
        if self.use_embeddings and self.retriever:
            # Get embedding-based context
            embedding_context = self._extract_with_embeddings(
                test_file_path,
                test_function_name,
                error_message
            )

            # Combine AST and embedding results
            combined_context = self._combine_contexts(
                ast_context,
                embedding_context
            )

            if self.verbose:
                # Count functions/classes, not just files
                def count_elements(context_dict):
                    total = 0
                    for code in context_dict.values():
                        total += code.count('# function:') + code.count('# class:') + code.count('# http_endpoint:')
                    return total

                ast_files = len(ast_context)
                ast_elements = count_elements(ast_context)
                embed_files = len(embedding_context)
                embed_elements = count_elements(embedding_context)
                combined_files = len(combined_context)
                combined_elements = count_elements(combined_context)

                print(f"Context extraction results:")
                print(f"AST: {ast_elements} elements in {ast_files} files")
                print(f"Embeddings: {embed_elements} elements in {embed_files} files")
                print(f"Combined: {combined_elements} elements in {combined_files} files (deduplicated)")

            return combined_context

        # Fallback: return AST results (might be empty)
        return ast_context

    def _extract_with_embeddings(
        self,
        test_file_path: str,
        test_function_name: str,
        error_message: str
    ) -> Dict[str, str]:
        """
        Extract context using embedding-based semantic search.

        Args:
            test_file_path: Path to test file
            test_function_name: Test function name
            error_message: Error message

        Returns:
            Dict mapping file paths to code
        """
        if not self.retriever:
            return {}

        # Read test code
        try:
            with open(test_file_path, 'r') as f:
                test_content = f.read()

            # Extract the specific test function
            import ast
            tree = ast.parse(test_content)
            base_test_name = test_function_name.split('[')[0]
            test_code = ""

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == base_test_name:
                        test_code = ast.unparse(node)
                        break

            if not test_code:
                test_code = test_content  # Fallback

        except Exception as e:
            if self.verbose:
                print(f"Could not read test file: {e}")
            test_code = ""

        # Perform semantic search
        results = self.retriever.search_by_test_failure(
            test_code=test_code,
            error_message=error_message,
            traceback=error_message,  # Traceback is part of error message
            top_k=10
        )

        if self.verbose and results:
            print(f"Embedding search found {len(results)} matches:")
            for result in results[:3]:
                print(f"  {result.rank}. {result.code_element.name} ({result.code_element.element_type}) - score: {result.similarity_score:.3f}")

        # Build context from results
        context = {}
        current_lines = 0

        for result in results:
            elem = result.code_element
            file_path = elem.file_path

            # Check line limit
            code_lines = elem.source_code.count('\n') + 1
            if current_lines + code_lines > self.max_source_lines:
                break

            # Add to context
            if file_path not in context:
                context[file_path] = []

            context[file_path].append({
                'name': elem.name,
                'type': elem.element_type,
                'code': elem.source_code,
                'line_start': elem.line_start
            })

            current_lines += code_lines

        # Format context as strings
        formatted_context = {}
        for file_path, elements in context.items():
            # Sort by line number
            elements.sort(key=lambda x: x['line_start'])

            # Combine code
            code_parts = []
            for elem in elements:
                code_parts.append(f"# {elem['type']}: {elem['name']} (line {elem['line_start']})")
                code_parts.append(elem['code'])
                code_parts.append("")

            formatted_context[file_path] = '\n'.join(code_parts)

        return formatted_context

    def _combine_contexts(
        self,
        ast_context: Dict[str, str],
        embedding_context: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Intelligently combine AST and embedding contexts.

        Strategy:
        - Merge functions from both AST and embeddings for the same file
        - Remove duplicate functions (by name)
        - Prefer AST version when duplicate (more precise)
        - Add embedding results for new functions/files

        Args:
            ast_context: Context from AST extraction
            embedding_context: Context from embedding search

        Returns:
            Combined context dict
        """
        combined = {}

        # Track which functions we've seen (to avoid duplicates)
        seen_functions = {}  # {file_path: {function_name, ...}}

        # Helper to parse function names from code string
        def extract_function_names(code_string: str) -> set:
            """Extract function/class names from formatted code string."""
            names = set()
            for line in code_string.split('\n'):
                if line.startswith('# function:') or line.startswith('# class:') or line.startswith('# http_endpoint:'):
                    # Parse "# function: my_func (line 123)"
                    parts = line.split(':')
                    if len(parts) >= 2:
                        name_part = parts[1].strip()
                        # Remove "(line X)" suffix
                        if '(' in name_part:
                            name_part = name_part.split('(')[0].strip()
                        names.add(name_part)
            return names

        # Step 1: Add all AST results (more precise)
        for file_path, code in ast_context.items():
            combined[file_path] = code
            seen_functions[file_path] = extract_function_names(code)

        # Step 2: Merge embedding results
        for file_path, embed_code in embedding_context.items():
            embed_funcs = extract_function_names(embed_code)

            if file_path not in combined:
                # New file from embeddings - add it
                combined[file_path] = embed_code
                seen_functions[file_path] = embed_funcs
            else:
                # File exists in AST results - merge new functions only
                existing_funcs = seen_functions[file_path]
                new_funcs = embed_funcs - existing_funcs

                if new_funcs:
                    # Extract only new functions from embedding code
                    new_code_parts = []
                    lines = embed_code.split('\n')
                    in_function = False
                    current_func_name = None
                    current_func_lines = []

                    for line in lines:
                        if line.startswith('# function:') or line.startswith('# class:') or line.startswith('# http_endpoint:'):
                            # Save previous function if it was new
                            if in_function and current_func_name in new_funcs:
                                new_code_parts.extend(current_func_lines)
                                new_code_parts.append("")

                            # Start new function
                            parts = line.split(':')
                            if len(parts) >= 2:
                                name_part = parts[1].strip()
                                if '(' in name_part:
                                    name_part = name_part.split('(')[0].strip()
                                current_func_name = name_part
                                current_func_lines = [line]
                                in_function = True
                        elif in_function:
                            current_func_lines.append(line)

                    # Don't forget last function
                    if in_function and current_func_name in new_funcs:
                        new_code_parts.extend(current_func_lines)
                        new_code_parts.append("")

                    # Append new functions to existing file
                    if new_code_parts:
                        combined[file_path] = combined[file_path].rstrip() + "\n\n" + '\n'.join(new_code_parts)
                        seen_functions[file_path].update(new_funcs)

        return combined

    def get_full_context_string(
        self,
        test_file_path: str,
        test_function_name: str,
        error_message: str = ""
    ) -> str:
        """
        Get formatted context string for LLM.

        Args:
            test_file_path: Path to test file
            test_function_name: Test function name
            error_message: Error message

        Returns:
            Formatted context string
        """
        context = self.extract_context(
            test_file_path,
            test_function_name,
            error_message
        )

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

    def verify_extraction_quality(
        self,
        test_file_path: str,
        test_function_name: str,
        error_message: str
    ) -> Dict[str, any]:
        """
        Verify quality of context extraction.

        Compares AST and embedding results to detect issues.

        Args:
            test_file_path: Path to test file
            test_function_name: Test function name
            error_message: Error message

        Returns:
            Dict with quality metrics
        """
        # Extract with both methods
        ast_context = self.ast_extractor.extract_context(
            test_file_path,
            test_function_name,
            error_message
        )

        embedding_context = {}
        if self.use_embeddings and self.retriever:
            embedding_context = self._extract_with_embeddings(
                test_file_path,
                test_function_name,
                error_message
            )

        # Calculate metrics
        ast_files = set(ast_context.keys())
        embedding_files = set(embedding_context.keys())

        metrics = {
            'ast_file_count': len(ast_files),
            'embedding_file_count': len(embedding_files),
            'overlap_count': len(ast_files & embedding_files),
            'ast_only': list(ast_files - embedding_files),
            'embedding_only': list(embedding_files - ast_files),
            'extraction_quality': 'good' if ast_context else 'failed'
        }

        # If AST found nothing but embeddings found something
        if not ast_context and embedding_context:
            metrics['extraction_quality'] = 'ast_failed_embedding_saved'

        # If both found different files
        if ast_files and embedding_files and not (ast_files & embedding_files):
            metrics['extraction_quality'] = 'divergent_results'

        return metrics