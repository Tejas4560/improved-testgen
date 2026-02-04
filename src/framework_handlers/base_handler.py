# src/framework_handlers/base_handler.py
"""
Base framework handler with common functionality for all framework handlers.
"""

import ast
import pathlib
from typing import Any, Dict, List, Optional, Set, Tuple


class BaseFrameworkHandler:
    """Base class for all framework handlers."""
    
    def __init__(self):
        self.framework_name = "base"
        self.supported_patterns = set()
    
    def can_handle(self, analysis: Dict[str, Any]) -> bool:
        """Check if this handler can handle the project based on analysis."""
        return False
    
    def analyze_framework_specifics(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:
        """Analyze framework-specific patterns in the AST."""
        return {}
    
    def generate_framework_tests(self, compact_analysis: Dict[str, Any], 
                               target_root: pathlib.Path,
                               output_dir: pathlib.Path) -> List[str]:
        """Generate framework-specific tests."""
        return []
    
    def get_framework_dependencies(self) -> List[str]:
        """Get framework-specific dependencies required for testing."""
        return []
    
    def setup_framework_environment(self, target_root: pathlib.Path):
        """Setup framework-specific testing environment."""
        pass
    
    def detect_framework_patterns(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect framework-specific patterns in the analysis."""
        return {}