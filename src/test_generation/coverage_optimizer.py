# src/test_generation/coverage_optimizer.py
"""
Coverage optimizer for maximizing test coverage.
"""

from typing import Any, Dict, List, Set, Tuple


class CoverageOptimizer:
    """Optimizes test generation for maximum coverage."""
    
    def __init__(self):
        self.coverage_target = 95  # Minimum coverage target
    
    def optimize_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize analysis for maximum coverage potential."""
        
        # Calculate coverage potential
        total_elements = (
            len(analysis.get("functions", [])) +
            len(analysis.get("classes", [])) +
            len(analysis.get("methods", [])) +
            len(analysis.get("routes", []))
        )
        
        # Add coverage metrics to analysis
        analysis["coverage_metrics"] = {
            "total_testable_elements": total_elements,
            "coverage_target": self.coverage_target,
            "estimated_tests_needed": self._calculate_tests_needed(total_elements),
            "complexity_score": self._calculate_complexity_score(analysis)
        }
        
        return analysis
    
    def optimize_targets(self, compact_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize targets for maximum coverage generation."""
        
        # Prioritize targets based on coverage potential
        functions = self._prioritize_functions(compact_analysis.get("functions", []))
        classes = self._prioritize_classes(compact_analysis.get("classes", []))
        methods = self._prioritize_methods(compact_analysis.get("methods", []))
        routes = self._prioritize_routes(compact_analysis.get("routes", []))
        
        optimized = compact_analysis.copy()
        optimized["functions"] = functions
        optimized["classes"] = classes
        optimized["methods"] = methods
        optimized["routes"] = routes
        
        # Add coverage optimization info
        optimized["coverage_optimization"] = {
            "optimized_functions": len(functions),
            "optimized_classes": len(classes),
            "optimized_methods": len(methods),
            "optimized_routes": len(routes),
            "total_optimized_targets": len(functions) + len(classes) + len(methods) + len(routes)
        }
        
        return optimized
    
    def _calculate_tests_needed(self, total_elements: int) -> int:
        """Calculate number of tests needed to achieve coverage target."""
        # Heuristic: 2-3 tests per element for >95% coverage
        return total_elements * 3
    
    def _calculate_complexity_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate complexity score of the codebase."""
        functions = analysis.get("functions", [])
        classes = analysis.get("classes", [])
        methods = analysis.get("methods", [])
        
        total_elements = len(functions) + len(classes) + len(methods)
        if total_elements == 0:
            return 0.0
        
        # Calculate average complexity metrics
        total_args = sum(func.get("args_count", 0) for func in functions)
        total_methods = len(methods)
        total_async = len(analysis.get("async_functions", []))
        
        complexity_score = (
            (total_args / max(len(functions), 1)) * 0.3 +
            (total_methods / max(total_elements, 1)) * 0.4 +
            (total_async / max(total_elements, 1)) * 0.3
        )
        
        return round(complexity_score, 2)
    
    def _prioritize_functions(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize functions for testing based on coverage potential."""
        if not functions:
            return []
        
        # Score each function
        scored_functions = []
        for func in functions:
            score = 0
            
            # Higher priority for functions with more parameters
            score += func.get("args_count", 0) * 2
            
            # Higher priority for async functions
            if func.get("is_async", False):
                score += 5
            
            # Higher priority for functions with decorators
            if func.get("has_decorators", False):
                score += 3
            
            # Higher priority for top-level functions
            if func.get("is_top_level", False):
                score += 2
            
            scored_functions.append((score, func))
        
        # Sort by score (descending)
        scored_functions.sort(key=lambda x: x[0], reverse=True)
        
        return [func for score, func in scored_functions]
    
    def _prioritize_classes(self, classes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize classes for testing based on coverage potential."""
        if not classes:
            return []
        
        # Score each class
        scored_classes = []
        for cls in classes:
            score = 0
            
            # Higher priority for classes with more methods
            score += cls.get("method_count", 0) * 3
            
            # Higher priority for classes with complex inheritance
            bases = cls.get("bases", [])
            if bases and bases != ['object']:
                score += len(bases) * 2
            
            # Higher priority for top-level classes
            if cls.get("is_top_level", False):
                score += 2
            
            scored_classes.append((score, cls))
        
        # Sort by score (descending)
        scored_classes.sort(key=lambda x: x[0], reverse=True)
        
        return [cls for score, cls in scored_classes]
    
    def _prioritize_methods(self, methods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize methods for testing based on coverage potential."""
        if not methods:
            return []
        
        # Score each method
        scored_methods = []
        for method in methods:
            score = 0
            
            # Higher priority for methods with more parameters
            score += method.get("args_count", 0) * 2
            
            # Higher priority for async methods
            if method.get("is_async", False):
                score += 5
            
            # Higher priority for properties
            if method.get("is_property", False):
                score += 3
            
            # Higher priority for class methods and static methods
            if method.get("is_classmethod", False) or method.get("is_staticmethod", False):
                score += 2
            
            # Higher priority for methods with decorators
            if method.get("has_decorators", False):
                score += 2
            
            scored_methods.append((score, method))
        
        # Sort by score (descending)
        scored_methods.sort(key=lambda x: x[0], reverse=True)
        
        return [method for score, method in scored_methods]
    
    def _prioritize_routes(self, routes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize routes for testing based on coverage potential."""
        if not routes:
            return []
        
        # Score each route
        scored_routes = []
        for route in routes:
            score = 0
            
            # Higher priority for routes with specific HTTP methods
            method = route.get("method", "").upper()
            if method in ["POST", "PUT", "DELETE"]:
                score += 5
            elif method in ["GET"]:
                score += 3
            
            # Higher priority for complex routes (with path parameters)
            path = route.get("path", "")
            if "{" in path or "}" in path or ":" in path:
                score += 4
            
            scored_routes.append((score, route))
        
        # Sort by score (descending)
        scored_routes.sort(key=lambda x: x[0], reverse=True)
        
        return [route for score, route in scored_routes]
    
    def validate_coverage_potential(self, analysis: Dict[str, Any], 
                                  generated_tests_count: int) -> Tuple[bool, str]:
        """Validate if generated tests can achieve coverage target."""
        coverage_metrics = analysis.get("coverage_metrics", {})
        tests_needed = coverage_metrics.get("estimated_tests_needed", 0)
        
        if generated_tests_count >= tests_needed:
            return True, f"Sufficient tests generated ({generated_tests_count}/{tests_needed}) for {self.coverage_target}% coverage"
        else:
            coverage_estimate = min(95, (generated_tests_count / max(tests_needed, 1)) * 100)
            return False, f"Insufficient tests ({generated_tests_count}/{tests_needed}). Estimated coverage: {coverage_estimate:.1f}%"