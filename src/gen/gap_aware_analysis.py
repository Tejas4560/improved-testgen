# src/gen/gap_aware_analysis.py
"""
Gap-Aware Analysis Extension
Integrates coverage gap analysis into the existing src/gen workflow.

This module filters the full Python analysis to focus only on uncovered code
when gap-focused mode is enabled, while maintaining all the existing framework
detection, import resolution, and analysis capabilities.
"""

import json
import os
import pathlib
from typing import Any, Dict, List, Set, Tuple


def is_gap_focused_mode() -> bool:
    """Check if gap-focused generation mode is enabled."""
    return os.getenv("GAP_FOCUSED_MODE", "").lower() in ("true", "1", "yes")


def get_coverage_gaps_file() -> pathlib.Path:
    """Get the path to the coverage gaps file."""
    gaps_file = os.getenv("COVERAGE_GAPS_FILE", "coverage_gaps.json")
    return pathlib.Path(gaps_file)


def load_coverage_gaps() -> Dict[str, Any]:
    """Load coverage gap analysis data."""
    gaps_file = get_coverage_gaps_file()
    
    if not gaps_file.exists():
        print("Gap-focused mode enabled but no coverage gaps file found")
        print(f"Looking for: {gaps_file}")
        return {}
    
    try:
        with open(gaps_file, 'r') as f:
            gaps_data = json.load(f)
        
        print(f" Loaded coverage gaps analysis from: {gaps_file}")
        print(f"Current Coverage: {gaps_data.get('overall_coverage', 0):.2f}%")
        print(f"Uncovered Functions: {len(gaps_data.get('uncovered_functions', []))}")
        print(f"Uncovered Classes: {len(gaps_data.get('uncovered_classes', []))}")
        print(f"Files with Gaps: {len(gaps_data.get('files_with_gaps', {}))}")
        
        return gaps_data
        
    except Exception as e:
        print(f"Error loading coverage gaps: {e}")
        return {}


def filter_analysis_by_coverage_gaps(full_analysis: Dict[str, Any], 
                                    coverage_gaps: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter the full analysis to include only uncovered code elements.
    
    This maintains the same structure as full_analysis but only includes
    functions, classes, methods, and routes that have coverage gaps.
    
    Args:
        full_analysis: Complete analysis from analyzer.py
        coverage_gaps: Coverage gap data from coverage_gap_analyzer.py
    
    Returns:
        Filtered analysis containing only uncovered elements
    """
    print("\n Filtering analysis to focus on coverage gaps...")
    
    # Start with full analysis structure
    gap_focused_analysis = {
        "functions": [],
        "classes": [],
        "methods": [],
        "routes": [],
        "modules": full_analysis.get("modules", []),
        "imports": full_analysis.get("imports", []),
        "django_patterns": full_analysis.get("django_patterns", {}),
        "fastapi_routes": [],
        "properties": full_analysis.get("properties", []),
        "async_functions": [],
        "files_analyzed": full_analysis.get("files_analyzed", []),
        "nested_functions": [],
        "project_structure": full_analysis.get("project_structure", {}),
        "framework_info": full_analysis.get("framework_info", {}),
        
        # Add gap-specific metadata
        "gap_focused_mode": True,
        "coverage_context": {
            "current_coverage": coverage_gaps.get("overall_coverage", 0),
            "target_coverage": 90.0,
            "missing_statements": coverage_gaps.get("missing_statements", 0),
            "total_statements": coverage_gaps.get("total_statements", 0),
        }
    }
    
    # Extract uncovered elements from coverage gaps
    uncovered_functions = coverage_gaps.get("uncovered_functions", [])
    uncovered_classes = coverage_gaps.get("uncovered_classes", [])
    files_with_gaps = coverage_gaps.get("files_with_gaps", {})
    uncovered_lines_by_file = coverage_gaps.get("uncovered_lines_by_file", {})
    
    # Create lookup sets for efficient filtering
    uncovered_func_keys = {
        (func["file"], func["name"]) 
        for func in uncovered_functions
    }
    
    uncovered_class_keys = {
        (cls["file"], cls["name"]) 
        for cls in uncovered_classes
    }
    
    # Build set of uncovered method keys
    uncovered_method_keys = set()
    for cls_data in uncovered_classes:
        file_name = cls_data["file"]
        class_name = cls_data["name"]
        for method in cls_data.get("uncovered_methods", []):
            uncovered_method_keys.add((file_name, class_name, method["name"]))
    
    # Filter functions - include only uncovered ones
    for func in full_analysis.get("functions", []):
        func_key = (func.get("file"), func.get("name"))
        if func_key in uncovered_func_keys:
            # Add coverage metadata to function
            func_with_gaps = func.copy()
            
            # Find matching uncovered function data
            for unc_func in uncovered_functions:
                if (unc_func["file"], unc_func["name"]) == func_key:
                    func_with_gaps["coverage_gaps"] = {
                        "uncovered_lines": unc_func.get("uncovered_lines", []),
                        "line_start": unc_func.get("line_start"),
                        "line_end": unc_func.get("line_end"),
                    }
                    break
            
            gap_focused_analysis["functions"].append(func_with_gaps)
            
            # Also add to async_functions if applicable
            if func.get("is_async"):
                gap_focused_analysis["async_functions"].append(func_with_gaps)
    
    # Filter classes - include only those with uncovered code
    for cls in full_analysis.get("classes", []):
        cls_key = (cls.get("file"), cls.get("name"))
        if cls_key in uncovered_class_keys:
            # Add coverage metadata to class
            cls_with_gaps = cls.copy()
            
            # Find matching uncovered class data
            for unc_cls in uncovered_classes:
                if (unc_cls["file"], unc_cls["name"]) == cls_key:
                    cls_with_gaps["coverage_gaps"] = {
                        "total_uncovered_lines": unc_cls.get("total_uncovered_lines", 0),
                        "uncovered_methods": unc_cls.get("uncovered_methods", []),
                    }
                    break
            
            gap_focused_analysis["classes"].append(cls_with_gaps)
    
    # Filter methods - include only uncovered ones
    for method in full_analysis.get("methods", []):
        method_key = (method.get("file"), method.get("class"), method.get("name"))
        if method_key in uncovered_method_keys:
            # Add coverage metadata to method
            method_with_gaps = method.copy()
            
            # Find uncovered lines for this method
            file_name = method.get("file")
            if file_name in uncovered_lines_by_file:
                method_start = method.get("lineno", 0)
                method_end = method.get("end_lineno", method_start)
                uncovered_in_file = set(uncovered_lines_by_file[file_name])
                method_lines = set(range(method_start, method_end + 1))
                
                method_with_gaps["coverage_gaps"] = {
                    "uncovered_lines": sorted(method_lines & uncovered_in_file),
                }
            
            gap_focused_analysis["methods"].append(method_with_gaps)
    
    # Filter routes - include only from files with gaps
    files_with_gaps_set = set(files_with_gaps.keys())
    for route in full_analysis.get("routes", []):
        route_file = route.get("file", "")
        if route_file in files_with_gaps_set:
            gap_focused_analysis["routes"].append(route)
    
    for route in full_analysis.get("fastapi_routes", []):
        route_file = route.get("file", "")
        if route_file in files_with_gaps_set:
            gap_focused_analysis["fastapi_routes"].append(route)
    
    # Filter nested functions
    for func in full_analysis.get("nested_functions", []):
        func_file = func.get("file", "")
        if func_file in files_with_gaps_set:
            # Check if this nested function is in an uncovered area
            func_start = func.get("lineno", 0)
            func_end = func.get("end_lineno", func_start)
            
            if func_file in uncovered_lines_by_file:
                uncovered_in_file = set(uncovered_lines_by_file[func_file])
                func_lines = set(range(func_start, func_end + 1))
                
                if func_lines & uncovered_in_file:
                    gap_focused_analysis["nested_functions"].append(func)
    
    # Print filtering results
    print(f"\n Gap-Focused Analysis Results:")
    print(f"Original Functions: {len(full_analysis.get('functions', []))} → Uncovered: {len(gap_focused_analysis['functions'])}")
    print(f"Original Classes: {len(full_analysis.get('classes', []))} → Uncovered: {len(gap_focused_analysis['classes'])}")
    print(f"Original Methods: {len(full_analysis.get('methods', []))} → Uncovered: {len(gap_focused_analysis['methods'])}")
    print(f"Original Routes: {len(full_analysis.get('routes', []))} → Uncovered: {len(gap_focused_analysis['routes'])}")
    
    total_original = (
        len(full_analysis.get('functions', [])) + 
        len(full_analysis.get('classes', [])) + 
        len(full_analysis.get('methods', []))
    )
    total_filtered = (
        len(gap_focused_analysis['functions']) + 
        len(gap_focused_analysis['classes']) + 
        len(gap_focused_analysis['methods'])
    )
    
    reduction_pct = ((total_original - total_filtered) / max(total_original, 1)) * 100
    print(f"Total Reduction: {reduction_pct:.1f}% (focusing only on gaps)")
    
    return gap_focused_analysis


def enhance_prompt_with_coverage_context(coverage_gaps: Dict[str, Any]) -> str:
    """
    Build detailed coverage context to be added to AI prompts.
    
    This ensures the AI understands it should ONLY generate tests for
    uncovered code sections.
    """
    context_lines = []
    
    context_lines.append("\n" + "=" * 80)
    context_lines.append("CRITICAL: GAP-FOCUSED TEST GENERATION MODE")
    context_lines.append("=" * 80)
    context_lines.append("")
    context_lines.append("CURRENT SITUATION:")
    context_lines.append(f"- Existing manual test coverage: {coverage_gaps.get('overall_coverage', 0):.2f}%")
    context_lines.append(f"- Target coverage: 90%+")
    context_lines.append(f"- Gap to fill: {90 - coverage_gaps.get('overall_coverage', 0):.2f}%")
    context_lines.append(f"- Uncovered statements: {coverage_gaps.get('missing_statements', 0)}")
    context_lines.append("")
    
    context_lines.append("CRITICAL INSTRUCTIONS:")
    context_lines.append("-" * 80)
    context_lines.append("1. Generate tests ONLY for uncovered code sections specified below")
    context_lines.append("2. DO NOT generate tests for already-covered code")
    context_lines.append("3. Focus tests on hitting specific uncovered line numbers")
    context_lines.append("4. Prioritize covering multiple uncovered lines per test")
    context_lines.append("5. All targets in this request have coverage gaps that need filling")
    context_lines.append("")
    
    # Add file-specific gap information
    files_with_gaps = coverage_gaps.get("files_with_gaps", {})
    if files_with_gaps:
        context_lines.append("FILES WITH COVERAGE GAPS:")
        context_lines.append("-" * 80)
        for filename, file_data in list(files_with_gaps.items())[:10]:  # Limit to avoid token overflow
            context_lines.append(f"\n {filename}")
            context_lines.append(f"   Current Coverage: {file_data.get('coverage_percentage', 0):.2f}%")
            missing_lines = file_data.get("missing_lines", [])
            if missing_lines:
                if len(missing_lines) <= 20:
                    context_lines.append(f"   Uncovered Lines: {', '.join(map(str, sorted(missing_lines)))}")
                else:
                    context_lines.append(f"   Uncovered Lines: {', '.join(map(str, sorted(missing_lines)[:20]))}... ({len(missing_lines)} total)")
    
    context_lines.append("")
    
    # Add function-specific gap information
    uncovered_functions = coverage_gaps.get("uncovered_functions", [])
    if uncovered_functions:
        context_lines.append("UNCOVERED FUNCTIONS (Must Test):")
        context_lines.append("-" * 80)
        for func in uncovered_functions[:15]:  # Limit to avoid token overflow
            context_lines.append(f"\n🔧 {func['file']}::{func['name']} (lines {func['line_start']}-{func['line_end']})")
            uncovered_lines = func.get('uncovered_lines', [])
            if uncovered_lines:
                if len(uncovered_lines) <= 10:
                    context_lines.append(f"   Uncovered Lines: {', '.join(map(str, uncovered_lines))}")
                else:
                    context_lines.append(f"   Uncovered Lines: {', '.join(map(str, uncovered_lines[:10]))}... ({len(uncovered_lines)} total)")
    
    context_lines.append("")
    context_lines.append("=" * 80)
    context_lines.append("Remember: Only generate tests for the UNCOVERED code above!")
    context_lines.append("=" * 80)
    context_lines.append("")
    
    return "\n".join(context_lines)


def apply_gap_aware_filtering(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point: Apply gap-aware filtering if in gap-focused mode.
    
    This function is called by the main generate_all() in enhanced_generate.py
    to filter the analysis when coverage gaps are detected.
    
    Args:
        analysis: Full analysis from analyze_python_tree()
    
    Returns:
        Filtered analysis (gap-focused) or original analysis (normal mode)
    """
    if not is_gap_focused_mode():
        # Normal mode: return full analysis
        return analysis
    
    print("\n" + "=" * 80)
    print("GAP-FOCUSED MODE ACTIVE")
    print("=" * 80)
    print("Filtering analysis to target only uncovered code...")
    
    # Load coverage gaps
    coverage_gaps = load_coverage_gaps()
    
    if not coverage_gaps:
        print("No coverage gaps data available, falling back to full analysis")
        return analysis
    
    # Check if we actually need gap-focused generation
    current_coverage = coverage_gaps.get("overall_coverage", 100)
    if current_coverage >= 90:
        print(f"Coverage is already {current_coverage:.2f}% (≥90%)")
        print("No gap-focused generation needed- coverage goal achieved!")
        return {"skip_generation": True, "reason": "coverage_adequate"}
    
    # Filter analysis to only uncovered elements
    gap_focused_analysis = filter_analysis_by_coverage_gaps(analysis, coverage_gaps)
    
    # Add coverage context for prompts
    gap_focused_analysis["coverage_prompt_context"] = enhance_prompt_with_coverage_context(coverage_gaps)
    
    print("\n Gap-focused analysis prepared")
    print(f"   Targeting {gap_focused_analysis.get('coverage_context', {}).get('missing_statements', 0)} uncovered statements")
    print("=" * 80 + "\n")
    
    return gap_focused_analysis


def get_coverage_context_for_prompts() -> str:
    """
    Get coverage context string to add to AI prompts.
    
    This is called by enhanced_prompt.py to add gap-specific context
    to the test generation prompts.
    """
    if not is_gap_focused_mode():
        return ""
    
    coverage_gaps = load_coverage_gaps()
    if not coverage_gaps:
        return ""
    
    return enhance_prompt_with_coverage_context(coverage_gaps)