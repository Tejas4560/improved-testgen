#!/usr/bin/env python3
"""
Coverage Gap Analyzer - Analyzes coverage reports and identifies uncovered code sections.

This module parses pytest coverage reports (coverage.xml, htmlcov, output.log) and identifies
specific lines, functions, and branches that are not covered by manual tests.
"""

import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


class CoverageGapAnalyzer:
    """Analyzes coverage gaps and generates detailed reports for AI test generation."""
    
    def __init__(self, target_root: str, current_dir: str):
        self.target_root = Path(target_root)
        self.current_dir = Path(current_dir)
        self.coverage_xml = self.current_dir / "coverage.xml"
        self.output_log = self.current_dir / "output.log"
        self.htmlcov_dir = self.current_dir / "htmlcov"
        
    def analyze_coverage(self) -> Dict[str, Any]:
        """
        Analyze all coverage sources and return comprehensive gap analysis.
        
        Returns:
            Dictionary with coverage gaps by file, function, and line
        """
        coverage_data = {
            "overall_coverage": 0.0,
            "total_statements": 0,
            "covered_statements": 0,
            "missing_statements": 0,
            "files_with_gaps": {},
            "uncovered_functions": [],
            "uncovered_classes": [],
            "uncovered_lines_by_file": {},
            "branch_coverage": {},
            "test_failures": [],
            "needs_ai_generation": False
        }
        
        # Parse coverage.xml for detailed line-by-line coverage
        if self.coverage_xml.exists():
            self._parse_coverage_xml(coverage_data)
        
        # Parse output.log for additional context
        if self.output_log.exists():
            self._parse_output_log(coverage_data)
        
        # Parse htmlcov for detailed function/class coverage
        if self.htmlcov_dir.exists():
            self._parse_htmlcov(coverage_data)
        
        # Determine if AI generation is needed
        coverage_data["needs_ai_generation"] = coverage_data["overall_coverage"] < 90.0
        
        return coverage_data
    
    def _parse_coverage_xml(self, coverage_data: Dict[str, Any]):
        """Parse coverage.xml to extract detailed coverage information."""
        try:
            tree = ET.parse(self.coverage_xml)
            root = tree.getroot()
            
            # Extract overall coverage
            line_rate = float(root.attrib.get("line-rate", 0))
            branch_rate = float(root.attrib.get("branch-rate", 0))
            coverage_data["overall_coverage"] = line_rate * 100
            coverage_data["branch_coverage"]["overall"] = branch_rate * 100
            
            # Parse each package and class
            for package in root.findall(".//package"):
                for cls in package.findall(".//class"):
                    filename = cls.attrib.get("filename", "")
                    
                    # Skip test files
                    if "test" in filename.lower():
                        continue
                    
                    # Extract line coverage for this file
                    file_coverage = {
                        "filename": filename,
                        "covered_lines": set(),
                        "missing_lines": set(),
                        "total_lines": 0,
                        "coverage_percentage": 0.0
                    }
                    
                    for line in cls.findall(".//line"):
                        line_num = int(line.attrib.get("number", 0))
                        hits = int(line.attrib.get("hits", 0))
                        
                        if hits > 0:
                            file_coverage["covered_lines"].add(line_num)
                        else:
                            file_coverage["missing_lines"].add(line_num)
                        
                        file_coverage["total_lines"] += 1
                    
                    # Calculate file coverage percentage
                    if file_coverage["total_lines"] > 0:
                        covered = len(file_coverage["covered_lines"])
                        total = file_coverage["total_lines"]
                        file_coverage["coverage_percentage"] = (covered / total) * 100
                    
                    # Store if there are coverage gaps
                    if file_coverage["missing_lines"]:
                        coverage_data["files_with_gaps"][filename] = file_coverage
                        coverage_data["uncovered_lines_by_file"][filename] = sorted(
                            file_coverage["missing_lines"]
                        )
                    source_path = self.target_root / filename
                    if source_path.exists():
                        self._identify_uncovered_elements(
                            filename,
                            source_path,
                            file_coverage["missing_lines"],
                            coverage_data
                        )
                    # Update totals
                    coverage_data["total_statements"] += file_coverage["total_lines"]
                    coverage_data["covered_statements"] += len(file_coverage["covered_lines"])
                    coverage_data["missing_statements"] += len(file_coverage["missing_lines"])
            
        except Exception as e:
            print(f"Error parsing coverage.xml: {e}")
    
    def _parse_output_log(self, coverage_data: Dict[str, Any]):
        """Parse output.log for test failures and coverage summary."""
        try:
            log_content = self.output_log.read_text(encoding="utf-8")
            
            # Extract test failures
            failure_pattern = re.compile(r"FAILED\s+(.+?)\s+-\s+(.+?)$", re.MULTILINE)
            for match in failure_pattern.finditer(log_content):
                test_name = match.group(1)
                failure_reason = match.group(2)
                coverage_data["test_failures"].append({
                    "test": test_name,
                    "reason": failure_reason
                })
            
            # Extract coverage summary
            coverage_pattern = re.compile(r"TOTAL\s+(\d+)\s+(\d+)\s+(\d+)%")
            match = coverage_pattern.search(log_content)
            if match:
                total_stmts = int(match.group(1))
                miss_stmts = int(match.group(2))
                coverage_pct = int(match.group(3))
                
                coverage_data["total_statements"] = total_stmts
                coverage_data["missing_statements"] = miss_stmts
                coverage_data["covered_statements"] = total_stmts - miss_stmts
                coverage_data["overall_coverage"] = float(coverage_pct)
            
        except Exception as e:
            print(f"Error parsing output.log: {e}")
    
    def _parse_htmlcov(self, coverage_data: Dict[str, Any]):
        """Parse htmlcov directory for detailed function/class coverage."""
        try:
            # Parse index.html for file-level summary
            index_html = self.htmlcov_dir / "index.html"
            if not index_html.exists():
                return
            
            index_content = index_html.read_text(encoding="utf-8")
            
            # Extract file coverage data from HTML table
            file_pattern = re.compile(
                r'<tr.*?><td.*?><a href="(.+?\.html)">(.+?)</a></td>'
                r'<td>(\d+)</td><td>(\d+)</td><td>(\d+)%</td>',
                re.DOTALL
            )
            
            for match in file_pattern.finditer(index_content):
                html_file = match.group(1)
                source_file = match.group(2)
                stmts = int(match.group(3))
                miss = int(match.group(4))
                coverage_pct = int(match.group(5))
                
                # Skip test files
                if "test" in source_file.lower():
                    continue
                
                # Parse individual file HTML for function-level details
                file_html_path = self.htmlcov_dir / html_file
                if file_html_path.exists():
                    self._parse_file_html(
                        source_file, 
                        file_html_path, 
                        coverage_data
                    )
            
        except Exception as e:
            print(f"Error parsing htmlcov: {e}")
    
    def _parse_file_html(self, source_file: str, html_path: Path, 
                        coverage_data: Dict[str, Any]):
        """Parse individual file HTML to extract function/class coverage."""
        try:
            html_content = html_path.read_text(encoding="utf-8")
            
            # Extract uncovered line numbers
            uncovered_pattern = re.compile(r'<span class="mis">(\d+)</span>')
            uncovered_lines = [
                int(match.group(1)) 
                for match in uncovered_pattern.finditer(html_content)
            ]
            
            if uncovered_lines:
                if source_file not in coverage_data["uncovered_lines_by_file"]:
                    coverage_data["uncovered_lines_by_file"][source_file] = []
                coverage_data["uncovered_lines_by_file"][source_file].extend(
                    uncovered_lines
                )
            
            # Try to identify uncovered functions/classes from the source
            source_path = self.target_root / source_file
            if source_path.exists():
                self._identify_uncovered_elements(
                    source_file, 
                    source_path, 
                    set(uncovered_lines),
                    coverage_data
                )
            
        except Exception as e:
            print(f"Error parsing file HTML {html_path}: {e}")
    
    def _identify_uncovered_elements(self, source_file: str, source_path: Path,
                                    uncovered_lines: Set[int],
                                    coverage_data: Dict[str, Any]):
        """Identify uncovered functions and classes from source code."""
        try:
            import ast
            
            source_code = source_path.read_text(encoding="utf-8")
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                # Check functions
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_lines = set(range(node.lineno, node.end_lineno + 1))
                    if func_lines & uncovered_lines:
                        coverage_data["uncovered_functions"].append({
                            "file": source_file,
                            "name": node.name,
                            "line_start": node.lineno,
                            "line_end": node.end_lineno,
                            "is_async": isinstance(node, ast.AsyncFunctionDef),
                            "uncovered_lines": sorted(func_lines & uncovered_lines)
                        })
                
                # Check classes
                elif isinstance(node, ast.ClassDef):
                    class_lines = set(range(node.lineno, node.end_lineno + 1))
                    if class_lines & uncovered_lines:
                        # Find uncovered methods within the class
                        uncovered_methods = []
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                method_lines = set(range(item.lineno, item.end_lineno + 1))
                                if method_lines & uncovered_lines:
                                    uncovered_methods.append({
                                        "name": item.name,
                                        "line_start": item.lineno,
                                        "line_end": item.end_lineno,
                                        "uncovered_lines": sorted(method_lines & uncovered_lines)
                                    })
                        
                        coverage_data["uncovered_classes"].append({
                            "file": source_file,
                            "name": node.name,
                            "line_start": node.lineno,
                            "line_end": node.end_lineno,
                            "uncovered_methods": uncovered_methods,
                            "total_uncovered_lines": len(class_lines & uncovered_lines)
                        })
        
        except Exception as e:
            print(f"Error identifying uncovered elements in {source_file}: {e}")
    
    def generate_gap_report(self, coverage_data: Dict[str, Any]) -> str:
        """Generate human-readable coverage gap report."""
        report = []
        report.append("=" * 80)
        report.append("COVERAGE GAP ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall summary
        report.append(f"Overall Coverage: {coverage_data['overall_coverage']:.2f}%")
        report.append(f"Total Statements: {coverage_data['total_statements']}")
        report.append(f"Covered Statements: {coverage_data['covered_statements']}")
        report.append(f"Missing Statements: {coverage_data['missing_statements']}")
        report.append(f"AI Generation Needed: {'YES' if coverage_data['needs_ai_generation'] else 'NO'}")
        report.append("")
        
        # Files with gaps
        if coverage_data["files_with_gaps"]:
            report.append("FILES WITH COVERAGE GAPS:")
            report.append("-" * 80)
            for filename, file_data in coverage_data["files_with_gaps"].items():
                report.append(f"\n {filename}")
                report.append(f"   Coverage: {file_data['coverage_percentage']:.2f}%")
                report.append(f"   Missing Lines: {len(file_data['missing_lines'])}")
                report.append(f"   Uncovered: {self._format_line_ranges(file_data['missing_lines'])}")
        
        report.append("")
        
        # Uncovered functions
        if coverage_data["uncovered_functions"]:
            report.append("UNCOVERED FUNCTIONS:")
            report.append("-" * 80)
            for func in coverage_data["uncovered_functions"]:
                report.append(f"\n {func['file']}::{func['name']} (lines {func['line_start']}-{func['line_end']})")
                report.append(f"   Async: {func['is_async']}")
                report.append(f"   Uncovered Lines: {self._format_line_ranges(func['uncovered_lines'])}")
        
        report.append("")
        
        # Uncovered classes
        if coverage_data["uncovered_classes"]:
            report.append("UNCOVERED CLASSES:")
            report.append("-" * 80)
            for cls in coverage_data["uncovered_classes"]:
                report.append(f"\n {cls['file']}::{cls['name']} (lines {cls['line_start']}-{cls['line_end']})")
                report.append(f"   Total Uncovered Lines: {cls['total_uncovered_lines']}")
                if cls["uncovered_methods"]:
                    report.append("   Uncovered Methods:")
                    for method in cls["uncovered_methods"]:
                        report.append(f"      - {method['name']} (lines {method['line_start']}-{method['line_end']})")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _format_line_ranges(self, lines: Set[int]) -> str:
        """Format line numbers into readable ranges (e.g., '1-5, 10, 15-20')."""
        if not lines:
            return "None"
        
        sorted_lines = sorted(lines)
        ranges = []
        start = sorted_lines[0]
        end = sorted_lines[0]
        
        for line in sorted_lines[1:]:
            if line == end + 1:
                end = line
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = line
                end = line
        
        # Add last range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        
        return ", ".join(ranges)
    
    def save_gap_analysis(self, coverage_data: Dict[str, Any], 
                         output_file: str = "coverage_gaps.json"):
        """Save coverage gap analysis to JSON file."""
        output_path = self.current_dir / output_file
        
        # Convert sets to lists for JSON serialization
        serializable_data = self._make_serializable(coverage_data)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"Coverage gap analysis saved to: {output_path}")
        return str(output_path)
    
    def _make_serializable(self, obj):
        """Convert sets and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, set):
            return sorted(list(obj))
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj


def main():
    """Main entry point for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze coverage gaps and generate reports for AI test generation"
    )
    parser.add_argument(
        "--target", 
        default=os.getenv("TARGET_ROOT", "target"),
        help="Target root directory"
    )
    parser.add_argument(
        "--current-dir",
        default=".",
        help="Current directory with coverage files"
    )
    parser.add_argument(
        "--output",
        default="coverage_gaps.json",
        help="Output JSON file for gap analysis"
    )
    
    args = parser.parse_args()
    
    analyzer = CoverageGapAnalyzer(args.target, args.current_dir)
    
    print("Analyzing coverage gaps...")
    coverage_data = analyzer.analyze_coverage()
    
    # Generate and print report
    report = analyzer.generate_gap_report(coverage_data)
    print(report)
    
    # Save detailed analysis
    analyzer.save_gap_analysis(coverage_data, args.output)
    
    # Return exit code based on coverage
    if coverage_data["needs_ai_generation"]:
        print(f"\n  Coverage is below 90% ({coverage_data['overall_coverage']:.2f}%)")
        print("AI test generation recommended for uncovered code")
        return 1
    else:
        print(f"\n Coverage is above 90% ({coverage_data['overall_coverage']:.2f}%)")
        print("No AI test generation needed")
        return 0


if __name__ == "__main__":
    sys.exit(main())