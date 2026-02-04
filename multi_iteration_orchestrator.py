#!/usr/bin/env python3
"""
Multi-Iteration Test Generation Orchestrator

This orchestrator runs up to 3 iterations of AI-powered test generation
to achieve >90% code coverage on uncovered parts of the codebase.

Each iteration:
1. Analyzes coverage gaps (coverage_gap_analyzer.py)
2. Generates AI tests for uncovered code (src.gen)
3. Runs all tests (manual + AI generated)
4. Measures coverage and checks if target achieved

Target: 90%+ coverage on uncovered code parts
Max Iterations: 3
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class IterationMetrics:
    """Stores metrics for a single iteration"""
    def __init__(self, iteration: int):
        self.iteration = iteration
        self.start_time = time.time()
        self.end_time = None
        self.initial_coverage = 0.0
        self.final_coverage = 0.0
        self.coverage_gain = 0.0
        self.tests_generated = 0
        self.gaps_analyzed = 0
        self.success = False
        self.error = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "iteration": self.iteration,
            "duration_seconds": round(self.end_time - self.start_time, 2) if self.end_time else None,
            "initial_coverage": round(self.initial_coverage, 2),
            "final_coverage": round(self.final_coverage, 2),
            "coverage_gain": round(self.coverage_gain, 2),
            "tests_generated": self.tests_generated,
            "gaps_analyzed": self.gaps_analyzed,
            "success": self.success,
            "error": self.error
        }


class MultiIterationOrchestrator:
    """Orchestrates multiple iterations of test generation"""

    def __init__(
        self,
        target_dir: str,
        current_dir: str = ".",
        max_iterations: int = 3,
        target_coverage: float = 90.0,
        output_dir: str = "./tests/generated"
    ):
        self.target_dir = Path(target_dir)
        self.current_dir = Path(current_dir)
        self.max_iterations = max_iterations
        self.target_coverage = target_coverage
        self.output_dir = Path(output_dir)

        # Files and directories
        self.coverage_gaps_file = self.current_dir / "coverage_gaps.json"
        self.coverage_xml_file = self.current_dir / "coverage.xml"
        self.iteration_report_file = self.current_dir / "iteration_report.json"

        # Iteration tracking
        self.iterations: List[IterationMetrics] = []
        self.initial_coverage = 0.0
        self.final_coverage = 0.0
        self.target_achieved = False

    def print_header(self):
        """Print orchestrator header"""
        print("\n" + "="*80)
        print("MULTI-ITERATION TEST GENERATION ORCHESTRATOR")
        print("="*80)
        print(f"Target Directory: {self.target_dir}")
        print(f"Max Iterations: {self.max_iterations}")
        print(f"Target Coverage: {self.target_coverage}%")
        print(f"Output Directory: {self.output_dir}")
        print("="*80 + "\n")

    def run_command(self, cmd: List[str], description: str) -> Tuple[bool, str]:
        """
        Run a shell command and return success status and output

        Args:
            cmd: Command to run as list
            description: Description for logging

        Returns:
            Tuple of (success, output)
        """
        print(f"  ▶ {description}")
        print(f"    Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.current_dir,
                check=False
            )

            # Check if command succeeded (allow return code 0 or if explicitly allowed to fail)
            success = result.returncode == 0
            output = result.stdout + result.stderr

            if success:
                print(f"Success")
            else:
                print(f"Warning: Command returned code {result.returncode}")

            return success, output

        except Exception as e:
            print(f"Error: {str(e)}")
            return False, str(e)

    def get_current_coverage(self) -> Optional[float]:
        """
        Parse coverage from coverage_gaps.json

        Returns:
            Coverage percentage or None if not found
        """
        try:
            if self.coverage_gaps_file.exists():
                with open(self.coverage_gaps_file, 'r') as f:
                    gaps_data = json.load(f)
                    coverage = gaps_data.get('overall_coverage', 0.0)
                    return float(coverage)
            else:
                print(f" Coverage gaps file not found: {self.coverage_gaps_file}")
                return None
        except Exception as e:
            print(f"Error reading coverage: {str(e)}")
            return None

    def analyze_coverage_gaps(self, iteration: int, is_final: bool = False) -> bool:
        """
        Run coverage gap analyzer

        Args:
            iteration: Current iteration number
            is_final: If True, this is final analysis after tests run

        Returns:
            True if successful, False otherwise
        """
        phase = "Post-Test Analysis" if is_final else "Pre-Test Analysis"
        print(f"\n ITERATION {iteration}: {phase}")
        print("-" * 80)

        cmd = [
            "python", "src/coverage_gap_analyzer.py",
            "--target", str(self.target_dir),
            "--current-dir", str(self.current_dir),
            "--output", str(self.coverage_gaps_file)
        ]

        success, output = self.run_command(cmd, "Running coverage gap analyzer")

        if success or self.coverage_gaps_file.exists():
            # Load and display gap info
            try:
                with open(self.coverage_gaps_file, 'r') as f:
                    gaps = json.load(f)

                coverage = gaps.get('overall_coverage', 0.0)
                missing = gaps.get('missing_statements', 0)
                total = gaps.get('total_statements', 0)
                files_with_gaps = len(gaps.get('files_with_gaps', {}))

                print(f"\n  Coverage Analysis:")
                print(f"     Current Coverage: {coverage}%")
                print(f"     Missing Statements: {missing}/{total}")
                print(f"     Files with Gaps: {files_with_gaps}")

                return True

            except Exception as e:
                print(f"Error parsing coverage gaps: {str(e)}")
                return False
        else:
            print(f"Coverage gap analysis failed")
            return False

    def generate_ai_tests(self, iteration: int) -> bool:
        """
        Generate AI tests using gap-focused mode

        Args:
            iteration: Current iteration number

        Returns:
            True if successful, False otherwise
        """
        print(f"\n ITERATION {iteration}: Generating AI Tests (Gap-Focused)")
        print("-" * 80)

        # Set environment variable for gap-focused mode
        env = os.environ.copy()
        env['GAP_FOCUSED_MODE'] = 'true'

        cmd = [
            "python", "-m", "src.gen",
            "--target", str(self.target_dir),
            "--outdir", str(self.output_dir),
            "--force",
            "--coverage-mode", "gap-focused"
        ]

        print(f"  ▶ Generating tests for uncovered code")
        print(f"    Command: {' '.join(cmd)}")
        print(f"    Mode: GAP_FOCUSED_MODE=true")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.current_dir,
                env=env,
                check=False
            )

            output = result.stdout + result.stderr

            # Check if tests were generated
            if self.output_dir.exists():
                test_files = list(self.output_dir.glob("test_*.py"))
                print(f"Generated {len(test_files)} test files")
                return True
            else:
                print(f"Output directory not found: {self.output_dir}")
                return False

        except Exception as e:
            print(f"Error generating tests: {str(e)}")
            return False

    def run_all_tests(self, iteration: int) -> bool:
        """
        Run all tests (manual + generated) with coverage

        Args:
            iteration: Current iteration number

        Returns:
            True if successful, False otherwise
        """
        print(f"\n ITERATION {iteration}: Running All Tests with Coverage")
        print("-" * 80)

        # Build pytest command
        test_paths = []

        # Include manual tests if they exist
        manual_tests_dir = self.current_dir / "tests" / "manual"
        if manual_tests_dir.exists():
            test_paths.append(str(manual_tests_dir))
            print(f"Including manual tests: {manual_tests_dir}")

        # Include generated tests
        if self.output_dir.exists():
            test_paths.append(str(self.output_dir))
            print(f"Including generated tests: {self.output_dir}")

        if not test_paths:
            print(f"No test directories found")
            return False

        cmd = [
            "pytest",
            *test_paths,
            f"--cov={self.target_dir}",
            "--cov-config=pytest.ini",
            "--cov-report=xml",
            "--cov-report=html",
            "--cov-report=term-missing",
            "-v"
        ]

        success, output = self.run_command(cmd, "Running pytest with coverage")

        # Coverage report is generated even if some tests fail
        # So we check if coverage.xml exists
        if self.coverage_xml_file.exists():
            print(f"Coverage report generated")
            return True
        else:
            print(f"Coverage report not found")
            return success

    def run_iteration(self, iteration: int) -> IterationMetrics:
        """
        Run a single iteration of the test generation cycle

        Args:
            iteration: Iteration number (1-based)

        Returns:
            IterationMetrics object
        """
        metrics = IterationMetrics(iteration)

        print("\n" + "="*80)
        print(f"STARTING ITERATION {iteration}/{self.max_iterations}")
        print("="*80)

        try:
            # Get initial coverage for this iteration
            initial_cov = self.get_current_coverage()
            if initial_cov is not None:
                metrics.initial_coverage = initial_cov
                print(f"\n Initial Coverage: {initial_cov}%")

            # Step 1: Analyze coverage gaps
            if not self.analyze_coverage_gaps(iteration):
                metrics.error = "Coverage gap analysis failed"
                metrics.end_time = time.time()
                return metrics

            # Count gaps
            try:
                with open(self.coverage_gaps_file, 'r') as f:
                    gaps = json.load(f)
                    metrics.gaps_analyzed = gaps.get('missing_statements', 0)
            except:
                pass

            # Step 2: Generate AI tests
            if not self.generate_ai_tests(iteration):
                metrics.error = "Test generation failed"
                metrics.end_time = time.time()
                return metrics

            # Count generated tests
            if self.output_dir.exists():
                metrics.tests_generated = len(list(self.output_dir.glob("test_*.py")))

            # Step 3: Run all tests with coverage
            if not self.run_all_tests(iteration):
                print(f"Some tests may have failed, but continuing with coverage analysis")

            # Step 4: CRITICAL - Re-analyze coverage gaps to get updated coverage
            # This reads the NEW coverage.xml created by running tests above
            # and updates coverage_gaps.json so that:
            # 1. Final coverage for THIS iteration is correct
            # 2. Initial coverage for NEXT iteration is correct
            if not self.analyze_coverage_gaps(iteration, is_final=True):
                print(f"Could not re-analyze coverage, using previous data")

            # Get final coverage for this iteration (now reads freshly updated coverage_gaps.json)
            final_cov = self.get_current_coverage()
            if final_cov is not None:
                metrics.final_coverage = final_cov
                metrics.coverage_gain = final_cov - metrics.initial_coverage
                print(f"\n Final Coverage: {final_cov}%")
                print(f" Coverage Gain: +{metrics.coverage_gain:.2f}%")
            else:
                metrics.error = "Could not determine final coverage"

            metrics.success = True
            metrics.end_time = time.time()

        except Exception as e:
            metrics.error = str(e)
            metrics.end_time = time.time()
            print(f"\n Iteration {iteration} failed: {str(e)}")

        return metrics

    def check_target_achieved(self, current_coverage: float) -> bool:
        """Check if target coverage has been achieved"""
        return current_coverage >= self.target_coverage

    def print_iteration_summary(self, metrics: IterationMetrics):
        """Print summary for a single iteration"""
        print("\n" + "="*80)
        print(f" ITERATION {metrics.iteration} SUMMARY")
        print("="*80)
        print(f"Duration: {metrics.end_time - metrics.start_time:.2f}s")
        print(f"Initial Coverage: {metrics.initial_coverage:.2f}%")
        print(f"Final Coverage: {metrics.final_coverage:.2f}%")
        print(f"Coverage Gain: +{metrics.coverage_gain:.2f}%")
        print(f"Tests Generated: {metrics.tests_generated}")
        print(f"Gaps Analyzed: {metrics.gaps_analyzed}")
        print(f"Success: {'successful' if metrics.success else 'not successful'}")
        if metrics.error:
            print(f"Error: {metrics.error}")
        print("="*80 + "\n")

    def print_final_report(self):
        """Print final report for all iterations"""
        print("\n" + "="*80)
        print(" FINAL MULTI-ITERATION REPORT")
        print("="*80)

        print(f"\nTotal Iterations Run: {len(self.iterations)}")
        print(f"Initial Coverage: {self.initial_coverage:.2f}%")
        print(f"Final Coverage: {self.final_coverage:.2f}%")
        print(f"Total Coverage Gain: +{self.final_coverage - self.initial_coverage:.2f}%")
        print(f"Target Coverage: {self.target_coverage}%")
        print(f"Target Achieved: {'YES' if self.target_achieved else 'NO'}")

        print("\n Iteration Breakdown:")
        print("-" * 80)
        for metrics in self.iterations:
            status = "yes" if metrics.success else "No"
            print(f"  Iteration {metrics.iteration}: {status} "
                  f"{metrics.initial_coverage:.2f}% → {metrics.final_coverage:.2f}% "
                  f"(+{metrics.coverage_gain:.2f}%) | "
                  f"{metrics.tests_generated} tests | "
                  f"{metrics.end_time - metrics.start_time:.2f}s")

        print("\n" + "="*80)

        if self.target_achieved:
            print(" SUCCESS! Target coverage achieved!")
        else:
            remaining = self.target_coverage - self.final_coverage
            print(f"Target not achieved. Remaining gap: {remaining:.2f}%")
            print(f"Consider running additional iterations or manual test improvements")

        print("="*80 + "\n")

    def save_report(self):
        """Save iteration report to JSON file"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "target_dir": str(self.target_dir),
                "max_iterations": self.max_iterations,
                "target_coverage": self.target_coverage,
                "output_dir": str(self.output_dir)
            },
            "summary": {
                "total_iterations": len(self.iterations),
                "initial_coverage": round(self.initial_coverage, 2),
                "final_coverage": round(self.final_coverage, 2),
                "total_coverage_gain": round(self.final_coverage - self.initial_coverage, 2),
                "target_achieved": self.target_achieved
            },
            "iterations": [metrics.to_dict() for metrics in self.iterations]
        }

        try:
            with open(self.iteration_report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {self.iteration_report_file}")
        except Exception as e:
            print(f"Could not save report: {str(e)}")

    def run(self) -> bool:
        """
        Run the multi-iteration orchestrator

        Returns:
            True if target coverage achieved, False otherwise
        """
        self.print_header()

        start_time = time.time()

        # Get initial coverage
        self.initial_coverage = self.get_current_coverage() or 0.0
        print(f"Starting Coverage: {self.initial_coverage:.2f}%")
        print(f"Target Coverage: {self.target_coverage:.2f}%")
        print(f"Gap to Target: {self.target_coverage - self.initial_coverage:.2f}%\n")

        # Check if already at target
        if self.check_target_achieved(self.initial_coverage):
            print("Target coverage already achieved!")
            self.final_coverage = self.initial_coverage
            self.target_achieved = True
            return True

        # Run iterations
        for i in range(1, self.max_iterations + 1):
            metrics = self.run_iteration(i)
            self.iterations.append(metrics)
            self.print_iteration_summary(metrics)

            # Update final coverage
            if metrics.final_coverage > 0:
                self.final_coverage = metrics.final_coverage

            # Check if target achieved
            if self.check_target_achieved(self.final_coverage):
                print(f"Target coverage {self.target_coverage}% achieved!")
                self.target_achieved = True
                break

            # Check if no improvement
            if metrics.coverage_gain <= 0 and i < self.max_iterations:
                print(f"No coverage improvement in iteration {i}")
                print(f"Continuing to next iteration...")

        # Print final report
        end_time = time.time()
        total_time = end_time - start_time

        self.print_final_report()
        print(f"Total Time: {total_time:.2f}s\n")

        # Save report
        self.save_report()

        return self.target_achieved


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Iteration Test Generation Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (3 iterations, 90% target)
  python multi_iteration_orchestrator.py --target app

  # Run with custom iterations and target
  python multi_iteration_orchestrator.py --target app --iterations 5 --target-coverage 95

  # Specify custom output directory
  python multi_iteration_orchestrator.py --target app --outdir ./tests/ai_generated
"""
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target directory to analyze and generate tests for (e.g., 'app')"
    )

    parser.add_argument(
        "--current-dir",
        default=".",
        help="Current working directory (default: current directory)"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Maximum number of iterations to run (default: 3)"
    )

    parser.add_argument(
        "--target-coverage",
        type=float,
        default=90.0,
        help="Target coverage percentage to achieve (default: 90.0)"
    )

    parser.add_argument(
        "--outdir",
        default="./tests/generated",
        help="Output directory for generated tests (default: ./tests/generated)"
    )

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = MultiIterationOrchestrator(
        target_dir=args.target,
        current_dir=args.current_dir,
        max_iterations=args.iterations,
        target_coverage=args.target_coverage,
        output_dir=args.outdir
    )

    # Run orchestrator
    success = orchestrator.run()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()