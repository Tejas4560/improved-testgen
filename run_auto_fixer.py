#!/usr/bin/env python3
"""
Auto Test Fixer CLI

Command-line interface for running the automatic test fixer.

Usage:
    python run_auto_fixer.py [options]

Options:
    --test-dir DIR          Test directory (default: tests)
    --project-root DIR      Project root directory (default: .)
    --max-iterations N      Maximum iterations (default: 3)
    --pytest-args ARGS      Additional pytest arguments (comma-separated)

Examples:
    python run_auto_fixer.py
    python run_auto_fixer.py --test-dir tests/generated --max-iterations 5
    python run_auto_fixer.py --pytest-args "-v,-x"
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))
# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from auto_fixer import AutoTestFixerOrchestrator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Automatically fix failing tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--test-dir',
        default='tests',
        help='Test directory (default: tests)'
    )

    parser.add_argument(
        '--project-root',
        default='.',
        help='Project root directory (default: .)'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        default=3,
        help='Maximum number of fix iterations (default: 3)'
    )

    parser.add_argument(
        '--pytest-args',
        nargs='*',           # <-- FIX: Accept multiple values
        default=[],
        help='Additional pytest arguments (space-separated, like: --pytest-args -v -x)'
    )

    args = parser.parse_args()

    # Parse pytest args
    pytest_args = args.pytest_args or []

    # Create orchestrator
    print(f"Initializing Auto Test Fixer...")
    print(f"  Test directory: {args.test_dir}")
    print(f"  Project root: {args.project_root}")
    print(f"  Max iterations: {args.max_iterations}")
    if pytest_args:
        print(f"  Pytest args: {pytest_args}")

    orchestrator = AutoTestFixerOrchestrator(
        test_directory=args.test_dir,
        project_root=args.project_root,
        max_iterations=args.max_iterations
    )

    # Run the fixer
    try:
        summary = orchestrator.run(pytest_args)

        # Exit with appropriate code
        if summary['code_bugs'] > 0:
            print(f"\n {summary['code_bugs']} code bug(s) remain - these need manual fixes")
            sys.exit(1)
        elif summary['failed_fixes'] > 0:
            print(f"\n {summary['failed_fixes']} test mistake(s) could not be fixed")
            sys.exit(1)
        else:
            print(f"\n All test mistakes have been fixed!")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
