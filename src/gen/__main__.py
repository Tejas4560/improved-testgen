# src/gen/__main__.py
"""
Command-line entry point for professional test generation.

Usage:
    python -m src.gen [options]
    
Examples:
    python -m src.gen --target ./my_project
    python -m src.gen --target ./backend --force
    TESTGEN_FORCE=true python -m src.gen
"""

import os
import sys


def main():
    """Main entry point with error handling."""
    try:
        from .enhanced_generate import main as generate_main
        return generate_main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the correct directory and all dependencies are installed.")
        return 1
    except KeyboardInterrupt:
        print("\nTest generation interrupted by user.")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
