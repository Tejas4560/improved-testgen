import os
import json
import sys
from typing import List, Dict

def find_common_test_root(test_dirs: List[str]) -> str:
    """
    Find the common parent directory for all test directories.
    Always returns the top-level 'tests' directory, not subdirectories within it.

    Example:
        test_dirs = ['/repo/tests/test_models', '/repo/tests/unit']
        Returns: '/repo/tests' (not '/repo/tests/test_models')
    """
    if not test_dirs:
        return ""

    # Valid test directory names (exact matches or starting with these)
    test_dir_patterns = ['tests', 'test']

    def is_test_directory(dirname: str) -> bool:
        """Check if directory name is a test directory."""
        dirname_lower = dirname.lower()
        # Exact match for 'test' or 'tests'
        if dirname_lower in ['test', 'tests']:
            return True
        # Starts with 'test_' (e.g., 'test_integration')
        if dirname_lower.startswith('test_'):
            return True
        if dirname_lower.endswith('_test'):
            return True
        return False

    # Find common prefix of all paths
    if len(test_dirs) == 1:
        # Single directory - go up to find the 'test' parent directory
        single_dir = test_dirs[0]
        parts = single_dir.split(os.sep)

        # Find the directory named 'test' or 'tests'
        for i, part in enumerate(parts):
            if is_test_directory(part):
                # Return this directory, not subdirectories within it
                return os.sep.join(parts[:i+1])

        # If no 'test' directory found, return the directory itself
        return single_dir

    # Multiple directories - find common path
    common = os.path.commonpath(test_dirs)

    # Check if common path ends with a test directory
    common_parts = common.split(os.sep)
    if common_parts and is_test_directory(common_parts[-1]):
        return common

    # If not, search for test directory in paths
    for d in test_dirs:
        parts = d.split(os.sep)
        for i, part in enumerate(parts):
            if is_test_directory(part):
                return os.sep.join(parts[:i+1])

    return common


def find_all_manual_test_dirs(repo_root: str = ".") -> Dict[str, any]:
    """
    Finds ALL test files in the repository (manual OR AI-generated).
    No distinction between types - just collect all unique test files.

    Returns:
    {
        "test_root": "/path/to/tests",  # Common root directory
        "files_by_relative_path": {
            "test_user.py": "/full/path/to/tests/test_user.py",
            "test_models/test_user.py": "/full/path/to/tests/test_models/test_user.py"
        },
        "all_test_dirs": [list of directories containing tests]
    }

    Preserves directory structure to avoid import conflicts
    Only includes folders that contain .py test files
    """
    candidate_dirs = {}

    for root, dirs, files in os.walk(repo_root):
        # Skip unwanted folders (cache, venv, etc)
        if any(skip in root.lower() for skip in ["__pycache__", ".git", "venv", "env"]):
            continue

        # Include any folder under a test-related path
        if not any("test" in part for part in root.lower().split(os.sep)):
            continue

        test_files = []
        for file in files:
            # Include conftest.py for pytest configuration
            # Include test files (test_*.py or *_test.py)
            if file.endswith(".py") and (
                file == "conftest.py" or
                file.startswith("test_") or
                file.endswith("_test.py") or
                "test" in file.lower()
            ):
                file_path = os.path.join(root, file)
                try:
                    # Validate the file is readable
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        f.read()
                    test_files.append(file_path)
                except Exception:
                    continue

        if test_files:
            candidate_dirs[root] = test_files

    # Find common test root
    if candidate_dirs:
        test_root = find_common_test_root(list(candidate_dirs.keys()))

        # Build files_by_relative_path with unique files
        files_by_relative_path = {}
        for test_dir, files in candidate_dirs.items():
            for file_path in files:
                # Calculate relative path from test_root
                rel_path = os.path.relpath(file_path, test_root)
                # Use relative path as key to ensure uniqueness
                if rel_path not in files_by_relative_path:
                    files_by_relative_path[rel_path] = file_path

        return {
            "test_root": test_root,
            "files_by_relative_path": files_by_relative_path,
            "all_test_dirs": list(candidate_dirs.keys())
        }

    return {
        "test_root": "",
        "files_by_relative_path": {},
        "all_test_dirs": []
    }


def main():
    """Main entry point for detecting all test directories."""
    repo_root = sys.argv[1] if len(sys.argv) > 1 else "."

    print(f"Scanning repository for test files in: {os.path.abspath(repo_root)}")

    detection_result = find_all_manual_test_dirs(repo_root)

    # Check if we have any tests
    has_tests = bool(detection_result.get("files_by_relative_path"))

    if not has_tests:
        print("No test files found.")
        result = {
            "manual_tests_found": False,
            "test_root": "",
            "manual_test_paths": [],
            "test_files_count": 0,
            "files_by_relative_path": {}
        }
    else:
        result = {
            "manual_tests_found": True,
            "test_root": detection_result["test_root"],
            "manual_test_paths": detection_result["all_test_dirs"],
            "test_files_count": len(detection_result["files_by_relative_path"]),
            "files_by_relative_path": detection_result["files_by_relative_path"]
        }

        print(f"\n Found {result['test_files_count']} test files")
        print(f"Test root: {result['test_root']}")
        print(f"Test directories: {len(result['manual_test_paths'])}")
        print("\n Test files:")
        for rel_path in sorted(result["files_by_relative_path"].keys()):
            print(f"   {rel_path}")

    print("\n Detection Result:")
    print(json.dumps(result, indent=2))

    # Save detailed result with structure preservation info
    with open("manual_test_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return 0 if result["manual_tests_found"] else 1


if __name__ == "__main__":
    sys.exit(main())