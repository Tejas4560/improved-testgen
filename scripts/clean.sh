#!/bin/bash
# AI TestGen - Cleanup Script
# Removes temporary files and artifacts

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "AI TestGen Cleanup Script"
echo "=========================================="
echo ""

# Count files before cleanup
TOTAL_REMOVED=0

echo "Cleaning up temporary files..."
echo ""

# Remove log files
if ls *.log 1> /dev/null 2>&1; then
    echo "Removing log files..."
    rm -f *.log
    echo "  ✓ Removed *.log files"
    ((TOTAL_REMOVED++))
fi

# Remove coverage files
if [ -f ".coverage" ] || [ -f ".coveragec" ] || [ -f "coverage.xml" ]; then
    echo "Removing coverage files..."
    rm -f .coverage .coveragec coverage.xml
    echo "  ✓ Removed coverage files"
    ((TOTAL_REMOVED++))
fi

if [ -d "htmlcov" ]; then
    echo "Removing HTML coverage reports..."
    rm -rf htmlcov/
    echo "  ✓ Removed htmlcov/"
    ((TOTAL_REMOVED++))
fi

# Remove pytest cache
if [ -d ".pytest_cache" ]; then
    echo "Removing pytest cache..."
    rm -rf .pytest_cache/
    echo "  ✓ Removed .pytest_cache/"
    ((TOTAL_REMOVED++))
fi

# Remove Python cache
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "  ✓ Removed __pycache__ and *.pyc files"
((TOTAL_REMOVED++))

# Remove test artifacts
if [ -f "test.db" ]; then
    rm -f test.db
    echo "  ✓ Removed test.db"
    ((TOTAL_REMOVED++))
fi

if [ -f "test-results.xml" ]; then
    rm -f test-results.xml
    echo "  ✓ Removed test-results.xml"
    ((TOTAL_REMOVED++))
fi

# Remove JSON artifacts (but keep package.json if exists)
echo "Removing JSON artifacts..."
rm -f manual_test_result.json coverage_gaps.json iteration_report.json \
      auto_fixer_report.json .pytest_*.json pytest_report.json 2>/dev/null || true
echo "  ✓ Removed JSON artifacts"
((TOTAL_REMOVED++))

# Remove SonarQube scanner work
if [ -d ".scannerwork" ]; then
    echo "Removing SonarQube scanner work..."
    rm -rf .scannerwork/
    echo "  ✓ Removed .scannerwork/"
    ((TOTAL_REMOVED++))
fi

# Remove codebase index
if [ -d ".codebase_index" ]; then
    echo "Removing codebase index..."
    rm -rf .codebase_index/
    echo "  ✓ Removed .codebase_index/"
    ((TOTAL_REMOVED++))
fi

# Clean logs directory (keep .gitkeep)
if [ -d "logs" ]; then
    echo "Cleaning logs directory..."
    find logs/ -type f ! -name '.gitkeep' -delete 2>/dev/null || true
    echo "  ✓ Cleaned logs/ directory"
    ((TOTAL_REMOVED++))
fi

echo ""
echo "=========================================="
echo "Cleanup Summary"
echo "=========================================="
echo "Categories cleaned: $TOTAL_REMOVED"
echo ""
echo "✓ Cleanup completed successfully!"
echo ""
echo "Note: Source code and configuration files were preserved."
echo ""
