#!/usr/bin/env python3
"""
Materialize generated test files into source-file-mapped test files.

Transforms internal timestamped test files:
    test_unit_20250207_143022_01.py
    test_integ_20250207_143022_01.py
    test_e2e_20250207_143022_01.py

Into clean, source-file-mapped files with class-based separation:
    test_app.py       -> TestAppUnit, TestAppIntegration, TestAppE2E
    test_utils.py     -> TestUtilsUnit, ...

This is a pure transformation step — same tests, different presentation.
Runs after all tests pass and coverage is validated.
"""

import ast
import json
import os
import pathlib
import re
import sys
import textwrap
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


# Pattern to match internal timestamped test files
GENERATED_RE = re.compile(r'^test_(unit|integ|e2e)_\d{8}_\d{6}_\d+\.py$')

# Standard-library top-level package names (Python 3.8+)
_STDLIB = {
    '__future__', '_thread', 'abc', 'aifc', 'argparse', 'array', 'ast',
    'asynchat', 'asyncio', 'asyncore', 'atexit', 'base64', 'bdb',
    'binascii', 'binhex', 'bisect', 'builtins', 'bz2', 'calendar',
    'cgi', 'cgitb', 'chunk', 'cmath', 'cmd', 'code', 'codecs',
    'codeop', 'collections', 'colorsys', 'compileall', 'concurrent',
    'configparser', 'contextlib', 'contextvars', 'copy', 'copyreg',
    'cProfile', 'crypt', 'csv', 'ctypes', 'curses', 'dataclasses',
    'datetime', 'dbm', 'decimal', 'difflib', 'dis', 'distutils',
    'doctest', 'email', 'encodings', 'enum', 'errno', 'faulthandler',
    'fcntl', 'filecmp', 'fileinput', 'fnmatch', 'fractions', 'ftplib',
    'functools', 'gc', 'getopt', 'getpass', 'gettext', 'glob', 'grp',
    'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'idlelib',
    'imaplib', 'imghdr', 'imp', 'importlib', 'inspect', 'io',
    'ipaddress', 'itertools', 'json', 'keyword', 'lib2to3',
    'linecache', 'locale', 'logging', 'lzma', 'mailbox', 'mailcap',
    'marshal', 'math', 'mimetypes', 'mmap', 'modulefinder',
    'multiprocessing', 'netrc', 'numbers', 'operator', 'optparse',
    'os', 'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes',
    'pkgutil', 'platform', 'plistlib', 'poplib', 'posix', 'posixpath',
    'pprint', 'profile', 'pstats', 'pty', 'pwd', 'py_compile',
    'pyclbr', 'pydoc', 'queue', 'quopri', 'random', 're', 'readline',
    'reprlib', 'resource', 'rlcompleter', 'runpy', 'sched', 'secrets',
    'select', 'selectors', 'shelve', 'shlex', 'shutil', 'signal',
    'site', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'sqlite3',
    'ssl', 'stat', 'statistics', 'string', 'stringprep', 'struct',
    'subprocess', 'sunau', 'symtable', 'sys', 'sysconfig', 'syslog',
    'tabnanny', 'tarfile', 'tempfile', 'termios', 'textwrap',
    'threading', 'time', 'timeit', 'tkinter', 'token', 'tokenize',
    'trace', 'traceback', 'tracemalloc', 'tty', 'turtle', 'types',
    'typing', 'unicodedata', 'unittest', 'urllib', 'uu', 'uuid',
    'venv', 'warnings', 'wave', 'weakref', 'webbrowser', 'wsgiref',
    'xml', 'xmlrpc', 'zipapp', 'zipfile', 'zipimport', 'zlib',
}

# Third-party packages to skip when detecting source imports
_SKIP_PACKAGES = {
    'pytest', 'mock', 'unittest', 'flask', 'django', 'fastapi',
    'starlette', 'werkzeug', 'sqlalchemy', 'requests', 'httpx',
    'pydantic', 'marshmallow', 'celery', 'redis', 'pymongo',
    'boto3', 'botocore', 'numpy', 'pandas', 'conftest', 'jinja2',
    'click', 'gunicorn', 'uvicorn', 'alembic', 'wtforms',
    'flask_sqlalchemy', 'flask_login', 'flask_wtf', 'flask_cors',
    'rest_framework', 'django_filters', 'corsheaders',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_external(module: str) -> bool:
    """Return True if *module* is stdlib or a known third-party package."""
    top = module.split('.')[0]
    return top in _STDLIB or top in _SKIP_PACKAGES


def _camel(stem: str) -> str:
    """snake_case -> CamelCase."""
    return ''.join(w.capitalize() for w in stem.replace('-', '_').split('_'))


def _get_test_kind(filename: str) -> str:
    """Extract kind (unit / integ / e2e) from a timestamped filename."""
    m = re.match(r'^test_(unit|integ|e2e)_', filename)
    return m.group(1) if m else 'unit'


def _node_start(node: ast.AST) -> int:
    """Return the first line of *node*, including decorators."""
    if hasattr(node, 'decorator_list') and node.decorator_list:
        return min(d.lineno for d in node.decorator_list)
    return node.lineno


# ---------------------------------------------------------------------------
# Source-module scanner
# ---------------------------------------------------------------------------

def scan_source_modules(target_root: str) -> Dict[str, str]:
    """Build {module_name: stem} for every .py file in the target repo.

    Both the dotted module path *and* the bare stem are registered so that
    ``from app import X`` and ``from mypackage.app import X`` both resolve
    to stem ``app``.
    """
    root = pathlib.Path(target_root)
    modules: Dict[str, str] = {}

    skip_dirs = {
        'venv', '.venv', 'site-packages', 'node_modules',
        '__pycache__', '.git', 'dist', 'build', '.tox', '.eggs',
    }

    for py_file in root.rglob('*.py'):
        parts_set = set(py_file.parts)
        if parts_set & skip_dirs:
            continue
        if py_file.name.startswith('test_') or 'tests' in py_file.parts:
            continue
        if py_file.name == '__init__.py':
            continue

        stem = py_file.stem
        rel = py_file.relative_to(root)
        dotted = '.'.join(list(rel.parts[:-1]) + [stem])

        modules[dotted] = stem
        modules[stem] = stem          # bare-stem shortcut

    return modules


# ---------------------------------------------------------------------------
# Import-based source detection
# ---------------------------------------------------------------------------

def detect_source(tree: ast.AST, source_modules: Dict[str, str]) -> Optional[str]:
    """Return the source-file stem that *tree* most likely tests, or None."""
    counts: Dict[str, int] = defaultdict(int)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            if _is_external(mod):
                continue
            top = mod.split('.')[0]
            for candidate in (mod, top):
                if candidate in source_modules:
                    counts[source_modules[candidate]] += (
                        len(node.names) if node.names else 1
                    )
                    break
        elif isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                if _is_external(mod):
                    continue
                top = mod.split('.')[0]
                if top in source_modules:
                    counts[source_modules[top]] += 1

    return max(counts, key=counts.get) if counts else None


# ---------------------------------------------------------------------------
# AST extraction
# ---------------------------------------------------------------------------

def _extract_node_source(lines: List[str], node: ast.AST) -> str:
    """Extract the full source of *node* (including decorators) from *lines*."""
    start = _node_start(node) - 1          # 0-based
    end = getattr(node, 'end_lineno', node.lineno)
    return ''.join(lines[start:end])


def _is_fixture(node: ast.FunctionDef) -> bool:
    """Return True if *node* is decorated with @pytest.fixture (any form)."""
    for d in node.decorator_list:
        if isinstance(d, ast.Name) and d.id == 'fixture':
            return True
        if isinstance(d, ast.Attribute) and d.attr == 'fixture':
            return True
        if isinstance(d, ast.Call):
            func = d.func
            if isinstance(func, ast.Name) and func.id == 'fixture':
                return True
            if isinstance(func, ast.Attribute) and func.attr == 'fixture':
                return True
    return False


def extract_items(source_code: str) -> Dict[str, Any]:
    """Parse *source_code* and return categorized top-level items.

    Keys returned:
        tree          – the parsed AST (or None on syntax error)
        imports       – list of import-line strings
        setup         – sys.path manipulation / top-level assignments
        fixtures      – fixture function source strings
        test_funcs    – [(name, source_with_decorators), ...]
        test_classes  – [(class_name, [(method_name, method_source), ...]), ...]
    """
    empty: Dict[str, Any] = dict(
        tree=None, imports=[], setup=[], fixtures=[],
        test_funcs=[], test_classes=[],
    )
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return empty

    lines = source_code.splitlines(keepends=True)
    imports: List[str] = []
    setup: List[str] = []
    fixtures: List[str] = []
    test_funcs: List[Tuple[str, str]] = []
    test_classes: List[Tuple[str, List[Tuple[str, str]]]] = []

    for node in ast.iter_child_nodes(tree):
        src = _extract_node_source(lines, node)

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(src.rstrip())

        elif isinstance(node, ast.FunctionDef):
            if _is_fixture(node):
                fixtures.append(src)
            elif node.name.startswith('test_'):
                test_funcs.append((node.name, src))
            else:
                setup.append(src)

        elif isinstance(node, ast.ClassDef):
            methods: List[Tuple[str, str]] = []
            class_lines = source_code.splitlines(keepends=True)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.FunctionDef) and child.name.startswith('test_'):
                    m_start = _node_start(child) - 1
                    m_end = getattr(child, 'end_lineno', child.lineno)
                    m_src = ''.join(class_lines[m_start:m_end])
                    methods.append((child.name, m_src))
            if methods or node.name.startswith('Test'):
                test_classes.append((node.name, methods))
            else:
                setup.append(src)

        elif isinstance(node, ast.Expr) and isinstance(
            getattr(node, 'value', None), (ast.Constant,)
        ):
            pass  # module docstring — skip

        else:
            setup.append(src.rstrip())

    return dict(
        tree=tree, imports=imports, setup=setup, fixtures=fixtures,
        test_funcs=test_funcs, test_classes=test_classes,
    )


# ---------------------------------------------------------------------------
# Method conversion helpers
# ---------------------------------------------------------------------------

_DEF_RE = re.compile(r'(def\s+\w+\s*\()([^)]*)\)')


def _ensure_self(func_source: str) -> str:
    """Make sure the first parameter is ``self``.

    Works for both standalone functions (no ``self``) and methods that
    already have it.
    """
    m = _DEF_RE.search(func_source)
    if not m:
        return func_source
    params = m.group(2).strip()
    if params.startswith('self'):
        return func_source                          # already a method
    new_params = f'self, {params}' if params else 'self'
    return func_source[:m.start(2)] + new_params + func_source[m.end(2):]


def _indent_block(text: str, spaces: int = 4) -> str:
    """Indent every non-empty line by *spaces* spaces."""
    pad = ' ' * spaces
    out: List[str] = []
    for line in text.splitlines(keepends=True):
        if line.strip():
            out.append(pad + line)
        else:
            out.append(line)
    return ''.join(out)


def _dedent_method(method_source: str) -> str:
    """Remove the existing class-level indentation from a method."""
    return textwrap.dedent(method_source)


# ---------------------------------------------------------------------------
# Build materialized file
# ---------------------------------------------------------------------------

def _dedup(tests: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Drop duplicate test names, keeping the first occurrence."""
    seen: Set[str] = set()
    out: List[Tuple[str, str]] = []
    for name, src in tests:
        if name not in seen:
            seen.add(name)
            out.append((name, src))
    return out


def build_materialized(
    source_stem: str,
    unit: List[Tuple[str, str]],
    integ: List[Tuple[str, str]],
    e2e: List[Tuple[str, str]],
    imports: List[str],
    setup: List[str],
    fixtures: List[str],
) -> str:
    """Return the contents of the final ``test_<source_stem>.py`` file."""
    cls = _camel(source_stem)
    unit = _dedup(unit)
    integ = _dedup(integ)
    e2e = _dedup(e2e)

    # --- de-duplicate imports & setup ----
    seen_imp: Set[str] = set()
    unique_imports: List[str] = []
    for imp in imports:
        normalized = imp.strip()
        if normalized and normalized not in seen_imp:
            # keep sys.path lines in setup, not imports
            if 'sys.path' in normalized:
                continue
            seen_imp.add(normalized)
            unique_imports.append(normalized)

    seen_setup: Set[str] = set()
    unique_setup: List[str] = []
    for s in setup:
        normalized = s.strip()
        if normalized and normalized not in seen_setup:
            seen_setup.add(normalized)
            unique_setup.append(normalized)

    seen_fix: Set[str] = set()
    unique_fixtures: List[str] = []
    for f in fixtures:
        normalized = f.strip()
        if normalized and normalized not in seen_fix:
            seen_fix.add(normalized)
            unique_fixtures.append(f)

    # --- assemble ----
    parts: List[str] = []
    parts.append(f'"""Tests for {source_stem} module."""\n')

    if unique_imports:
        parts.append('\n'.join(unique_imports))
        parts.append('')

    if unique_setup:
        parts.append('\n'.join(unique_setup))
        parts.append('')

    if unique_fixtures:
        for fix in unique_fixtures:
            parts.append(fix.rstrip())
            parts.append('')

    def _write_class(label: str, kind_label: str, tests: List[Tuple[str, str]],
                     marker: Optional[str] = None):
        if not tests:
            return
        parts.append('')
        if marker:
            parts.append(marker)
        parts.append(f'class Test{cls}{label}:')
        parts.append(f'    """{kind_label} tests for {source_stem}."""\n')
        for _name, src in tests:
            cleaned = _dedent_method(src)
            method = _ensure_self(cleaned)
            parts.append(_indent_block(method.rstrip()))
            parts.append('')

    _write_class('Unit', 'Unit', unit)
    _write_class('Integration', 'Integration', integ)
    _write_class('E2E', 'End-to-end', e2e, marker='@pytest.mark.e2e')

    return '\n'.join(parts) + '\n'


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def materialize(generated_dir: str, target_root: str) -> List[str]:
    """Transform timestamped test files into source-file-mapped test files.

    Returns the list of newly created materialized file paths.
    """
    gen = pathlib.Path(generated_dir)
    if not gen.exists():
        print('Materialization: no generated tests directory', file=sys.stderr)
        return []

    source_modules = scan_source_modules(target_root)
    print(f'Materialization: {len(source_modules)} source modules detected')

    timestamped = sorted(
        f for f in gen.glob('test_*.py') if GENERATED_RE.match(f.name)
    )
    if not timestamped:
        print('Materialization: no timestamped files found — nothing to do')
        return []

    print(f'Materializing {len(timestamped)} generated test files …')

    # Accumulate per-source-stem data
    GroupT = Dict[str, Any]
    groups: Dict[str, GroupT] = defaultdict(lambda: {
        'unit': [], 'integ': [], 'e2e': [],
        'imports': [], 'setup': [], 'fixtures': [],
    })

    file_map: Dict[str, str] = {}  # old_filename -> new_filename

    for tf in timestamped:
        kind = _get_test_kind(tf.name)
        try:
            code = tf.read_text(encoding='utf-8')
        except Exception as exc:
            print(f'  Warning: cannot read {tf.name}: {exc}', file=sys.stderr)
            continue

        items = extract_items(code)
        if items['tree'] is None:
            print(f'  Warning: syntax error in {tf.name}, skipping',
                  file=sys.stderr)
            continue

        stem = detect_source(items['tree'], source_modules) or 'misc'
        g = groups[stem]
        g['imports'].extend(items['imports'])
        g['setup'].extend(items['setup'])
        g['fixtures'].extend(items['fixtures'])

        # Standalone test functions
        for name, src in items['test_funcs']:
            g[kind].append((name, src))

        # Methods from test classes
        for _cls_name, methods in items['test_classes']:
            for name, src in methods:
                g[kind].append((name, src))

        file_map[tf.name] = f'test_{stem}.py'
        print(f'  {tf.name} -> test_{stem}.py ({kind})')

    # Write materialized files
    created: List[str] = []
    for stem, g in sorted(groups.items()):
        total = len(g['unit']) + len(g['integ']) + len(g['e2e'])
        if total == 0:
            continue

        content = build_materialized(
            stem, g['unit'], g['integ'], g['e2e'],
            g['imports'], g['setup'], g['fixtures'],
        )

        out = gen / f'test_{stem}.py'

        # Syntax-check before writing
        try:
            ast.parse(content)
        except SyntaxError as exc:
            print(f'  Warning: test_{stem}.py has syntax issues: {exc}',
                  file=sys.stderr)
            content = (
                f'# WARNING: materialization produced a syntax error — review needed\n'
                f'# Error: {exc}\n\n'
                + content
            )

        out.write_text(content, encoding='utf-8')
        created.append(str(out))
        print(f'  -> test_{stem}.py: '
              f'{len(_dedup(g["unit"]))} unit, '
              f'{len(_dedup(g["integ"]))} integ, '
              f'{len(_dedup(g["e2e"]))} e2e')

    # Remove old timestamped files
    removed = 0
    for tf in timestamped:
        try:
            tf.unlink()
            removed += 1
        except Exception as exc:
            print(f'  Warning: cannot remove {tf.name}: {exc}',
                  file=sys.stderr)

    # Write mapping file for the dashboard
    map_path = gen / '_materialization_map.json'
    map_data = {
        'file_map': file_map,
        'source_stems': sorted(groups.keys()),
        'stats': {
            'input_files': len(timestamped),
            'output_files': len(created),
            'removed': removed,
        },
    }
    map_path.write_text(json.dumps(map_data, indent=2), encoding='utf-8')

    print(f'\nMaterialization complete:')
    print(f'  Input:   {len(timestamped)} timestamped files')
    print(f'  Output:  {len(created)} source-mapped files')
    print(f'  Removed: {removed} old files')

    return created


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: materialize.py <generated_dir> <target_root>',
              file=sys.stderr)
        sys.exit(1)

    result = materialize(sys.argv[1], sys.argv[2])
    sys.exit(0 if result else 1)
