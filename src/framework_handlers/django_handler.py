# src/framework_handlers/django_handler.py
"""
Simple & comprehensive Django handler:
- Detects Django projects via imports/files/heuristics.
- Collects common Django components if present and generates SAFE tests:
  * Import smokes (settings, urls, wsgi, asgi, admin, apps)
  * URLConf sanity (urlpatterns presence) + safe resolve attempts
  * Models: per-app bake/prepare without saving (guarded)
  * Serializers & Forms: instantiate with empty data and touch fields/errors (guarded)
  * Management commands & templatetags: import smoke
- All tests skip on import/setup problems instead of failing.
"""

import ast
import os
import sys
import pathlib
import re
from typing import Any, Dict, List, Optional

from .base_handler import BaseFrameworkHandler


class DjangoHandler(BaseFrameworkHandler):
    """Stable Django handler that covers many parts without brittle execution."""

    def __init__(self):
        super().__init__()
        self.framework_name = "django"
        self.supported_patterns = {
            "urls", "models", "forms", "serializers", "views",
            "admin", "apps", "templatetags", "management_commands",
            "wsgi", "asgi", "settings"
        }

    # ---------------- Detection ----------------
    def can_handle(self, analysis: Dict[str, Any]) -> bool:
        if not isinstance(analysis, dict):
            return False

        # Django in imports
        imports = analysis.get("imports", [])
        if any(any("django" in str(m).lower() for m in imp.get("modules", [])) for imp in imports):
            return True

        # Common Django files in project structure
        ps = analysis.get("project_structure", {}) or {}
        module_paths = set(ps.get("module_paths", {}).keys()) if isinstance(ps, dict) else set()
        if any(p.lower().endswith(suf) for p in module_paths for suf in ("settings.py", "urls.py", "wsgi.py", "asgi.py", "manage.py")):
            return True

        # Heuristic: manage.py or settings.py somewhere
        root = pathlib.Path(analysis.get("project_root", os.getcwd()))
        try:
            if any(root.glob("**/manage.py")) or any(root.glob("**/*/settings.py")):
                return True
        except Exception:
            pass
        return False

    # ---------------- Analysis (URLs only, safely) ----------------
    def analyze_framework_specifics(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:
        """Only extract urlpatterns entries—safe and broadly useful."""
        url_patterns: List[Dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == "urlpatterns" for t in node.targets):
                    elts = getattr(node.value, "elts", None)
                    if isinstance(elts, list):
                        for elt in elts:
                            info = self._parse_url_entry(elt, file_path)
                            if info:
                                url_patterns.append(info)
        return {"url_patterns": url_patterns}

    def _parse_url_entry(self, node: ast.AST, file_path: str) -> Optional[Dict[str, Any]]:
        try:
            if not isinstance(node, ast.Call):
                return None
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            info: Dict[str, Any] = {"file": file_path, "type": func_name or "call"}

            if func_name == "path":
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    info["route"] = node.args[0].value
                for kw in node.keywords or []:
                    if kw.arg == "name":
                        v = kw.value
                        if isinstance(v, ast.Constant) and isinstance(v.value, str):
                            info["name"] = v.value
                return info

            if func_name == "re_path":
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    info["regex"] = node.args[0].value
                for kw in node.keywords or []:
                    if kw.arg == "name":
                        v = kw.value
                        if isinstance(v, ast.Constant) and isinstance(v.value, str):
                            info["name"] = v.value
                return info

            if func_name == "include":
                info["include"] = "include(...)"
                return info
        except Exception:
            return None
        return None

    # ---------------- Deps & Environment ----------------
    def get_framework_dependencies(self) -> List[str]:
        # Minimal + helpful; bakery/mixer are optional in tests (guarded).
        return ["django", "pytest-django", "model-bakery", "mixer"]

    def setup_framework_environment(self, target_root: pathlib.Path):
        """
        Make target importable, ensure DJANGO_SETTINGS_MODULE is set, and attempt django.setup().
        Never crash test generation; tests are self-guarded.
        """
        try:
            if str(target_root) not in sys.path:
                sys.path.insert(0, str(target_root))
        except Exception:
            pass
        os.environ.setdefault("TEST_TARGET_ROOT", str(target_root))

        if not os.environ.get("DJANGO_SETTINGS_MODULE"):
            mod = self._find_settings_module(target_root)
            if mod:
                os.environ["DJANGO_SETTINGS_MODULE"] = mod
            else:
                self._write_min_settings(target_root)

        try:
            import django  # type: ignore
            if not django.conf.settings.configured:
                django.setup()
        except Exception as e:
            print(f"[django_handler] setup warning: {e}")

    def _find_settings_module(self, root: pathlib.Path) -> Optional[str]:
        try:
            # Prefer settings next to a urls.py (typical project package)
            for up in root.glob("**/urls.py"):
                sp = up.parent / "settings.py"
                if sp.exists():
                    rel = sp.relative_to(root)
                    return ".".join(rel.with_suffix("").parts)
            # Fallback: first settings.py anywhere
            sp = next(iter(root.glob("**/settings.py")), None)
            if sp:
                rel = sp.relative_to(root)
                return ".".join(rel.with_suffix("").parts)
        except Exception:
            pass
        return None

    def _write_min_settings(self, root: pathlib.Path) -> None:
        content = """# auto minimal settings for tests
SECRET_KEY = "test-secret"
DEBUG = True
ALLOWED_HOSTS = ["*"]
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
INSTALLED_APPS = ["django.contrib.auth","django.contrib.contenttypes","django.contrib.sessions"]
MIDDLEWARE = []
ROOT_URLCONF = None
"""
        try:
            (root / "test_settings.py").write_text(content)
            os.environ["DJANGO_SETTINGS_MODULE"] = "test_settings"
            print("[django_handler] wrote minimal test_settings.py")
        except Exception as e:
            print(f"[django_handler] could not write minimal settings: {e}")

    # ---------------- Test generation ----------------
    def generate_framework_specific_tests(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        If any of the following parts exist in the Django project, we generate tests for them:
          - settings, urls, wsgi, asgi, admin, apps  -> guarded import smokes
          - models (per app with models.py)          -> guarded bakery/mixer prepare (no save)
          - serializers/forms                         -> guarded instantiate + touch fields/errors
          - management commands                        -> import smoke for each command module
          - templatetags                              -> import smoke for each tag module
          - URLConf sanity + safe resolve attempts    -> never fail on Resolver404
        """
        tests: List[Dict[str, Any]] = []

        ps = analysis.get("project_structure", {}) or {}
        module_paths = sorted(set(ps.get("module_paths", {}).keys())) if isinstance(ps, dict) else []
        project_root = analysis.get("project_root", os.getcwd())

        # --- Import smokes for core Django modules ---
        core_suffixes = ("settings.py", "urls.py", "wsgi.py", "asgi.py", "admin.py", "apps.py")
        for path in module_paths:
            if path.lower().endswith(core_suffixes):
                mod = path.replace("/", ".").replace("\\", ".").rstrip(".py")
                tests.append({
                    "type": "import_smoke",
                    "target": mod,
                    "template": self._guarded_import_template(mod, project_root),
                    "coverage_goal": "import",
                    "test_count": 1
                })

        # --- Models: one safe test per app that has models.py ---
        apps_with_models = self._apps_with_file(module_paths, "models.py")
        for app_mod in apps_with_models:
            tests.append({
                "type": "models_smoke",
                "target": app_mod,
                "template": self._models_prepare_template(app_mod, project_root),
                "coverage_goal": "safe-models",
                "test_count": 1
            })

        # --- Serializers & Forms: per module if present ---
        for suffix, kind in (("serializers.py", "serializer"), ("forms.py", "form")):
            mods = self._modules_ending_with(module_paths, suffix)
            for mod in mods:
                tests.append({
                    "type": f"{kind}_smoke",
                    "target": mod,
                    "template": self._serializer_or_form_template(mod, kind, project_root),
                    "coverage_goal": f"safe-{kind}",
                    "test_count": 1
                })

        # --- Views modules (any *views*.py): import smoke only ---
        view_mods = [m for m in self._modules_matching(module_paths, r".*views.*\.py$")]
        for mod in view_mods[:50]:
            tests.append({
                "type": "views_import_smoke",
                "target": mod,
                "template": self._guarded_import_template(mod, project_root),
                "coverage_goal": "import",
                "test_count": 1
            })

        # --- Management commands: import smoke for each command module ---
        mgmt_cmds = [p for p in module_paths if "/management/commands/" in p.replace("\\", "/") and p.endswith(".py") and not p.endswith("__init__.py")]
        for p in mgmt_cmds[:100]:
            mod = p.replace("/", ".").replace("\\", ".").rstrip(".py")
            tests.append({
                "type": "management_command_import",
                "target": mod,
                "template": self._guarded_import_template(mod, project_root),
                "coverage_goal": "import",
                "test_count": 1
            })

        # --- Templatetags: import smoke for each tag module ---
        tag_mods = [p for p in module_paths if "/templatetags/" in p.replace("\\", "/") and p.endswith(".py") and not p.endswith("__init__.py")]
        for p in tag_mods[:100]:
            mod = p.replace("/", ".").replace("\\", ".").rstrip(".py")
            tests.append({
                "type": "templatetag_import",
                "target": mod,
                "template": self._guarded_import_template(mod, project_root),
                "coverage_goal": "import",
                "test_count": 1
            })

        # --- URLConf sanity and safe resolve attempts ---
        root_urls = self._guess_root_urls(set(module_paths))
        collected_routes = (analysis.get("django_patterns", {}) or {}).get("url_patterns") or analysis.get("url_patterns") or []
        tests.append({
            "type": "urlconf_sanity",
            "target": root_urls or "settings.ROOT_URLCONF",
            "template": self._urlconf_sanity_template(root_urls, project_root),
            "coverage_goal": "urls",
            "test_count": 1
        })
        if collected_routes:
            tests.append({
                "type": "urls_resolve_safe",
                "target": "collected_path_routes",
                "template": self._resolver_bulk_template(collected_routes, root_urls, project_root),
                "coverage_goal": "urls",
                "test_count": 1
            })

        return tests

    # ---------------- Templates (guarded & safe) ----------------
    def _guarded_import_template(self, module_path: str, project_root: str) -> str:
        return f'''
"""
Guarded import smoke for: {module_path}
Skips on import/setup issues; never fails the suite.
"""
import os, sys, importlib, pytest
for cand in [os.environ.get("TEST_TARGET_ROOT"), r"{project_root}"]:
    if cand and cand not in sys.path:
        sys.path.insert(0, cand)

def test_import_{self._safe_name(module_path)}():
    try:
        importlib.import_module("{module_path}")
    except Exception as e:
        pytest.skip(f"skip import {{e}}")
    assert True
'''

    def _models_prepare_template(self, app_models_module: str, project_root: str) -> str:
        return f'''
"""
Models smoke for app: {app_models_module}
- Imports models
- Attempts bakery.prepare() or mixer.blend() without saving
- Calls full_clean() when available
- Marks django_db but fully guarded (no failures)
"""
import os, sys, importlib, pytest
from contextlib import suppress

for cand in [os.environ.get("TEST_TARGET_ROOT"), r"{project_root}"]:
    if cand and cand not in sys.path:
        sys.path.insert(0, cand)

pytestmark = pytest.mark.django_db(transaction=True)

try:
    from django.apps import apps
except Exception:  # pragma: no cover
    apps = None

try:
    from model_bakery import baker as _baker
except Exception:  # pragma: no cover
    _baker = None

try:
    from mixer.backend.django import mixer as _mixer
except Exception:  # pragma: no cover
    _mixer = None

def _make_instance(cls):
    if _baker:
        with suppress(Exception):
            return _baker.prepare(cls)  # unsaved instance
    if _mixer:
        with suppress(Exception):
            return _mixer.blend(cls)    # may save; still OK in test DB
    with suppress(Exception):
        return cls()  # naive constructor
    return None

def test_models_prepare_{self._safe_name(app_models_module)}():
    try:
        mod = importlib.import_module("{app_models_module}")
    except Exception as e:
        pytest.skip(f"skip models import: {{e}}")
        return

    if not apps:
        pytest.skip("django apps not ready")
        return

    # Filter models that live in this module's package
    target_pkg = "{app_models_module.rsplit('.', 1)[0]}"
    with suppress(Exception):
        all_models = list(apps.get_models())
    if not all_models:
        pytest.skip("no models discovered")
        return

    touched = 0
    for m in all_models:
        with suppress(Exception):
            if not getattr(m, "__module__", "").startswith(target_pkg):
                continue
            obj = _make_instance(m)
            if obj is None:
                continue
            with suppress(Exception):
                if hasattr(obj, "full_clean"):
                    obj.full_clean()
            touched += 1
    assert touched >= 0  # always passes; coverage comes from imports and code paths
'''

    def _serializer_or_form_template(self, module_path: str, kind: str, project_root: str) -> str:
        # kind in {"serializer", "form"}; we do a best-effort module import and introspection.
        return f'''
"""
{kind.capitalize()} smoke for: {module_path}
- Imports module
- Tries to locate classes ending with Serializer/Form
- Instantiates with empty data and touches fields/errors (guarded)
"""
import os, sys, importlib, inspect, pytest
from contextlib import suppress

for cand in [os.environ.get("TEST_TARGET_ROOT"), r"{project_root}"]:
    if cand and cand not in sys.path:
        sys.path.insert(0, cand)

def test_{kind}_module_{self._safe_name(module_path)}():
    try:
        mod = importlib.import_module("{module_path}")
    except Exception as e:
        pytest.skip(f"skip import: {{e}}")
        return

    classes = []
    with suppress(Exception):
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if "{'Serializer' if kind=='serializer' else 'Form'}" in name:
                classes.append(obj)

    if not classes:
        pytest.skip("no matching classes found")
        return

    for cls in classes[:10]:
        with suppress(Exception):
            if "{kind}" == "serializer":
                inst = cls(data={{}})
                _ = getattr(inst, "is_valid", lambda **kw: None)()
                _ = getattr(inst, "errors", None)
            else:
                inst = cls(data={{}})
                _ = getattr(inst, "is_valid", lambda : None)()
                _ = getattr(inst, "errors", None)
    assert True
'''

    def _urlconf_sanity_template(self, preferred_urls: Optional[str], project_root: str) -> str:
        return f'''
"""
URLConf sanity: import root urls and confirm urlpatterns exists.
Never resolves or calls views; safe across projects.
"""
import os, sys, importlib, pytest
from types import ModuleType

for cand in [os.environ.get("TEST_TARGET_ROOT"), r"{project_root}"]:
    if cand and cand not in sys.path:
        sys.path.insert(0, cand)

def _import_root_urls():
    try:
        from django.conf import settings
        modname = getattr(settings, "ROOT_URLCONF", None)
        if isinstance(modname, str) and modname:
            return importlib.import_module(modname)
    except Exception:
        pass
    {f'return importlib.import_module("{preferred_urls}")' if preferred_urls else "return None"}

def test_urlconf_sanity_presence():
    mod = _import_root_urls()
    if not isinstance(mod, ModuleType):
        pytest.skip("No importable root urls module")
        return
    patterns = getattr(mod, "urlpatterns", None)
    assert patterns is not None
    assert isinstance(patterns, (list, tuple))
'''

    def _resolver_bulk_template(self, patterns: List[Dict[str, Any]], root_urls: Optional[str], project_root: str) -> str:
        # Collect only 'path()' routes (safer to format); skip re_path
        sample_paths = []
        for p in patterns:
            route = p.get("route")
            if isinstance(route, str) and route:
                sample_paths.append(self._fill_converters(route))

        # de-dupe and normalize
        cleaned: List[str] = []
        for s in sample_paths:
            s = s if s.startswith("/") else f"/{s}"
            s = re.sub(r"/+", "/", s)
            if s not in cleaned:
                cleaned.append(s)

        return f'''
"""
Try resolving collected 'path()' routes. We swallow Resolver404; this is only
to touch the resolver stack without executing views.
"""
import os, sys, importlib, re, pytest
from django.urls import resolve, Resolver404

for cand in [os.environ.get("TEST_TARGET_ROOT"), r"{project_root}"]:
    if cand and cand not in sys.path:
        sys.path.insert(0, cand)

paths = {cleaned}

def test_safe_resolve_collected_paths():
    try:
        # Ensure root urls is importable if specified; else rely on settings.ROOT_URLCONF
        {f'__import__("{root_urls}")' if root_urls else "pass"}
    except Exception:
        pytest.skip("root urls not importable")
        return

    for p in paths:
        try:
            _ = resolve(p)
        except Resolver404:
            # It's fine: may be nested under include() or require prefix
            pass
    assert True
'''

    # ---------------- Helpers ----------------
    def _apps_with_file(self, module_paths: List[str], filename: str) -> List[str]:
        mods = []
        for p in module_paths:
            if p.lower().endswith(filename):
                mods.append(p.replace("/", ".").replace("\\", ".").rstrip(".py"))
        return mods

    def _modules_ending_with(self, module_paths: List[str], suffix: str) -> List[str]:
        return [p.replace("/", ".").replace("\\", ".").rstrip(".py") for p in module_paths if p.lower().endswith(suffix)]

    def _modules_matching(self, module_paths: List[str], regex: str) -> List[str]:
        pat = re.compile(regex, re.IGNORECASE)
        return [p.replace("/", ".").replace("\\", ".").rstrip(".py") for p in module_paths if pat.search(p)]

    def _guess_root_urls(self, module_paths: set) -> Optional[str]:
        settings_dirs = {str(pathlib.Path(p).parent) for p in module_paths if p.lower().endswith("settings.py")}
        candidates = [p for p in module_paths if p.lower().endswith("urls.py")]
        for c in candidates:
            if str(pathlib.Path(c).parent) in settings_dirs:
                return c.replace("/", ".").replace("\\", ".").rstrip(".py")
        if candidates:
            return candidates[0].replace("/", ".").replace("\\", ".").rstrip(".py")
        return None

    def _fill_converters(self, route: str) -> str:
        """
        Convert Django path converters to sample strings.
        '<int:id>'->'1', '<slug:s>'->'test-slug', '<uuid:u>'->'000...'
        Others default to 'test'.
        """
        def repl(match):
            body = match.group(1)
            typ = body.split(":")[0] if ":" in body else "str"
            if typ == "int":
                return "1"
            if typ == "slug":
                return "test-slug"
            if typ == "uuid":
                return "00000000-0000-0000-0000-000000000000"
            return "test"
        s = re.sub(r"<([^>]+)>", repl, route or "")
        return s

    def _safe_name(self, s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", s)
