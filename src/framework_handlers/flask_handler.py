# src/framework_handlers/flask_handler.py
"""
Flask-specific framework handler for analysis and test generation.
"""

import ast
import pathlib
from typing import Any, Dict, List

from .base_handler import BaseFrameworkHandler


class FlaskHandler(BaseFrameworkHandler):
    """Flask framework handler."""
    
    def __init__(self):
        super().__init__()
        self.framework_name = "flask"
        self.supported_patterns = {"routes", "blueprints", "extensions"}
    
    def can_handle(self, analysis: Dict[str, Any]) -> bool:
        """Check if project uses Flask based on analysis."""
        # Check for Flask routes
        routes = analysis.get("routes", [])
        flask_routes = [route for route in routes if route.get("framework") == "flask"]
        if len(flask_routes) > 0:
            return True
        
        # Check imports for Flask
        imports = analysis.get("imports", [])
        flask_imports = any(
            any("flask" in str(module).lower() for module in imp.get("modules", []))
            for imp in imports
        )
        
        return flask_imports
    
    def analyze_framework_specifics(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:
        """Analyze Flask-specific patterns in the AST."""
        flask_routes = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    route_info = self._extract_flask_route_info(decorator)
                    if route_info and route_info.get("path"):
                        route_info.update({
                            "handler": node.name,
                            "file": file_path,
                            "framework": "flask"
                        })
                        flask_routes.append(route_info)
        
        return {"flask_routes": flask_routes}
    
    def _extract_flask_route_info(self, dec) -> Dict[str, Any]:
        """Extract route information from Flask decorators."""
        info = {}
        try:
            # Handle @app.route("/path") style (Flask)
            if isinstance(dec, ast.Call):
                if (isinstance(dec.func, ast.Attribute) and 
                    dec.func.attr in {"route", "as_view", "add_url_rule"}):
                    info["is_view"] = True
                    if dec.args:
                        arg0 = dec.args[0]
                        if isinstance(arg0, ast.Constant):
                            info["path"] = arg0.value
            
            # Handle @api.route() style
            if (isinstance(dec, ast.Call) and
                isinstance(dec.func, ast.Attribute) and
                'route' in dec.func.attr.lower()):
                info["is_view"] = True
                if dec.args:
                    arg0 = dec.args[0]
                    if isinstance(arg0, ast.Constant):
                        info["path"] = arg0.value
                        
        except Exception:
            pass
        return info
    
    def get_framework_dependencies(self) -> List[str]:
        """Get Flask-specific dependencies."""
        return [
            "flask",
            "flask-sqlalchemy",
            "flask-login",
            "flask-wtf",
            "flask-bcrypt",
            "flask-testing"
        ]
    
    def detect_framework_patterns(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect Flask-specific patterns in the analysis."""
        flask_routes = [route for route in analysis.get("routes", []) if route.get("framework") == "flask"]
        
        flask_specific = {
            "framework": "flask",
            "route_count": len(flask_routes),
            "has_blueprints": any("blueprint" in str(route.get("path", "")).lower() for route in flask_routes),
        }
        return flask_specific


# ----------------------- APPENDED ENHANCEMENTS BELOW (no deletions) -----------------------

class FlaskHandler(FlaskHandler):  # type: ignore[misc]
    """
    Extended Flask handler with realistic detection (no scoring), deeper AST analysis,
    and pragmatic test templates that import and exercise the real application.
    """

    # -------- Detection upgrades (realistic) ---------------------------------
    def can_handle(self, analysis: Dict[str, Any]) -> bool:  # type: ignore[override]
        """
        Enhanced detection checks:
        - explicit flask imports (flask, flask_* extensions)
        - presence of app factory (def create_app)
        - app = Flask(__name__) or Blueprint(...) sightings
        - @app.route / blueprint.route decorators OR add_url_rule() calls
        - typical entry modules: app.py / wsgi.py
        """
        # 1) Imports
        imports = analysis.get("imports", []) or []
        has_flask_imports = any(
            any("flask" in str(m).lower() for m in imp.get("modules", []))
            for imp in imports
        )
        if has_flask_imports:
            return True

        # 2) Structural hints
        ps = analysis.get("project_structure", {}) or {}
        module_paths = set(ps.get("module_paths", {}).keys()) if isinstance(ps, dict) else set()
        if any(p.lower().endswith(("app.py", "wsgi.py")) for p in module_paths):
            return True

        # 3) AST-derived hints from upstream analyzers (if present)
        if analysis.get("flask_app_inits") or analysis.get("flask_blueprints"):
            return True

        # 4) Route evidence (from generic route collector, if any)
        routes = analysis.get("routes", []) or []
        if any(r.get("framework") == "flask" for r in routes):
            return True

        return False

    # -------- Analysis upgrades ----------------------------------------------
    def analyze_framework_specifics(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:  # type: ignore[override]
        """
        Discover:
          - @app.route / blueprint.route decorators (path, methods, endpoint)
          - app.add_url_rule(...)
          - Blueprint(...) declarations & app.register_blueprint(...)
          - MethodView.as_view(...) endpoints
          - before_request / after_request / teardown_request handlers
          - errorhandler(...) registrations
          - create_app() factories
        """
        routes: List[Dict[str, Any]] = []
        blueprints: List[Dict[str, Any]] = []
        blueprint_vars: List[str] = []
        register_blueprint_calls: List[Dict[str, Any]] = []
        add_url_rule_calls: List[Dict[str, Any]] = []
        methodviews: List[Dict[str, Any]] = []
        lifecycle: Dict[str, List[str]] = {
            "before_request": [],
            "after_request": [],
            "teardown_request": [],
        }
        error_handlers: List[Dict[str, Any]] = []
        app_factories: List[str] = []

        # Collect blueprint variable names and app = Flask(__name__) in Assign
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                name = self._call_name(node.value)
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if name == "Blueprint":
                    blueprint_vars.extend(targets)
                    blueprints.append({
                        "name_vars": targets,
                        "file": file_path,
                        "lineno": getattr(node, "lineno", 1),
                    })

        # Detect create_app factory
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "create_app":
                app_factories.append(node.name)

        # Route decorators and lifecycle hooks on functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # lifecycle hooks
                for dec in node.decorator_list:
                    attr = self._attr_chain(dec)
                    if attr.endswith("before_request"):
                        lifecycle["before_request"].append(node.name)
                    if attr.endswith("after_request"):
                        lifecycle["after_request"].append(node.name)
                    if attr.endswith("teardown_request"):
                        lifecycle["teardown_request"].append(node.name)

                # route decorators (app/blueprint.route)
                for dec in node.decorator_list:
                    info = self._extract_flask_route_info_extended(dec)
                    if info.get("path"):
                        info.update({
                            "handler": node.name,
                            "file": file_path,
                            "framework": "flask",
                            "lineno": getattr(node, "lineno", 1),
                            "end_lineno": getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                        })
                        routes.append(info)

        # app.add_url_rule(...) calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and self._attr_chain(node.func).endswith("add_url_rule"):
                entry = {
                    "file": file_path,
                    "lineno": getattr(node, "lineno", 1),
                    "rule": self._safe_value(node.args[0]) if node.args else None,
                    "endpoint": self._kwarg(node, "endpoint"),
                    "methods": self._kwarg(node, "methods"),
                }
                add_url_rule_calls.append(entry)
                # treat as route for test generation
                rule = entry["rule"] or "/"
                routes.append({"path": rule, "methods": entry["methods"] or ["GET"], "handler": str(entry["endpoint"] or "endpoint"), "file": file_path})

        # app.register_blueprint(...)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and self._attr_chain(node.func).endswith("register_blueprint"):
                register_blueprint_calls.append({
                    "file": file_path,
                    "lineno": getattr(node, "lineno", 1),
                    "args": [self._safe_value(a) for a in node.args],
                    "kwargs": {kw.arg: self._safe_value(kw.value) for kw in node.keywords if kw.arg},
                })

        # MethodView.as_view('endpoint')
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and self._attr_chain(node.func).endswith("as_view"):
                endpoint = self._safe_value(node.args[0]) if node.args else None
                methodviews.append({
                    "file": file_path,
                    "lineno": getattr(node, "lineno", 1),
                    "endpoint": endpoint
                })

        # errorhandler(ExceptionType) / app.register_error_handler(...)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for dec in node.decorator_list:
                    # @app.errorhandler(404) or @bp.errorhandler(Exception)
                    if isinstance(dec, ast.Call) and self._attr_chain(dec.func).endswith("errorhandler"):
                        error_handlers.append({
                            "file": file_path,
                            "lineno": getattr(node, "lineno", 1),
                            "target": self._safe_value(dec.args[0]) if dec.args else None,
                            "handler": node.name,
                        })
            if isinstance(node, ast.Call) and self._attr_chain(node.func).endswith("register_error_handler"):
                error_handlers.append({
                    "file": file_path,
                    "lineno": getattr(node, "lineno", 1),
                    "target": self._safe_value(node.args[0]) if node.args else None,
                    "handler": self._safe_value(node.args[1]) if len(node.args) > 1 else None,
                })

        return {
            "flask_routes": routes,
            "flask_blueprints": blueprints,
            "flask_blueprint_vars": blueprint_vars,
            "flask_register_blueprint": register_blueprint_calls,
            "flask_add_url_rule": add_url_rule_calls,
            "flask_methodviews": methodviews,
            "flask_lifecycle": lifecycle,
            "flask_error_handlers": error_handlers,
            "flask_app_factories": app_factories,
        }

    # Keep your basic extractor; add an extended version capturing methods & endpoint
    def _extract_flask_route_info_extended(self, dec) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        try:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                if dec.func.attr == "route":
                    # path arg
                    if dec.args:
                        a0 = dec.args[0]
                        if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                            info["path"] = a0.value
                    # keywords
                    kw = {k.arg: k.value for k in getattr(dec, "keywords", []) if k.arg}
                    # methods can be ["GET","POST"] etc.
                    if "methods" in kw and isinstance(kw["methods"], ast.List):
                        info["methods"] = [self._safe_value(e) for e in kw["methods"].elts]
                    elif "methods" in kw:
                        info["methods"] = [self._safe_value(kw["methods"])]
                    else:
                        info["methods"] = ["GET"]
                    if "endpoint" in kw:
                        info["endpoint"] = self._safe_value(kw["endpoint"])
        except Exception:
            pass
        return info

    def can_handle(self, analysis: Dict[str, Any]) -> bool:
        # Guard: analysis might not be a dict
        if not isinstance(analysis, dict):
            return False

    # -------- Test generation -------------------------------------------------
    def generate_framework_specific_tests(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Emit tests that import the *real* Flask app (app or create_app()) and issue HTTP
        calls with the Werkzeug test client. Also include blueprint smoke tests if only
        register_blueprint evidence is found.
        """
        tests: List[Dict[str, Any]] = []
        routes = analysis.get("flask_routes") or []
        app_module_guess = self._guess_app_module(analysis)

        # Per-route tests
        for r in routes[:200]:
            path = r.get("path", "/")
            methods = r.get("methods") or ["GET"]
            template = self._route_test_template(path=path, methods=methods, app_module=app_module_guess)
            tests.append({
                "type": "flask_route_test",
                "target": f"{methods[0]} {path}",
                "template": template,
                "coverage_goal": "95%+",
                "test_count": len(methods),
            })

        # If we saw register_blueprint but no concrete routes discovered, add a smoke test
        if not routes and analysis.get("flask_register_blueprint"):
            tests.append({
                "type": "flask_blueprint_smoke",
                "target": "register_blueprint",
                "template": self._blueprint_smoke_template(app_module_guess),
                "coverage_goal": "95%+",
                "test_count": 1
            })

        return tests

    def _route_test_template(self, *, path: str, methods: List[str], app_module: str) -> str:
        safe_name = self._sanitize_name(path)
        lines = [
            '"""',
            f'Route tests for {path} on real Flask app.',
            '"""',
            "",
            "import importlib",
            "import pytest",
            "",
            "def _load_app():",
            f'    mod = importlib.import_module("{app_module}")',
            '    app = getattr(mod, "app", None)',
            '    if app is None and hasattr(mod, "create_app"):',
            "        app = mod.create_app()",
            "    assert app is not None, 'Could not obtain Flask app instance'",
            "    return app",
            "",
        ]
        for m in methods:
            m_upper = (m or "GET").upper()
            lines += [
                f"def test_{m_upper.lower()}_{safe_name}():",
                "    app = _load_app()",
                "    client = app.test_client()",
                f'    resp = client.open("{path}", method="{m_upper}")',
                "    assert resp.status_code in (200, 201, 202, 204, 301, 302, 400, 401, 403, 404, 405)",
                "",
            ]
        return "\n".join(lines)

    def _blueprint_smoke_template(self, app_module: str) -> str:
        return f'''
"""
Blueprint registration smoke test to exercise import-time side effects.
"""
def test_blueprint_registration_smoke():
    __import__("{app_module}")
    assert True
'''

    # -------- Dependencies / env helpers -------------------------------------
    def get_framework_dependencies(self) -> List[str]:  # type: ignore[override]
        """
        Include pytest + pytest-flask so tests can run naturally with client fixtures if desired.
        """
        base = [
            "flask",
            "flask-sqlalchemy",
            "flask-login",
            "flask-wtf",
            "flask-bcrypt",
            "flask-testing",
        ]
        return base + ["pytest", "pytest-cov", "pytest-flask"]

    def setup_framework_environment(self, target_root: pathlib.Path) -> None:
        """
        Flask typically needs no special env to import the app; this is a no-op hook to
        match other handlers’ interface.
        """
        _ = target_root  # intentionally unused

    # -------- Utility helpers -------------------------------------------------
    def _guess_app_module(self, analysis: Dict[str, Any]) -> str:
        """
        Guess the module path exposing `app` or `create_app`.
        Preference: app.py, wsgi.py, then any module with 'app' in name.
        """
        ps = analysis.get("project_structure", {}) or {}
        module_paths = list(ps.get("module_paths", {}).keys()) if isinstance(ps, dict) else []
        for pref in ("app.py", "wsgi.py"):
            for p in module_paths:
                if p.lower().endswith(pref):
                    return p.replace("/", ".").replace("\\", ".").rstrip(".py")
        for p in module_paths:
            if "app" in p.lower():
                return p.replace("/", ".").replace("\\", ".").rstrip(".py")
        return "app"

    def _sanitize_name(self, s: str) -> str:
        return (s or "/").strip("/").replace("/", "_").replace("<", "").replace(">", "").replace(":", "_") or "root"

    def _kwarg(self, call: ast.Call, name: str):
        for kw in call.keywords or []:
            if kw.arg == name:
                return self._safe_value(kw.value)
        return None

    def _attr_chain(self, node: ast.AST) -> str:
        """Return dotted attribute chain (e.g., 'app.register_blueprint')."""
        try:
            if isinstance(node, ast.Attribute):
                left = self._attr_chain(node.value)
                return f"{left}.{node.attr}" if left else node.attr
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Call):
                return self._attr_chain(node.func)
        except Exception:
            pass
        return ""

    def _call_name(self, call: ast.Call) -> str:
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return ""

    def _safe_value(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._safe_value(node.value)}.{node.attr}"
        if isinstance(node, ast.List):
            return [self._safe_value(e) for e in node.elts]
        if isinstance(node, ast.Call):
            return f"{self._call_name(node)}(...)"
        return str(getattr(node, "id", node.__class__.__name__))

    # -------- Pattern summary for UI/decisions --------------------------------
    def detect_framework_patterns(self, analysis: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        routes = analysis.get("flask_routes", []) or analysis.get("routes", [])
        routes = [r for r in routes if r.get("framework") in (None, "flask")]  # accept local discovery
        return {
            "framework": "flask",
            "route_count": len(routes),
            "blueprint_count": len(analysis.get("flask_blueprints", []) or []),
            "has_app_factory": bool(analysis.get("flask_app_factories")),
            "has_error_handlers": bool(analysis.get("flask_error_handlers")),
            "lifecycle_hooks": analysis.get("flask_lifecycle", {"before_request": [], "after_request": [], "teardown_request": []}),
        }
