# src/framework_handlers/manager.py
"""
Framework manager to detect and manage framework handlers.
"""

from typing import Any, Dict, List, Optional

from .base_handler import BaseFrameworkHandler
from .django_handler import DjangoHandler
from .fastapi_handler import FastAPIHandler
from .flask_handler import FlaskHandler
from .universal_handler import UniversalHandler


class FrameworkManager:
    """Manages framework detection and handler selection (no scoring)."""
    
    def __init__(self):
        self.handlers = [
            DjangoHandler(),
            FastAPIHandler(),
            FlaskHandler(),
            UniversalHandler()  # Always last as fallback
        ]
        self.detected_framework: Optional[str] = None
        self.active_handler: Optional[BaseFrameworkHandler] = None

    # -----------------------------------------------------------------------
    def detect_framework(self, analysis: Dict[str, Any]) -> str:
        """
        Detect the main framework used in the project.
        Uses each handler's `can_handle()` to identify the framework realistically,
        without assigning numeric priorities or weights.
        The first positive match wins (ordered by detection confidence, not priority).
        """
        for handler in self.handlers:
            try:
                if handler.can_handle(analysis):
                    self.detected_framework = handler.framework_name
                    self.active_handler = handler
                    break
            except Exception as e:
                # Defensive logging: one handler failing should not block detection
                print(f"[framework_manager] Warning: {handler.framework_name} detection failed -> {e}")
                continue

        # Fallback to universal if nothing matched
        if not self.detected_framework:
            self.detected_framework = "universal"
            self.active_handler = self._get_handler("universal")

        print(f"[framework_manager] Detected framework: {self.detected_framework}")
        return self.detected_framework

    # -----------------------------------------------------------------------
    def get_active_handler(self) -> Optional[BaseFrameworkHandler]:
        """Return the currently active handler."""
        return self.active_handler

    def get_framework_dependencies(self) -> List[str]:
        """Get dependencies for the detected framework."""
        if self.active_handler:
            return self.active_handler.get_framework_dependencies()
        return []

    def setup_framework_environment(self, target_root):
        """Setup the environment for the detected framework (if needed)."""
        if self.active_handler:
            self.active_handler.setup_framework_environment(target_root)

    def get_framework_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get framework-specific pattern analysis."""
        if self.active_handler:
            return self.active_handler.detect_framework_patterns(analysis)
        return {}

    def get_all_handlers(self) -> List[BaseFrameworkHandler]:
        """Return all available framework handlers."""
        return self.handlers

    # -----------------------------------------------------------------------
    def _get_handler(self, name: str) -> Optional[BaseFrameworkHandler]:
        """Find a handler by framework name."""
        for h in self.handlers:
            if h.framework_name == name:
                return h
        return None
    

# ----------------------- APPENDED ENHANCEMENTS BELOW -----------------------

class FrameworkManager(FrameworkManager):  # type: ignore[misc]
    """
    Extended manager providing realistic evidence-based detection logging
    and explicit multi-framework awareness.
    """

    def detect_framework(self, analysis: Dict[str, Any]) -> str:  # type: ignore[override]
        """
        Extended detection pipeline:
        - Each handler runs its `can_handle()` check.
        - Collects all frameworks that match (since some hybrid repos use multiple frameworks).
        - If multiple frameworks detected, select the most specific one:
            - Django if 'manage.py' or 'settings.py' found
            - FastAPI if 'FastAPI()' or 'APIRouter' present
            - Flask if 'Flask(__name__)' or '@app.route' found
          Otherwise, fallback to universal.
        """
        detected: List[str] = []

        for handler in self.handlers:
            try:
                if handler.can_handle(analysis):
                    detected.append(handler.framework_name)
            except Exception as e:
                print(f"[framework_manager] Detection error in {handler.framework_name}: {e}")

        if not detected:
            self.detected_framework = "universal"
            self.active_handler = self._get_handler("universal")
            print("[framework_manager] No specific framework detected, using universal handler.")
            return "universal"

        # If multiple frameworks match, refine by evidence in analysis
        chosen = self._resolve_conflicts(detected, analysis)
        self.detected_framework = chosen
        self.active_handler = self._get_handler(chosen)

        print(f"[framework_manager] Framework(s) detected: {detected} -> selected: {chosen}")
        return chosen

    def _resolve_conflicts(self, detected: List[str], analysis: Dict[str, Any]) -> str:
        """
        Resolve ambiguous detections by using real project clues instead of numeric priority.
        """
        if len(detected) == 1:
            return detected[0]

        # Django signal: manage.py or settings.py
        ps = analysis.get("project_structure", {}) or {}
        module_paths = set(ps.get("module_paths", {}).keys()) if isinstance(ps, dict) else set()
        if "django" in detected and any(p.endswith(("manage.py", "settings.py")) for p in module_paths):
            return "django"

        # Helper to safely extract module names from imports (handles both dict and str)
        def _import_mentions(framework_name: str) -> bool:
            for imp in analysis.get("imports", []):
                if isinstance(imp, dict):
                    modules = imp.get("modules", [])
                    if isinstance(modules, str):
                        modules = [modules]
                    if any(framework_name in str(m).lower() for m in modules):
                        return True
                elif isinstance(imp, str):
                    if framework_name in imp.lower():
                        return True
            return False

        # FastAPI signal: presence of FastAPI(), APIRouter(), or fastapi_routes
        if "fastapi" in detected and (
            analysis.get("fastapi_routes")
            or analysis.get("fastapi_app_inits")
            or _import_mentions("fastapi")
        ):
            return "fastapi"

        # Flask signal: @app.route or app = Flask(__name__)
        if "flask" in detected and (
            analysis.get("flask_routes")
            or analysis.get("flask_app_inits")
            or _import_mentions("flask")
        ):
            return "flask"

        # Default to first detected alphabetically to ensure determinism
        return sorted(detected)[0]

class FrameworkManager(FrameworkManager):  # type: ignore[misc]
    """
    Safe wrapper to normalize 'analysis' into a dict so handlers never crash
    when a string or other type is accidentally passed down.
    """

    def _normalize_analysis(self, analysis: Any) -> Dict[str, Any]:
        if isinstance(analysis, dict):
            return analysis
        # Preserve the original value for debugging
        return {"_raw_analysis": analysis}

    def detect_framework(self, analysis: Any) -> str:  # type: ignore[override]
        a = self._normalize_analysis(analysis)

        detected = []
        for handler in self.handlers:
            try:
                if handler.can_handle(a):
                    detected.append(handler.framework_name)
            except Exception as e:
                print(f"[framework_manager] Detection error in {handler.framework_name}: {e}")

        if not detected:
            self.detected_framework = "universal"
            self.active_handler = self._get_handler("universal")
            print("[framework_manager] No specific framework detected, using universal handler.")
            return "universal"

        # Deterministic resolution: pick alphabetically among detected
        chosen = sorted(detected)[0]
        self.detected_framework = chosen
        self.active_handler = self._get_handler(chosen)
        print(f"[framework_manager] Framework(s) detected: {detected} -> selected: {chosen}")
        return chosen
