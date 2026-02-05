# src/gen/conftest_text.py — Framework-aware conftest generation
# Generates lean, framework-specific conftest.py files

import os


def conftest_text(framework: str = "universal", target_root: str = "") -> str:
    """
    Generate framework-specific conftest.py content.

    Args:
        framework: One of "flask", "django", "fastapi", "universal"
        target_root: The target project root path (for sys.path setup)

    Returns:
        Conftest.py content optimized for the detected framework
    """
    framework = framework.lower().strip()

    if framework == "flask":
        return _flask_conftest(target_root)
    elif framework == "django":
        return _django_conftest(target_root)
    elif framework == "fastapi":
        return _fastapi_conftest(target_root)
    else:
        return _universal_conftest(target_root)


def _base_conftest(target_root: str = "") -> str:
    """Common base for all conftest files."""
    return f'''"""
pytest configuration for AI-generated tests.
Framework-specific setup with minimal dependencies.
"""

import os
import sys
import pytest
import warnings

# Suppress deprecation warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

# Set testing environment
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("LOG_LEVEL", "ERROR")

# Add target project to Python path
TARGET_ROOT = os.environ.get("TARGET_ROOT", "{target_root or ''}")
if TARGET_ROOT and TARGET_ROOT not in sys.path:
    sys.path.insert(0, TARGET_ROOT)

# Also add current directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
'''


def _flask_conftest(target_root: str = "") -> str:
    """Generate Flask-specific conftest (~60 lines)."""
    return _base_conftest(target_root) + '''

# ============== FLASK-SPECIFIC CONFIGURATION ==============

# Try to import the Flask app
_flask_app = None
try:
    # Try common app module names
    for module_name in ['app', 'application', 'main', 'server', 'api']:
        try:
            mod = __import__(module_name)
            # Look for app object or create_app factory
            if hasattr(mod, 'app'):
                _flask_app = mod.app
                break
            elif hasattr(mod, 'create_app'):
                _flask_app = mod.create_app()
                break
            elif hasattr(mod, 'application'):
                _flask_app = mod.application
                break
        except ImportError:
            continue
except Exception:
    pass


@pytest.fixture(scope="session")
def app():
    """Flask application fixture."""
    if _flask_app is None:
        pytest.skip("No Flask app found")

    _flask_app.config['TESTING'] = True
    _flask_app.config['WTF_CSRF_ENABLED'] = False

    ctx = _flask_app.app_context()
    ctx.push()
    yield _flask_app
    ctx.pop()


@pytest.fixture
def client(app):
    """Flask test client fixture."""
    return app.test_client()


@pytest.fixture(autouse=True)
def reset_app_state():
    """Reset any global state between tests."""
    # Import app module to access global state
    for module_name in ['app', 'application', 'main']:
        try:
            mod = __import__(module_name)
            # Clear common global state patterns
            if hasattr(mod, 'tasks') and isinstance(getattr(mod, 'tasks'), list):
                getattr(mod, 'tasks').clear()
            if hasattr(mod, 'data') and isinstance(getattr(mod, 'data'), (list, dict)):
                if isinstance(getattr(mod, 'data'), list):
                    getattr(mod, 'data').clear()
                else:
                    getattr(mod, 'data').clear()
            break
        except ImportError:
            continue
    yield


@pytest.fixture
def sample_data():
    """Sample test data for Flask apps."""
    return {
        "title": "Test Item",
        "description": "Test Description",
        "name": "Test Name",
        "email": "test@example.com",
        "username": "testuser",
        "password": "testpass123",
    }


@pytest.fixture
def auth_headers():
    """Authorization headers for API testing."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
'''


def _django_conftest(target_root: str = "") -> str:
    """Generate Django-specific conftest (~150 lines)."""
    return _base_conftest(target_root) + '''

# ============== DJANGO-SPECIFIC CONFIGURATION ==============

import tempfile

# Setup Django before any model imports
django_setup = False
try:
    import django
    from django.conf import settings as _dj_settings

    if not _dj_settings.configured:
        settings_module = os.environ.get('DJANGO_SETTINGS_MODULE')

        if not settings_module:
            import glob
            search_root = TARGET_ROOT if TARGET_ROOT else '.'
            settings_files = glob.glob(f'{search_root}/**/settings.py', recursive=True)
            for sf in settings_files:
                if 'venv' in sf or 'site-packages' in sf:
                    continue
                rel = os.path.relpath(sf, start=search_root)
                rel = rel.replace('\\\\', '/').replace('\\\\', '/')
                if rel.endswith('.py'):
                    rel = rel[:-3]
                settings_module = rel.replace('/', '.').lstrip('.')
                break

        if settings_module:
            os.environ['DJANGO_SETTINGS_MODULE'] = settings_module
            django.setup()
            django_setup = True
        else:
            _dj_settings.configure(
                DEBUG=True,
                TESTING=True,
                SECRET_KEY='test-secret-key',
                DATABASES={"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}},
                INSTALLED_APPS=[
                    'django.contrib.auth',
                    'django.contrib.contenttypes',
                    'django.contrib.sessions',
                    'django.contrib.messages',
                ],
                MIDDLEWARE=[],
                ROOT_URLCONF=None,
                USE_TZ=True,
            )
            django.setup()
            django_setup = True
except ImportError:
    pass


if django_setup:
    from django.test import Client, RequestFactory
    from django.contrib.auth import get_user_model

    # Auto-enable database for all tests
    def pytest_collection_modifyitems(config, items):
        marker = pytest.mark.django_db(transaction=True)
        for item in items:
            item.add_marker(marker)

    @pytest.fixture(scope="session")
    def django_db_setup():
        from django.conf import settings
        settings.DATABASES["default"] = {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": ":memory:",
        }

    @pytest.fixture
    def client():
        """Django test client."""
        return Client()

    @pytest.fixture
    def rf():
        """Django RequestFactory."""
        return RequestFactory()

    @pytest.fixture
    def user(db):
        """Create a test user."""
        User = get_user_model()
        return User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )

    @pytest.fixture
    def admin_user(db):
        """Create an admin user."""
        User = get_user_model()
        return User.objects.create_superuser(
            username='admin',
            email='admin@example.com',
            password='adminpass123'
        )

    @pytest.fixture
    def authenticated_client(client, user):
        """Client with logged-in user."""
        client.force_login(user)
        return client

    @pytest.fixture
    def rf_with_session(rf):
        """RequestFactory with session support."""
        from django.contrib.sessions.middleware import SessionMiddleware
        from django.contrib.messages.middleware import MessageMiddleware

        def _make_request(method="get", path="/", data=None, **kwargs):
            method_func = getattr(rf, method.lower(), rf.get)
            request = method_func(path, data=data or {}, **kwargs)

            # Add session
            middleware = SessionMiddleware(lambda r: None)
            middleware.process_request(request)
            request.session.save()

            # Add messages
            msg_middleware = MessageMiddleware(lambda r: None)
            msg_middleware.process_request(request)

            return request
        return _make_request

    @pytest.fixture
    def sample_data():
        """Sample data for Django tests."""
        return {
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpass123",
            "first_name": "Test",
            "last_name": "User",
        }

    # Auto-enable db fixture
    @pytest.fixture(autouse=True)
    def enable_db_access_for_all(db):
        yield
'''


def _fastapi_conftest(target_root: str = "") -> str:
    """Generate FastAPI-specific conftest (~70 lines)."""
    return _base_conftest(target_root) + '''

# ============== FASTAPI-SPECIFIC CONFIGURATION ==============

import asyncio

# Try to import the FastAPI app
_fastapi_app = None
try:
    for module_name in ['main', 'app', 'api', 'server', 'application']:
        try:
            mod = __import__(module_name)
            if hasattr(mod, 'app'):
                _fastapi_app = mod.app
                break
            elif hasattr(mod, 'create_app'):
                _fastapi_app = mod.create_app()
                break
        except ImportError:
            continue
except Exception:
    pass


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def app():
    """FastAPI application fixture."""
    if _fastapi_app is None:
        pytest.skip("No FastAPI app found")
    return _fastapi_app


@pytest.fixture
def client(app):
    """FastAPI TestClient fixture."""
    try:
        from fastapi.testclient import TestClient
        return TestClient(app)
    except ImportError:
        from starlette.testclient import TestClient
        return TestClient(app)


@pytest.fixture
def async_client(app):
    """Async client for FastAPI."""
    try:
        from httpx import AsyncClient
        return AsyncClient(app=app, base_url="http://test")
    except ImportError:
        pytest.skip("httpx not installed for async testing")


@pytest.fixture
def sample_data():
    """Sample test data for FastAPI apps."""
    return {
        "title": "Test Item",
        "description": "Test Description",
        "name": "Test Name",
        "email": "test@example.com",
        "username": "testuser",
        "password": "testpass123",
        "is_active": True,
    }


@pytest.fixture
def auth_headers():
    """Authorization headers for API testing."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer test-token",
    }


@pytest.fixture
def mock_db():
    """Mock database for testing without real DB."""
    return {}


@pytest.fixture
def override_dependencies(app):
    """
    Override FastAPI dependencies for testing.

    Usage:
        def test_with_mock_db(client, override_dependencies):
            # Dependencies are overridden for this test
            pass

    To override specific dependencies:
        app.dependency_overrides[get_db] = lambda: mock_db
    """
    original_overrides = app.dependency_overrides.copy()
    yield app.dependency_overrides
    # Restore original dependencies after test
    app.dependency_overrides.clear()
    app.dependency_overrides.update(original_overrides)


@pytest.fixture(autouse=True)
def reset_dependency_overrides(app):
    """Auto-reset dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()
'''


def _universal_conftest(target_root: str = "") -> str:
    """Generate universal conftest for any Python project (~80 lines)."""
    return _base_conftest(target_root) + '''

# ============== UNIVERSAL PYTHON CONFIGURATION ==============

import types
from unittest.mock import Mock

# Try to detect and import any application
_app = None
_framework = None

# Try Flask
try:
    for module_name in ['app', 'application', 'main', 'server']:
        try:
            mod = __import__(module_name)
            if hasattr(mod, 'app'):
                _app = mod.app
                if hasattr(_app, 'test_client'):
                    _framework = 'flask'
                break
        except ImportError:
            continue
except Exception:
    pass

# Try FastAPI if Flask not found
if _app is None:
    try:
        for module_name in ['main', 'app', 'api']:
            try:
                mod = __import__(module_name)
                if hasattr(mod, 'app'):
                    _app = mod.app
                    _framework = 'fastapi'
                    break
            except ImportError:
                continue
    except Exception:
        pass


@pytest.fixture(scope="session")
def app():
    """Universal application fixture."""
    if _app is None:
        pytest.skip("No application found")

    if _framework == 'flask':
        _app.config['TESTING'] = True
        ctx = _app.app_context()
        ctx.push()
        yield _app
        ctx.pop()
    else:
        yield _app


@pytest.fixture
def client(app):
    """Universal test client fixture."""
    if _framework == 'flask':
        return app.test_client()
    elif _framework == 'fastapi':
        try:
            from fastapi.testclient import TestClient
            return TestClient(app)
        except ImportError:
            pass
    pytest.skip("No test client available")


@pytest.fixture
def sample_data():
    """Universal sample test data."""
    return {
        "id": 1,
        "name": "Test Item",
        "title": "Test Title",
        "description": "Test Description",
        "email": "test@example.com",
        "username": "testuser",
        "password": "testpass123",
        "is_active": True,
        "data": {"key": "value"},
    }


@pytest.fixture
def mock_request():
    """Universal mock request object."""
    request = types.SimpleNamespace()
    request.method = "GET"
    request.path = "/test"
    request.data = {}
    request.args = {}
    request.headers = {}
    request.json = lambda: {}
    return request


@pytest.fixture
def authenticated_user():
    """Universal authenticated user mock."""
    user = types.SimpleNamespace()
    user.id = 1
    user.username = "testuser"
    user.email = "test@example.com"
    user.is_authenticated = True
    user.is_active = True
    return user
'''


# Keep backward compatibility - default to universal
def conftest_text_legacy() -> str:
    """Legacy function for backward compatibility - returns universal conftest."""
    return _universal_conftest("")
