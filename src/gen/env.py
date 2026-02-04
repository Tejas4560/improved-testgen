# src/gen/env.py
import json
import os
import pathlib
from typing import Any, Dict, List, Optional

# Project root detection with fallback handling
try:
    REPO_ROOT = pathlib.Path(".").resolve()
except Exception:
    REPO_ROOT = pathlib.Path(os.getcwd())

# Configuration with intelligent defaults
PROMPT_STYLE = os.getenv("TESTGEN_PROMPT_STYLE", "professional").strip().lower()
STRICT_FAIL = os.getenv("TESTGEN_STRICT_FAIL", "0").lower() in ("1", "true", "yes")
ENABLE_DEBUG = os.getenv("TESTGEN_DEBUG", "0").lower() in ("1", "true", "yes")

# Test generation behavior configuration
MAX_RETRIES = int(os.getenv("TESTGEN_MAX_RETRIES", "3"))
GENERATION_TIMEOUT = int(os.getenv("TESTGEN_TIMEOUT", "300"))  # 5 minutes default
FORCE_REGENERATION = os.getenv("TESTGEN_FORCE", "false").lower() == "true"

def get_any_env(*names: str) -> str:
    """
    Get environment variable from multiple possible names.
    Raises RuntimeError if none are found.
    """
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    
    raise RuntimeError(f"Missing required environment variable. Tried: {', '.join(names)}")

def get_optional_env(*names: str, default: str = "") -> str:
    """
    Get environment variable from multiple possible names with default fallback.
    """
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default

def norm_rel(path: str) -> str:
    """
    Normalize path to relative format for consistent handling.
    """
    if not path:
        return ""
    
    try:
        path_obj = pathlib.Path(path)
        
        # Convert absolute paths to relative if possible
        if path_obj.is_absolute():
            try:
                path_obj = path_obj.resolve().relative_to(REPO_ROOT)
            except ValueError:
                # Path is outside repo root, keep as-is
                pass
        
        # Normalize to forward slashes for consistency
        normalized = str(path_obj.as_posix())
        
        # Clean up common path prefixes
        if normalized.startswith("./"):
            normalized = normalized[2:]
        
        # Remove target/ prefix if present (for internal path handling)
        if normalized.startswith("target/"):
            normalized = normalized[len("target/"):]
        
        return normalized
        
    except Exception:
        # Fallback for problematic paths
        normalized = str(path).replace("\\", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        if normalized.startswith("target/"):
            normalized = normalized[len("target/"):]
        return normalized

def load_json_list(path: Optional[str]) -> Optional[List[str]]:
    """
    Load a JSON file containing a list of strings.
    Returns None if file doesn't exist or can't be parsed.
    """
    if not path:
        return None
    
    file_path = pathlib.Path(path)
    if not file_path.exists():
        if ENABLE_DEBUG:
            print(f"Debug: JSON list file not found: {path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Convert all items to strings and filter out empty ones
            return [str(item).strip() for item in data if str(item).strip()]
        else:
            print(f"Warning: Expected list in {path}, got {type(data).__name__}")
            return None
            
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not load JSON list from {path}: {e}")
        return None

def load_config_dict(path: Optional[str]) -> Dict[str, Any]:
    """
    Load a JSON configuration file.
    Returns empty dict if file doesn't exist or can't be parsed.
    """
    if not path:
        return {}
    
    file_path = pathlib.Path(path)
    if not file_path.exists():
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            return data
        else:
            print(f"Warning: Expected dict in {path}, got {type(data).__name__}")
            return {}
            
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not load config from {path}: {e}")
        return {}

def get_target_root() -> pathlib.Path:
    """Get the target root directory for analysis."""
    target_root_str = os.getenv("TARGET_ROOT", "target")
    target_root = pathlib.Path(target_root_str)
    
    # Ensure target root exists
    if not target_root.exists():
        print(f"Warning: Target root {target_root} does not exist")
        
        # Try alternative common locations
        alternatives = [".", "src", "app", "backend"]
        for alt in alternatives:
            alt_path = pathlib.Path(alt)
            if alt_path.exists() and any(alt_path.glob("*.py")):
                print(f"Using alternative target root: {alt}")
                return alt_path
        
        # Create target directory if none found
        print(f"Creating target directory: {target_root}")
        target_root.mkdir(parents=True, exist_ok=True)
    
    return target_root

def get_output_dir() -> pathlib.Path:
    """Get the output directory for generated tests."""
    output_dir_str = os.getenv("TESTGEN_OUTPUT_DIR", "tests/generated")
    output_dir = pathlib.Path(output_dir_str)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir

def validate_environment() -> bool:
    """
    Validate that the environment is properly configured for test generation.
    Returns True if valid, False otherwise.
    """
    issues = []
    
    # Check required Azure OpenAI environment variables
    try:
        get_any_env("AZURE_OPENAI_KEY", "AZURE_OPENAI_API_KEY")
    except RuntimeError:
        issues.append("Missing Azure OpenAI API key")
    
    try:
        get_any_env("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_ENDPOINT")
    except RuntimeError:
        issues.append("Missing Azure OpenAI endpoint")
    
    try:
        get_any_env("AZURE_OPENAI_DEPLOYMENT", "OPENAI_DEPLOYMENT")
    except RuntimeError:
        issues.append("Missing Azure OpenAI deployment name")
    
    # Check target root
    target_root = get_target_root()
    if not any(target_root.glob("*.py")):
        issues.append(f"No Python files found in target root: {target_root}")
    
    # Report issues
    if issues:
        print("Environment validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True

def get_generation_config() -> Dict[str, Any]:
    """Get comprehensive configuration for test generation."""
    return {
        "target_root": str(get_target_root()),
        "output_dir": str(get_output_dir()),
        "prompt_style": PROMPT_STYLE,
        "strict_fail": STRICT_FAIL,
        "debug_enabled": ENABLE_DEBUG,
        "max_retries": MAX_RETRIES,
        "timeout": GENERATION_TIMEOUT,
        "force_regeneration": FORCE_REGENERATION,
        
        # File limits
        "max_unit_files": int(os.getenv("TESTGEN_MAX_UNIT_FILES", "4")),
        "max_integ_files": int(os.getenv("TESTGEN_MAX_INTEG_FILES", "3")),
        "max_e2e_files": int(os.getenv("TESTGEN_MAX_E2E_FILES", "2")),
        
        # Generation behavior
        "enable_gui_shims": os.getenv("TESTGEN_ENABLE_GUI_SHIMS", "0").lower() in ("1", "true", "yes"),
        "pip_constraints": get_optional_env("TESTGEN_PIP_CONSTRAINTS", "PIP_CONSTRAINT"),
        
        # Focus configuration
        "focus_files_json": get_optional_env("FOCUS_FILES_JSON_PATH"),
        
        # Azure OpenAI configuration (with fallbacks)
        "azure_openai": {
            "api_key": get_optional_env("AZURE_OPENAI_KEY", "AZURE_OPENAI_API_KEY"),
            "endpoint": get_optional_env("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_ENDPOINT"),
            "deployment": get_optional_env("AZURE_OPENAI_DEPLOYMENT", "OPENAI_DEPLOYMENT"),
            "api_version": get_optional_env("AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION", default="2023-12-01-preview")
        }
    }

# Legacy function alias for backward compatibility
load_list = load_json_list