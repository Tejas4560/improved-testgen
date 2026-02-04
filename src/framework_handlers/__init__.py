# src/framework_handlers/__init__.py
"""
Framework-specific handlers for test generation.
Each handler manages framework-specific analysis and test generation patterns.
"""

from .base_handler import BaseFrameworkHandler
from .django_handler import DjangoHandler
from .fastapi_handler import FastAPIHandler
from .flask_handler import FlaskHandler
from .universal_handler import UniversalHandler

__all__ = [
    'BaseFrameworkHandler',
    'DjangoHandler', 
    'FastAPIHandler',
    'FlaskHandler',
    'UniversalHandler'
]