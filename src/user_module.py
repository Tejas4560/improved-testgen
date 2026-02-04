"""
Simple User Module for Testing Auto-Fixer

This is source code that will be tested.
"""


class User:
    """Simple user class."""

    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email
        self.is_active = True

    def activate(self):
        """Activate the user."""
        self.is_active = True
        return True

    def deactivate(self):
        """Deactivate the user."""
        self.is_active = False
        return True

    def get_display_name(self):
        """Get formatted display name."""
        return f"{self.name} ({self.email})"


def create_user(name: str, email: str) -> User:
    """Factory function to create a user."""
    return User(name, email)


def validate_email(email: str) -> bool:
    """Validate email format."""
    return "@" in email and "." in email.split("@")[1]
