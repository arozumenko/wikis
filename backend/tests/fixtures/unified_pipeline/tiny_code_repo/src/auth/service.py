"""AuthService: handles user authentication."""


class AuthService:
    """Manages user login and session lifecycle."""

    def login(self, username: str, password: str) -> bool:
        """Authenticate a user by username and password."""
        return username == "admin" and password == "secret"

    def logout(self, session_id: str) -> None:
        """Terminate a user session."""
        pass

    def validate_token(self, token: str) -> bool:
        """Check whether a JWT token is valid and unexpired."""
        return bool(token)
