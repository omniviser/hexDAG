import os


class Secret:
    """Minimal secret wrapper to avoid accidental str() in logs."""

    def __init__(self, value: str) -> None:
        self.__value = value  # Double underscore for name mangling

    def get(self) -> str:  # Added return type annotation
        """Retrieve the secret value securely."""
        return self.__value

    def __repr__(self) -> str:  # Added return type annotation
        return "<SECRET>"

    def __str__(self) -> str:  # Added return type annotation
        return "<SECRET>"

    @staticmethod
    def retrieve_secret_from_env(name: str) -> "Secret":
        """
        Fetch a secret from environment variable, or raise error if not found.
        Wraps the value in the Secret class to avoid accidental logging.

        Args:
            name (str): The environment variable name.
        Returns:
            Secret: The wrapped secret value.

        Raises:
            ValueError: If the secret is not found.
        """
        try:
            value = os.getenv(name)
            if value is None:
                raise ValueError(f"Secret '{name}' not found in environment variables.")
            return Secret(value)
        except Exception as e:
            raise ValueError(f"Error fetching secret '{name}': {e}")
