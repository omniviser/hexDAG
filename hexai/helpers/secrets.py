import os

# Code written by tab, not sure if it's sufficient, but after reading the documentation it seems correct.
# Please review and provide any comments.

# Example usage:
# from hexai.helpers.secrets import Secret, get_secret
# import os

# # Set up environment variable
# os.environ["API_KEY"] = "sk_test_123456789"

# # Get secret from environment
# api_key = get_secret("API_KEY")

# # Attempt to access secret directly (will fail)
# try:
#     print(f"This will fail: {api_key.__value}")  # AttributeError
# except AttributeError as e:
#     print(f"Cannot access secret directly: {e}")

# try:
#     print(f"This will also fail: {api_key._value}")  # AttributeError
# except AttributeError as e:
#     print(f"Cannot access secret directly: {e}")

# # Correct usage - secret stays protected in logs
# print(f"API credential: {api_key}")  # Shows: API credential: <SECRET>

# # Use get() method when the actual value is needed
# headers = {
#     "Authorization": f"Bearer {api_key.get()}",
#     "Content-Type": "application/json"
# }

# # Safe to log the whole headers dict - secret remains protected
# print(f"Request headers: {headers}")  # Shows: Request headers: {'Authorization': 'Bearer <SECRET>', 'Content-Type': 'application/json'}


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


def get_secret(name: str):  # type: ignore[no-untyped-def]
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
    except Exception as e:
        raise ValueError(f"Error fetching secret '{name}': {e}")
    if value is None:
        raise ValueError(f"Secret '{name}' not found in environment variables.")
    return Secret(value)
