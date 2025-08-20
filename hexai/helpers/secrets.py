import os

# Code written by tab, not sure if it's sufficient, but after reading the documentation it seems correct.
# Please review and provide any comments.


class Secret:
    """Minimal secret wrapper to avoid accidental str() in logs."""

    def __init__(self, value):  # type: ignore[no-untyped-def]
        self._value = value

    def get(self):  # type: ignore[no-untyped-def]
        return self._value

    def __repr__(self):  # type: ignore[no-untyped-def]
        return "<SECRET>"

    def __str__(self):  # type: ignore[no-untyped-def]
        return "<SECRET>"


def get_secret(name: str, required: bool = True):  # type: ignore[no-untyped-def]
    """
    Fetch a secret from environment variable, or raise error if not found.
    Optionally allow a default (use with care).
    """
    value = os.getenv(name)
    if required and value is None:
        raise RuntimeError(
            f"Required secret '{name}' not found in environment. Set it as an environment variable."
        )
    return value


# # Example usage:
# db_password = get_secret("POSTGRES_PASSWORD")
# openai_key = get_secret("OPENAI_API_KEY", required=True)

# print(db_password)  # This will print the secret value
# print(openai_key)  # This will print the secret value
# # Note: In production, avoid printing secrets to logs or console.
# # Use the Secret class to wrap sensitive values if needed.
# db_password_secret = Secret(db_password)
# openai_key_secret = Secret(openai_key)

# print(db_password_secret)  # This will print <SECRET>
# print(openai_key_secret)  # This will print <SECRET>
# # Use db_password_secret.get() to access the actual value when needed.
# # This ensures that secrets are not accidentally logged or printed.
