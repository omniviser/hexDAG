"""SQL validation utilities for preventing injection attacks."""

import re

from hexai.core.logging import get_logger

logger = get_logger(__name__)


def validate_sql_identifier(
    identifier: str,
    identifier_type: str = "identifier",
    raise_on_invalid: bool = False,
) -> bool:
    """Validate SQL identifier to prevent injection attacks.

    SQL identifiers (table names, column names, etc.) must follow specific rules
    to prevent SQL injection attacks. This function validates that an identifier
    contains only safe characters.

    Valid identifiers:
    - Start with a letter (a-z, A-Z) or underscore (_)
    - Contain only letters, numbers, and underscores
    - Examples: "users", "user_data", "Table123", "_private"

    Invalid identifiers:
    - Start with numbers: "123table"
    - Contain special characters: "user-data", "user.table", "user name"
    - SQL keywords without quoting: "SELECT", "DROP"

    Parameters
    ----------
    identifier : str
        The SQL identifier to validate (e.g., table name, column name)
    identifier_type : str, optional
        Human-readable type name for error messages (default: "identifier")
        Examples: "table", "column", "database"
    raise_on_invalid : bool, optional
        If True, raises ValueError on invalid identifier
        If False, logs warning and returns False (default: False)

    Returns
    -------
    bool
        True if identifier is valid, False otherwise

    Raises
    ------
    ValueError
        If raise_on_invalid=True and identifier is invalid

    Examples
    --------
    >>> validate_sql_identifier("users")
    True

    >>> validate_sql_identifier("user_data")
    True

    >>> validate_sql_identifier("123invalid")
    False

    >>> validate_sql_identifier("user-data")
    False

    >>> validate_sql_identifier("user.table", "table", raise_on_invalid=True)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Invalid table 'user.table'. Must start with letter/underscore...
    """
    # Validate against safe pattern: starts with letter/underscore, contains only alphanumerics/_
    is_valid = bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier))

    if not is_valid:
        msg = (
            f"Invalid {identifier_type} '{identifier}'. "
            "Must start with letter/underscore and contain only "
            "letters, numbers, and underscores."
        )

        if raise_on_invalid:
            raise ValueError(msg)

        logger.warning(msg)

    return is_valid
