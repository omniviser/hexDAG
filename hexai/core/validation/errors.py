"""Custom validation exceptions for the unified validation framework."""


class ValidationError(Exception):
    """Base exception for validation errors."""

    def __init__(self, message: str, context: str | None = None, field: str | None = None):
        """Initialize validation error.

        Parameters
        ----------
        message : str
            Error message describing the validation failure
        context : str | None
            Optional context information (node name, pipeline name)
        field : str | None
            Optional field name that failed validation
        """
        self.message = message
        self.context = context
        self.field = field

        # Build full error message
        parts = [message]
        if field:
            parts.insert(0, f"Field '{field}':")
        if context:
            parts.append(f"(in {context})")

        super().__init__(" ".join(parts))


class ConversionError(ValidationError):
    """Exception raised when type conversion fails."""

    def __init__(self, source_type: str, target_type: str, value: str, context: str | None = None):
        """Initialize conversion error.

        Parameters
        ----------
        source_type : str
            The original type name
        target_type : str
            The target type name
        value : str
            String representation of the value that failed conversion
        context : str | None
            Optional context information
        """
        message = f"Cannot convert {source_type} '{value}' to {target_type}"
        super().__init__(message, context)
        self.source_type = source_type
        self.target_type = target_type
        self.value = value


class SchemaError(ValidationError):
    """Exception raised when schema definition is invalid."""

    def __init__(self, schema_name: str, message: str):
        """Initialize schema error.

        Parameters
        ----------
        schema_name : str
            Name of the schema that is invalid
        message : str
            Description of the schema error
        """
        super().__init__(f"Invalid schema '{schema_name}': {message}")
        self.schema_name = schema_name


class FallbackError(ValidationError):
    """Exception raised when fallback value cannot be provided."""

    def __init__(self, field: str, reason: str, context: str | None = None):
        """Initialize fallback error.

        Parameters
        ----------
        field : str
            Name of the field that needs a fallback
        reason : str
            Reason why fallback failed
        context : str | None
            Optional context information
        """
        message = f"Cannot provide fallback for field '{field}': {reason}"
        super().__init__(message, context, field)
