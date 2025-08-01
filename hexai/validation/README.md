# HexAI Unified Validation Framework

A unified validation system for all HexAI node types and execution contexts, providing configurable strategies and extensible architecture.

## Overview

The validation framework provides three core validation strategies:

- **STRICT**: Fail fast on any type mismatch or validation error
- **COERCE**: Attempt type conversion and coercion when possible
- **PASSTHROUGH**: Minimal validation, allow data to pass through unchanged

## Quick Start

### Basic Usage

```python
from hexai.validation import ValidationStrategy, BaseValidator, ValidationContext

# Create a validator with coerce strategy
validator = BaseValidator(ValidationStrategy.COERCE)

# Create validation context
context = ValidationContext(node_name="text2sql", pipeline_name="main")

# Validate input data
result = validator.validate_input("123", int, context)

if result.is_valid:
    print(f"Validation successful: {result.data}")  # Output: 123 (int)
else:
    print(f"Validation failed: {result.errors}")
```

### Using Factory Functions

```python
from hexai.validation import strict_validator, coerce_validator, passthrough_validator

# Create validators using convenience functions
strict = strict_validator()
coerce = coerce_validator()
passthrough = passthrough_validator()

# Test type conversion
result = coerce.validate_input("42.5", float)
print(result.data)  # Output: 42.5 (float)
```

### Pydantic Model Validation

```python
from pydantic import BaseModel
from hexai.validation import ValidationStrategy, BaseValidator

class UserModel(BaseModel):
    name: str
    age: int

validator = BaseValidator(ValidationStrategy.STRICT)

# Valid data
result = validator.validate_input({"name": "John", "age": 30}, UserModel)
print(f"Valid: {result.is_valid}")  # True

# Invalid data
result = validator.validate_input({"name": "John", "age": "not_a_number"}, UserModel)
print(f"Valid: {result.is_valid}")  # False
```

## Validation Strategies

### STRICT Strategy
- Performs exact type checking
- Fails immediately on type mismatch
- No type conversion attempted
- Use for production systems where data integrity is critical

```python
from hexai.validation import strict_validator

validator = strict_validator()
result = validator.validate_input("123", int)
# Result: is_valid=False, errors=["Expected int, got str"]
```

### COERCE Strategy
- Attempts intelligent type conversion
- Converts between compatible types
- Provides warnings for conversions
- Use for development and data pipeline processing

```python
from hexai.validation import coerce_validator

validator = coerce_validator()
result = validator.validate_input("123", int)
# Result: is_valid=True, data=123, warnings=["Converted str to int"]
```

### PASSTHROUGH Strategy
- Minimal validation
- Allows all data to pass through
- Only logs warnings for type mismatches
- Use for debugging or when validation is not needed

```python
from hexai.validation import passthrough_validator

validator = passthrough_validator()
result = validator.validate_input("wrong_type", int)
# Result: is_valid=True, data="wrong_type"
```

## Type Conversion Features

The COERCE strategy supports intelligent type conversion:

### String to Numeric
```python
validator = coerce_validator()

# String to int
result = validator.validate_input("42", int)
# Result: data=42 (int)

# String to float
result = validator.validate_input("3.14", float)
# Result: data=3.14 (float)
```

### Collection Conversion
```python
# Tuple to list
result = validator.validate_input((1, 2, 3), list)
# Result: data=[1, 2, 3]

# Non-iterable to list
result = validator.validate_input("single_item", list)
# Result: data=["single_item"]
```

### Pydantic Model to Dict
```python
class TestModel(BaseModel):
    name: str
    value: int

model = TestModel(name="test", value=42)
result = validator.validate_input(model, dict)
# Result: data={"name": "test", "value": 42}
```

## Validation Context

Provide runtime context for better error messages and debugging:

```python
from hexai.validation import ValidationContext, strict_validator

context = ValidationContext(
    node_name="text2sql_agent",
    pipeline_name="customer_analytics",
    validation_stage="input",
    metadata={"user_id": "12345"}
)

validator = strict_validator()
result = validator.validate_input("invalid", int, context)
# Error message includes node and pipeline information
```

## Error Handling

### ValidationResult Object
```python
class ValidationResult:
    is_valid: bool          # True if validation succeeded
    data: Any              # Validated/converted data
    errors: list[str]      # List of error messages
    warnings: list[str]    # List of warning messages
    metadata: dict         # Additional metadata
```

### Boolean Conversion
```python
result = validator.validate_input(data, type_hint)

# Use as boolean
if result:
    process_data(result.data)
else:
    handle_errors(result.errors)
```

## Integration with HexAI Nodes

The validation framework is designed to integrate seamlessly with HexAI's node system:

```python
# In node implementation
from hexai.validation import coerce_validator

class MyNode:
    def __init__(self):
        self.validator = coerce_validator()

    def execute(self, input_data):
        # Validate input
        context = ValidationContext(node_name=self.__class__.__name__)
        result = self.validator.validate_input(input_data, self.input_type, context)

        if not result:
            raise ValidationError(f"Input validation failed: {result.errors}")

        # Process validated data
        processed = self.process(result.data)

        # Validate output
        output_result = self.validator.validate_output(processed, self.output_type, context)
        return output_result.data
```

## Best Practices

1. **Use STRICT in production** for data integrity
2. **Use COERCE in development** for flexibility
3. **Always check ValidationResult.is_valid** before using data
4. **Provide ValidationContext** for better error messages
5. **Handle conversion warnings** in logging/monitoring
6. **Use factory functions** for common patterns

## Future Extensions

This framework is designed to be extended with:
- YAML-based schema configuration (Phase 2)
- Custom type converters
- Fallback value providers
- Integration with Great Expectations
- Compile-time optimization
