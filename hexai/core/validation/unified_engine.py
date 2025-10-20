from pydantic import BaseModel

from hexai.core.validation.base_parser import ValidationResult
from hexai.core.validation.json_parser import SecureJSONParser
from hexai.core.validation.yaml_parser import SecureYAMLParser


class UnifiedParsingEngine:
    def __init__(self) -> None:
        self.parsers = [SecureJSONParser(), SecureYAMLParser()]

    def auto_detect_and_parse(self, content: str, schema: type[BaseModel]) -> ValidationResult:
        # Try to detect format (fenced blocks)
        for parser in self.parsers:
            extracted = parser.extract_from_response(content)
            if extracted:
                return parser.parse_and_validate(extracted, schema)

        # If nothing detected, try both sequentially
        for parser in self.parsers:
            result = parser.parse_and_validate(content, schema)
            if result.ok:
                return result

        return ValidationResult(ok=False, errors=["Failed to detect format."])
