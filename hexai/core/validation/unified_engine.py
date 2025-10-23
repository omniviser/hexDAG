from pydantic import BaseModel

from .base_parser import ValidationResult
from .json_parser import SecureJSONParser
from .yaml_parser import SecureYAMLParser


class UnifiedParsingEngine:
    def __init__(self) -> None:
        self.parsers = [SecureJSONParser(), SecureYAMLParser()]

    def auto_detect_and_parse(self, content: str, schema: type[BaseModel]) -> ValidationResult:
        errors: list[str] = []

        # Try to detect format (fenced blocks)
        for parser in self.parsers:
            extracted = parser.extract_from_response(content)
            if extracted:
                res = parser.parse_and_validate(extracted, schema)
                if res.ok:
                    return res
                if res.errors:
                    errors += [f"[{parser.get_format_name()}][extracted] {e}" for e in res.errors]

        for parser in self.parsers:
            res = parser.parse_and_validate(content, schema)
            if res.ok:
                return res
            if res.errors:
                errors += [f"[{parser.get_format_name()}][raw] {e}" for e in res.errors]

        return ValidationResult(ok=False, errors=errors or ["Failed to detect format."])
