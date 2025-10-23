from pydantic import BaseModel, ValidationError

from .base_parser import (
    BaseStructuredParser,
    ValidationResult,
)
from .secure_json import SafeJSON


class SecureJSONParser(BaseStructuredParser):
    def __init__(self) -> None:
        self.safe_json = SafeJSON()

    def get_format_name(self) -> str:
        return "json"

    def extract_from_response(self, response: str) -> str | None:
        return self.safe_json._extract_json(response)

    def parse_and_validate(self, content: str, schema: type[BaseModel]) -> ValidationResult:
        parse_result = self.safe_json.loads(content)
        if not parse_result.ok:
            message = parse_result.message or "Invalid JSON"
            return ValidationResult(
                ok=False,
                errors=[message],
                format_name=self.get_format_name(),
            )

        try:
            model = schema.model_validate(parse_result.data)
            return ValidationResult(
                ok=True,
                data=model,
                errors=None,
                format_name=self.get_format_name(),
            )
        except ValidationError as e:
            errors: list[str] = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            return ValidationResult(
                ok=False,
                errors=errors,
                format_name=self.get_format_name(),
            )
