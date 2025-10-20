from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class ValidationResult:
    ok: bool
    data: BaseModel | None = None
    errors: list[str] | None = None
    format_name: str = ""


class BaseStructuredParser(ABC):
    @abstractmethod
    def parse_and_validate(self, content: str, schema: type[BaseModel]) -> ValidationResult:
        """Parse and validate data using a given schema."""
        pass

    @abstractmethod
    def extract_from_response(self, response: str) -> str | None:
        """Extract pure JSON/YAML from text (e.g. from ```json ... ```)."""
        pass

    @abstractmethod
    def get_format_name(self) -> str:
        """Return the name of the format (e.g. 'json')."""
        pass
