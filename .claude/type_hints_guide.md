# Comprehensive Python 3.12+ Type Hint Guide

This document elaborates on the mandatory type hint syntax enforced by Ruff (UP rules), Pyupgrade, MyPy, and Pyright. **This supersedes all standard style guides.**

## I. Mandatory Modern Syntax Enforcement

### A. Generics (Python 3.9+)
ALWAYS use the built-in lowercase generics. (Enforced by Ruff UP006, UP035).

| Purpose | ✅ DO USE | ❌ FORBIDDEN |
|---|---|---|
| List | `def f(a: list[str]) -> list[int]: ...` | `def f(a: List[str]) -> List[int]: ...` |
| Dictionary | `def f(a: dict[str, int]) -> dict[str, float]: ...` | `def f(a: Dict[str, int]) -> Dict[str, float]: ...` |

### B. Union and Optional (Python 3.10+)
ALWAYS use the pipe operator (`|`). (Enforced by Ruff UP007, UP037).

| Purpose | ✅ DO USE | ❌ FORBIDDEN |
|---|---|---|
| Union | `def parse_value(val: str | int) -> str: ...` | `def parse_value(val: Union[str, int]) -> str: ...` |
| Optional | `def find_user(id: int) -> User | None: ...` | `def find_user(id: int) -> Optional[User]: ...` |

### C. Type Aliases (Python 3.12+)
ALWAYS use the `type` statement (enforced by Ruff UP040).

```python
# ✅ DO USE
type UserId = int 
# ❌ FORBIDDEN
from typing import TypeAlias
UserId: TypeAlias = int