"""Fix missing __init__ parameter assignments in migrated files."""

import re
from pathlib import Path


def fix_in_memory_memory():
    """Fix InMemoryMemory adapter."""
    file = Path("hexdag/builtin/adapters/memory/in_memory_memory.py")
    content = file.read_text()

    # Find __init__ and add missing assignments
    content = re.sub(
        r'(def __init__\(self, \*\*kwargs: Any\) -> None:.*?""")',
        r"""\1
        self.max_size = kwargs.get("max_size", 1000)
        self.delay_seconds = kwargs.get("delay_seconds", 0.0)""",
        content,
        flags=re.DOTALL,
    )

    file.write_text(content)
    print(f"✅ Fixed {file}")


def fix_file_memory():
    """Fix FileMemoryAdapter."""
    file = Path("hexdag/builtin/adapters/memory/file_memory_adapter.py")
    content = file.read_text()

    # Find the broken line and fix it
    content = re.sub(
        r'def __init__\(self, \*\*kwargs: Any\) -> None:.*?""".*?\n',
        '''def __init__(
        self,
        file_path: str = "memory.json",
        file_format: str = "json",
        auto_save: bool = True,
        backup_dir: str | None = None,
        **kwargs: Any
    ) -> None:
        """Initialize file memory adapter.

        Parameters
        ----------
        file_path : str, default="memory.json"
            Path to memory file
        file_format : str, default="json"
            File format (json or yaml)
        auto_save : bool, default=True
            Whether to auto-save on changes
        backup_dir : str | None, default=None
            Directory for backups
        """
        self.file_path = file_path
        self.file_format = file_format
        self.auto_save = auto_save
        self.backup_dir = backup_dir
''',
        content,
        flags=re.DOTALL,
        count=1,
    )

    file.write_text(content)
    print(f"✅ Fixed {file}")


def fix_sqlite_memory():
    """Fix SQLiteMemoryAdapter."""
    file = Path("hexdag/builtin/adapters/memory/sqlite_memory_adapter.py")
    content = file.read_text()

    content = re.sub(
        r'(def __init__\(self, \*\*kwargs: Any\) -> None:.*?""")',
        r"""\1
        self.db_path = kwargs.get("db_path", ":memory:")
        self.table_name = kwargs.get("table_name", "memory")
        self.auto_init = kwargs.get("auto_init", True)""",
        content,
        flags=re.DOTALL,
    )

    file.write_text(content)
    print(f"✅ Fixed {file}")


def fix_mock_database():
    """Fix MockDatabaseAdapter."""
    file = Path("hexdag/builtin/adapters/mock/mock_database.py")
    content = file.read_text()

    content = re.sub(
        r'(def __init__\(self, \*\*kwargs: Any\) -> None:.*?""")',
        r"""\1
        self.enable_sample_data = kwargs.get("enable_sample_data", True)
        self.delay_seconds = kwargs.get("delay_seconds", 0.0)""",
        content,
        flags=re.DOTALL,
    )

    file.write_text(content)
    print(f"✅ Fixed {file}")


def fix_mock_llm():
    """Fix MockLLM."""
    file = Path("hexdag/builtin/adapters/mock/mock_llm.py")
    content = file.read_text()

    content = re.sub(
        r"(# Process responses.*?self\.responses = \[.*?\])",
        r"""\1

        self.delay_seconds = kwargs.get("delay_seconds", 0.0)""",
        content,
        flags=re.DOTALL,
    )

    file.write_text(content)
    print(f"✅ Fixed {file}")


def fix_mock_tool_adapter():
    """Fix MockToolAdapter."""
    file = Path("hexdag/builtin/adapters/mock/mock_tool_adapter.py")
    content = file.read_text()

    content = re.sub(
        r'(def __init__\(self, \*\*kwargs: Any\) -> None:.*?""")',
        r"""\1
        self.responses = kwargs.get("responses", {})
        self.raise_on_unknown = kwargs.get("raise_on_unknown", False)
        self.default_response = kwargs.get("default_response", "Mock tool response")""",
        content,
        flags=re.DOTALL,
    )

    file.write_text(content)
    print(f"✅ Fixed {file}")


def fix_mock_tool_router():
    """Fix MockToolRouter."""
    file = Path("hexdag/builtin/adapters/mock/mock_tool_router.py")
    content = file.read_text()

    content = re.sub(
        r'(def __init__\(self, \*\*kwargs: Any\) -> None:.*?""")',
        r"""\1
        self.available_tools = kwargs.get("available_tools", [])
        self.delay_seconds = kwargs.get("delay_seconds", 0.0)
        self.raise_on_unknown_tool = kwargs.get("raise_on_unknown_tool", False)""",
        content,
        flags=re.DOTALL,
    )

    file.write_text(content)
    print(f"✅ Fixed {file}")


def fix_local_secret_adapter():
    """Fix LocalSecretAdapter."""
    file = Path("hexdag/builtin/adapters/secret/local_secret_adapter.py")
    content = file.read_text()

    content = re.sub(
        r'(def __init__\(self, \*\*kwargs: Any\) -> None:.*?""")',
        r"""\1
        self.env_prefix = kwargs.get("env_prefix", "")
        self.allow_empty = kwargs.get("allow_empty", False)""",
        content,
        flags=re.DOTALL,
    )

    file.write_text(content)
    print(f"✅ Fixed {file}")


def main():
    """Fix all files with missing init params."""
    print("Fixing missing __init__ parameters...")

    fix_in_memory_memory()
    fix_file_memory()
    fix_sqlite_memory()
    fix_mock_database()
    fix_mock_llm()
    fix_mock_tool_adapter()
    fix_mock_tool_router()
    fix_local_secret_adapter()

    print("\n✅ All files fixed!")


if __name__ == "__main__":
    main()
