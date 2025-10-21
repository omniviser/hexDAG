import pathlib

import pytest

from hexdag.adapters.database.csv.csv_adapter import CsvAdapter


@pytest.fixture
def csv_dir(tmp_path):
    """Create temporary CSV files for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create users.csv
    with pathlib.Path(data_dir / "users.csv").open("w", newline="") as f:
        f.write("id,name,email\n")
        f.write("1,Alice,alice@test.com\n")
        f.write("2,Bob,bob@test.com\n")

    return data_dir


async def test_get_table_schemas(csv_dir):
    """Test schema detection from CSV files."""
    adapter = CsvAdapter(csv_dir)
    schemas = await adapter.get_table_schemas()

    assert len(schemas) == 1
    assert schemas[0].name == "users"
    assert len(schemas[0].columns) == 3
    assert [col.name for col in schemas[0].columns] == ["id", "name", "email"]


async def test_query_with_filters(csv_dir):
    """Test querying with filters."""
    adapter = CsvAdapter(csv_dir)
    rows = []
    async for row in adapter.query("users", filters={"name": "Alice"}):
        rows.append(row)

    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"
    assert rows[0]["email"] == "alice@test.com"
