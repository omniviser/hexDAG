"""MySQL adapter plugin for hexDAG framework.

Plugin Structure
----------------
::

    mysql_adapter/
    ├── __init__.py          # This file - re-exports from adapters/
    ├── adapters/            # Adapter implementations
    │   └── mysql.py         # MySQLAdapter (document-store interface)
    └── tests/               # Test files

Available Adapters
------------------
- ``MySQLAdapter``: Document-store style MySQL adapter with JSON storage

Note: For standard SQL operations (raw queries, table schemas), use
``hexdag_plugins.storage.adapters.sql.MySQLAdapter`` instead.
"""

from hexdag_plugins.mysql_adapter.adapters import MySQLAdapter

__all__ = ["MySQLAdapter"]
