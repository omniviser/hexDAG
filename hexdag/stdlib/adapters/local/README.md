# Local Adapters

Local, in-process adapter implementations that don't require external services.

## Available Adapters

### InMemoryMemory

Simple in-memory key-value storage that implements the Memory port interface.

```python
from hexdag.stdlib.adapters.local import InMemoryMemory

# Create an instance
memory = InMemoryMemory()

# Store and retrieve data
await memory.aset("key", {"data": "value"})
data = await memory.aget("key")

# With configuration
memory = InMemoryMemory(
    max_size=100,  # Limit to 100 items
    delay_seconds=0.0  # No artificial delay
)
```

#### Features

- **In-process storage**: No external dependencies
- **Access history tracking**: Track all get/set operations
- **Size limits**: Optional maximum number of items
- **Testing utilities**: Clear, reset, and inspection methods

#### Configuration

```python
memory = InMemoryMemory(
    max_size=1000,      # Maximum items (None for unlimited)
    delay_seconds=0.0   # Artificial delay for testing
)
```

#### Testing Methods

```python
# Get storage statistics
size = memory.size()
keys = memory.keys()

# Access history
history = memory.get_access_history()

# Clear data but keep history
memory.clear()

# Reset everything
memory.reset()
```
