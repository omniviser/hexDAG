"""Test the pandas transform step by step."""

import sys

sys.path.insert(0, "/Users/jankwapisz/Documents/Praca/Omniviser/hexdag")
sys.path.insert(0, "/Users/jankwapisz/Documents/Praca/Omniviser/hexdag/hexdag_plugins")

import pandas as pd

from hexdag_etl.nodes.pandas_transform import PandasTransformNode

# Create the node
node_factory = PandasTransformNode()

# Build operations
operations = [
    {
        "type": "transform",
        "method": "pandas.DataFrame.sort_values",
        "kwargs": {"by": "amount", "ascending": False},
    },
    {
        "type": "transform",
        "method": "pandas.DataFrame.assign",
        "kwargs": {"amount_doubled": "{{ lambda df: df['amount'] * 2 }}"},
    },
    {"type": "transform", "method": "pandas.DataFrame.head", "args": [3]},
]

# Create node spec
node_spec = node_factory(name="test_transform", operations=operations)

# Test input
df = pd.DataFrame(
    {
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "name": ["Alice", "Bob", "Carol", "David", "Emma"],
        "amount": [150.0, 299.99, 150.0, 29.99, 299.99],
        "category": ["A", "B", "A", "C", "B"],
    }
)

print("Input DataFrame:")
print(df)
print()

# Try to run the function
import asyncio


async def test():
    result = await node_spec.fn({"data": df})
    print("Output:")
    print(result)
    return result


asyncio.run(test())
