"""Pandas transform node with multi-operation support for ETL pipelines."""

import asyncio
import importlib
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Literal

import pandas as pd
from hexdag.kernel.domain.dag import NodeSpec
from pydantic import BaseModel

from .base_node_factory import BaseNodeFactory

# Convention: Pandas operation types for dropdown menus in Studio UI
PandasOperationType = Literal["transform", "map", "filter", "assign"]


class PandasOperation(BaseModel):
    """Single pandas operation configuration."""

    type: PandasOperationType = "transform"
    """Operation type: 'transform', 'map', 'filter', 'assign'"""

    method: str | None = None
    """Pandas method path (e.g., 'pandas.DataFrame.groupby', 'pandas.merge')"""

    args: list[Any] | None = None
    """Positional arguments for the operation"""

    kwargs: dict[str, Any] | None = None
    """Keyword arguments for the operation"""

    columns: dict[str, str] | None = None
    """Column mappings (for 'map' or 'rename' operations)"""

    condition: str | None = None
    """Filter condition expression (for 'filter' operations)"""


class PandasTransformNode(BaseNodeFactory):
    """Node factory for multi-operation pandas transforms.

    Executes a sequence of pandas operations on DataFrames, supporting:
    - Chained transformations
    - Multiple input DataFrames
    - Artifact storage integration
    - Complex data cleaning and enrichment

    Examples
    --------
    YAML pipeline::

        - kind: pandas_transform_node
          metadata:
            name: clean_and_aggregate
          spec:
            input_artifacts:
              - slot: raw_customers
                key: customers_v1
              - slot: raw_transactions
                key: transactions_v1
            operations:
              # Operation 1: Join DataFrames
              - type: transform
                method: pandas.merge
                args:
                  - {{input_artifacts[0]}}
                  - {{input_artifacts[1]}}
                kwargs:
                  on: customer_id
                  how: left

              # Operation 2: Drop missing values
              - type: transform
                method: pandas.DataFrame.dropna
                kwargs:
                  subset: [customer_id, amount]

              # Operation 3: Calculate new column
              - type: transform
                method: pandas.DataFrame.assign
                kwargs:
                  revenue_tier: |
                    lambda df: pd.cut(
                      df['amount'],
                      bins=[0, 100, 500, float('inf')],
                      labels=['Low', 'Medium', 'High']
                    )

              # Operation 4: Rename columns
              - type: map
                columns:
                  transaction_id: txn_id
                  customer_id: cust_id
                  amount: total_amount

              # Operation 5: Filter rows
              - type: filter
                condition: "{{ df['amount'] > 0 }}"

              # Operation 6: Group and aggregate
              - type: transform
                method: pandas.DataFrame.groupby
                args:
                  - customer_id
                kwargs:
                  as_index: false

              # Operation 7: Calculate aggregations
              - type: transform
                method: pandas.DataFrame.agg
                kwargs:
                  amount: ['count', 'sum', 'mean']
                  customer_id: 'count'

            output_artifact:
              slot: enriched_customers
              key: enriched_v1
              format: parquet
              compression: snappy
    """

    # Studio UI metadata
    _hexdag_icon = "Table"
    _hexdag_color = "#8b5cf6"  # violet-500

    def __call__(
        self,
        name: str,
        operations: list[dict[str, Any]],
        input_artifacts: list[dict[str, Any]] | None = None,
        output_artifact: dict[str, Any] | None = None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a pandas transform node specification.

        Parameters
        ----------
        name : str
            Node name
        operations : list[dict]
            List of pandas operation configurations
        input_artifacts : list[dict], optional
            Artifact references for input DataFrames
        output_artifact : dict, optional
            Artifact configuration for output DataFrame
        deps : list[str], optional
            Dependency node names
        **kwargs : Any
            Additional node parameters

        Returns
        -------
        NodeSpec
            Node specification ready for execution
        """
        # Convert operation dicts to Pydantic models for validation
        operation_models = [PandasOperation(**op) for op in operations]

        # Create wrapped function
        wrapped_fn = self._create_transform_function(name, operation_models, input_artifacts, output_artifact)

        # Define input schema
        if input_artifacts:
            input_schema = {"input_data": dict, "**ports": dict}
        else:
            input_schema = {"input_data": dict, "**ports": dict}

        # Define output schema
        output_schema = {"output": dict}

        input_model = self.create_pydantic_model(f"{name}Input", input_schema)
        output_model = self.create_pydantic_model(f"{name}Output", output_schema)

        # Store parameters
        node_params = {
            "operations": operations,
            "input_artifacts": input_artifacts,
            "output_artifact": output_artifact,
            **kwargs,
        }

        return NodeSpec(
            name=name,
            fn=wrapped_fn,
            in_model=input_model,
            out_model=output_model,
            deps=frozenset(deps or []),
            params=node_params,
        )

    def _create_transform_function(
        self,
        name: str,
        operations: list[PandasOperation],
        input_artifacts: list[dict[str, Any]] | None,
        output_artifact: dict[str, Any] | None,
    ) -> Callable[..., dict[str, Any]]:
        """Create the wrapped transformation function.

        Parameters
        ----------
        name : str
            Node name
        operations : list[PandasOperation]
            Operations to execute
        input_artifacts : list[dict], optional
            Input artifact references
        output_artifact : dict, optional
            Output artifact configuration

        Returns
        -------
        Callable
            Async function that executes the transformation
        """

        async def wrapped_fn(input_data: Any, **ports: Any) -> dict[str, Any]:
            """Execute pandas transformation operations."""
            # Initialize result DataFrame
            df = None

            # Load input artifacts if specified
            if input_artifacts:
                artifact_store = ports.get("artifact_store")
                if not artifact_store:
                    raise ValueError("artifact_store port required when using input_artifacts")

                loaded_dfs = []
                for artifact_ref in input_artifacts:
                    slot = artifact_ref.get("slot")
                    key = artifact_ref.get("key")
                    format = artifact_ref.get("format")

                    if not slot or not key:
                        raise ValueError(f"Invalid artifact reference: {artifact_ref}")

                    # Load from artifact store
                    df_loaded = await artifact_store.read(name=slot, key=key, format=format)
                    loaded_dfs.append(df_loaded)

                # Start with first DataFrame if available
                if loaded_dfs:
                    df = loaded_dfs[0]
            else:
                # Use input_data directly
                if isinstance(input_data, dict) and "data" in input_data:
                    df = input_data["data"]
                else:
                    df = input_data

            if df is None:
                raise ValueError("No input DataFrame available")

            if not isinstance(df, pd.DataFrame):
                # Try to convert to DataFrame
                try:
                    df = pd.DataFrame(df)
                except Exception as e:
                    raise ValueError(f"Could not convert input to DataFrame: {e}")

            # Execute operations sequentially
            for _i, op in enumerate(operations):
                df = await self._execute_operation(df, op, loaded_dfs if input_artifacts else [df])

            # Store output artifact if specified
            result = {"output": df}

            if output_artifact:
                artifact_store = ports.get("artifact_store")
                if not artifact_store:
                    raise ValueError("artifact_store port required when using output_artifact")

                slot = output_artifact.get("slot")
                key = output_artifact.get("key", f"{name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
                format = output_artifact.get("format", "pickle")
                compression = output_artifact.get("compression")
                metadata = output_artifact.get("metadata")

                if not slot:
                    raise ValueError("output_artifact must specify 'slot' name")

                # Write to artifact store
                artifact_info = await artifact_store.write(
                    name=slot,
                    key=key,
                    data=df,
                    format=format,
                    compression=compression,
                    metadata=metadata,
                )

                result["artifact_info"] = asdict(artifact_info)
                result["records"] = len(df)

            return result

        # Preserve function metadata
        wrapped_fn.__name__ = f"pandas_transform_{name}"
        wrapped_fn.__doc__ = f"Multi-operation pandas transform: {name}"

        return wrapped_fn

    async def _execute_operation(
        self, df: pd.DataFrame, op: PandasOperation, input_dfs: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """Execute a single pandas operation.

        Parameters
        ----------
        df : pd.DataFrame
            Current DataFrame
        op : PandasOperation
            Operation to execute
        input_dfs : list[pd.DataFrame]
            Available input DataFrames

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame
        """
        op_type = op.type or "transform"

        if op_type == "transform":
            return await self._execute_transform(df, op, input_dfs)

        if op_type == "map":
            return await self._execute_map(df, op)

        if op_type == "filter":
            return await self._execute_filter(df, op)

        if op_type == "assign":
            return await self._execute_assign(df, op)

        raise ValueError(f"Unknown operation type: {op_type}")

    async def _execute_transform(
        self, df: pd.DataFrame, op: PandasOperation, input_dfs: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """Execute a transform operation (calls a pandas method).

        Parameters
        ----------
        df : pd.DataFrame
            Current DataFrame
        op : PandasOperation
            Operation configuration
        input_dfs : list[pd.DataFrame]
            Available input DataFrames

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame
        """
        if not op.method:
            raise ValueError("Transform operation requires 'method' parameter")

        # Resolve method
        method = self._resolve_method(op.method)

        # Prepare arguments (resolve template expressions)
        args = []
        if op.args:
            args.extend(self._resolve_arg(arg, df, input_dfs) for arg in op.args)

        # Prepare keyword arguments
        kwargs = {}
        if op.kwargs:
            for k, v in op.kwargs.items():
                kwargs[k] = self._resolve_arg(v, df, input_dfs)

        # Execute method (handle both sync and async)
        if asyncio.iscoroutinefunction(method):
            result = await method(df, *args, **kwargs)
        else:
            result = method(df, *args, **kwargs)

        return result

    async def _execute_map(self, df: pd.DataFrame, op: PandasOperation) -> pd.DataFrame:
        """Execute a map operation (column rename/mapping).

        Parameters
        ----------
        df : pd.DataFrame
            Current DataFrame
        op : PandasOperation
            Operation configuration with columns mapping

        Returns
        -------
        pd.DataFrame
            DataFrame with renamed columns
        """
        if not op.columns:
            return df

        return df.rename(columns=op.columns)

    async def _execute_filter(self, df: pd.DataFrame, op: PandasOperation) -> pd.DataFrame:
        """Execute a filter operation.

        Parameters
        ----------
        df : pd.DataFrame
            Current DataFrame
        op : PandasOperation
            Operation configuration with filter condition

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame
        """
        if not op.condition:
            return df

        # Evaluate condition
        # Note: This is a simplified implementation - production code should validate
        # the condition for security
        condition_result = self._resolve_arg(op.condition, df, [df])

        return df[condition_result]

    async def _execute_assign(self, df: pd.DataFrame, op: PandasOperation) -> pd.DataFrame:
        """Execute an assign operation (add new columns).

        Parameters
        ----------
        df : pd.DataFrame
            Current DataFrame
        op : PandasOperation
            Operation configuration

        Returns
        -------
        pd.DataFrame
            DataFrame with new columns
        """
        if not op.kwargs:
            return df

        # Prepare new column assignments
        new_cols = {}
        for col_name, col_expr in op.kwargs.items():
            new_cols[col_name] = self._resolve_arg(col_expr, df, [df])

        return df.assign(**new_cols)

    def _resolve_method(self, method_path: str) -> Callable:
        """Resolve a method from a path string or return callable directly.

        Parameters
        ----------
        method_path : str
            Path like "pandas.DataFrame.groupby" or "pandas.merge"

        Returns
        -------
        Callable
            The resolved method
        """
        # Already callable
        if callable(method_path):
            return method_path

        # Parse module path
        if "." not in method_path:
            raise ValueError(f"Method path must contain '.', got: {method_path}")

        try:
            # Handle pandas class paths like pandas.DataFrame.sort_values
            if method_path.startswith("pandas."):
                parts = method_path.split(".")
                if len(parts) >= 3 and parts[1] == "DataFrame":
                    # It's a DataFrame method: pandas.DataFrame.method_name
                    method_name = parts[2]
                    return getattr(pd.DataFrame, method_name)
                # It's a module-level function like pandas.merge
                module_path = ".".join(parts[:-1])
                attr_path = parts[-1]
                module = importlib.import_module(module_path)
                method = getattr(module, attr_path)

                if not callable(method):
                    raise ValueError(f"'{method_path}' is not callable")

                return method
            # Standard module attribute resolution
            module_path, attr_path = method_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            method = getattr(module, attr_path)

            if not callable(method):
                raise ValueError(f"'{method_path}' is not callable")

            return method
        except Exception as e:
            raise ValueError(f"Could not resolve method '{method_path}': {e}") from e

    def _resolve_arg(self, arg: Any, df: pd.DataFrame, input_dfs: list[pd.DataFrame]) -> Any:
        """Resolve an argument value (handles templates and expressions).

        Parameters
        ----------
        arg : Any
            Argument value or template expression
        df : pd.DataFrame
            Current DataFrame for context
        input_dfs : list[pd.DataFrame]
            All input DataFrames

        Returns
        -------
        Any
            Resolved argument value
        """
        # If it's a string template expression
        if isinstance(arg, str) and "{{" in arg and "}}" in arg:
            # Parse template expression
            import re

            pattern = r"\{\{\s*(.+?)\s*\}\}"
            match = re.search(pattern, arg)

            if match:
                expr = match.group(1)

                # Handle special variables
                if expr == "df":
                    return df
                if expr.startswith("input_artifacts["):
                    # Extract index
                    idx_match = re.search(r"input_artifacts\[(\d+)\]", expr)
                    if idx_match:
                        idx = int(idx_match.group(1))
                        if 0 <= idx < len(input_dfs):
                            return input_dfs[idx]
                        raise IndexError(f"input_artifacts[{idx}] out of range")

                # Try to evaluate as Python expression
                try:
                    # Safe evaluation - limited scope
                    scope = {"df": df, "input_artifacts": input_dfs, "pd": pd}
                    return eval(expr, {"__builtins__": {}}, scope)
                except Exception:
                    # Return as-is if evaluation fails
                    return arg

        # If it's a dict with lambda expression
        if isinstance(arg, dict):
            resolved = {}
            for k, v in arg.items():
                resolved[k] = self._resolve_arg(v, df, input_dfs)
            return resolved

        # Return as-is
        return arg
