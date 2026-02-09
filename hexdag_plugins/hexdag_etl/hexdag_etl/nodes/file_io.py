"""File I/O nodes for reading and writing data files.

These nodes provide file-based input and output for ETL pipelines,
supporting CSV, Parquet, JSON, and Excel formats.
"""

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from hexdag.core.domain.dag import NodeSpec
from pydantic import BaseModel

from .base_node_factory import BaseNodeFactory

# Convention: File format options for dropdown menus in Studio UI
FileFormat = Literal["csv", "parquet", "json", "jsonl", "excel", "feather", "pickle"]


class FileReaderOutput(BaseModel):
    """Output model for FileReaderNode."""

    data: Any  # DataFrame as dict for serialization
    rows: int
    columns: list[str]
    file_path: str


class FileWriterOutput(BaseModel):
    """Output model for FileWriterNode."""

    file_path: str
    rows: int
    format: str
    success: bool


class FileReaderNode(BaseNodeFactory):
    """Node for reading data files into DataFrames.

    Supports multiple file formats:
    - CSV (.csv)
    - Parquet (.parquet)
    - JSON (.json, .jsonl)
    - Excel (.xlsx, .xls)

    Examples
    --------
    YAML pipeline::

        - kind: etl:file_reader_node
          metadata:
            name: load_customers
          spec:
            file_path: data/customers.csv
            format: csv
            options:
              sep: ","
              encoding: utf-8
          dependencies: []

        - kind: etl:file_reader_node
          metadata:
            name: load_transactions
          spec:
            file_path: data/transactions.parquet
            format: parquet
          dependencies: []

        - kind: etl:file_reader_node
          metadata:
            name: load_products
          spec:
            file_path: data/products.json
            format: json
            options:
              orient: records
          dependencies: []
    """

    # Studio UI metadata
    _hexdag_icon = "FileInput"
    _hexdag_color = "#10b981"  # emerald-500

    def __call__(
        self,
        name: str,
        file_path: str,
        format: FileFormat | None = None,
        options: dict[str, Any] | None = None,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a file reader node specification.

        Parameters
        ----------
        name : str
            Node name
        file_path : str
            Path to the input file (relative to workspace or absolute)
        format : FileFormat, optional
            File format: 'csv', 'parquet', 'json', 'jsonl', 'excel', 'feather', 'pickle'
            Auto-detected from extension if not specified
        options : dict, optional
            Additional options passed to pandas read function
        deps : list[str], optional
            Dependency node names
        **kwargs : Any
            Additional node parameters

        Returns
        -------
        NodeSpec
            Node specification ready for execution
        """
        # Auto-detect format from file extension if not specified
        if format is None:
            format = self._detect_format(file_path)

        # Create wrapped function
        wrapped_fn = self._create_reader_function(name, file_path, format, options or {})

        # Define schemas
        input_schema = {"input_data": dict | None}
        output_model = FileReaderOutput

        input_model = self.create_pydantic_model(f"{name}Input", input_schema)

        # Store parameters
        node_params = {
            "file_path": file_path,
            "format": format,
            "options": options,
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

    def _detect_format(self, file_path: str) -> str:
        """Detect file format from extension."""
        path = Path(file_path)
        ext = path.suffix.lower()

        format_map = {
            ".csv": "csv",
            ".parquet": "parquet",
            ".pq": "parquet",
            ".json": "json",
            ".jsonl": "jsonl",
            ".xlsx": "excel",
            ".xls": "excel",
            ".feather": "feather",
            ".pickle": "pickle",
            ".pkl": "pickle",
        }

        if ext not in format_map:
            raise ValueError(f"Unknown file format for extension '{ext}'. Supported: {list(format_map.keys())}")

        return format_map[ext]

    def _create_reader_function(
        self,
        name: str,
        file_path: str,
        format: str,
        options: dict[str, Any],
    ) -> Any:
        """Create the file reading function."""

        async def read_file(input_data: Any = None) -> dict[str, Any]:
            """Read data file into DataFrame."""
            # Resolve file path
            path = Path(file_path)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            # Read based on format
            if format == "csv":
                df = pd.read_csv(path, **options)
            elif format == "parquet":
                df = pd.read_parquet(path, **options)
            elif format == "json":
                df = pd.read_json(path, **options)
            elif format == "jsonl":
                df = pd.read_json(path, lines=True, **options)
            elif format == "excel":
                df = pd.read_excel(path, **options)
            elif format == "feather":
                df = pd.read_feather(path, **options)
            elif format == "pickle":
                df = pd.read_pickle(path, **options)
            else:
                raise ValueError(f"Unsupported format: {format}")

            return {
                "data": df,  # Keep as DataFrame for downstream nodes
                "rows": len(df),
                "columns": df.columns.tolist(),
                "file_path": str(path.absolute()),
            }

        read_file.__name__ = f"file_reader_{name}"
        read_file.__doc__ = f"Read file: {file_path}"

        return read_file


class FileWriterNode(BaseNodeFactory):
    """Node for writing DataFrames to files.

    Supports multiple file formats:
    - CSV (.csv)
    - Parquet (.parquet)
    - JSON (.json, .jsonl)
    - Excel (.xlsx)

    Examples
    --------
    YAML pipeline::

        - kind: etl:file_writer_node
          metadata:
            name: save_results
          spec:
            file_path: output/results.parquet
            format: parquet
            options:
              compression: snappy
          dependencies:
            - transform_data

        - kind: etl:file_writer_node
          metadata:
            name: export_csv
          spec:
            file_path: output/report.csv
            format: csv
            options:
              index: false
          dependencies:
            - transform_data
    """

    # Studio UI metadata
    _hexdag_icon = "FileOutput"
    _hexdag_color = "#f59e0b"  # amber-500

    def __call__(
        self,
        name: str,
        file_path: str,
        format: FileFormat | None = None,
        input_key: str = "data",
        options: dict[str, Any] | None = None,
        create_dirs: bool = True,
        deps: list[str] | None = None,
        **kwargs: Any,
    ) -> NodeSpec:
        """Create a file writer node specification.

        Parameters
        ----------
        name : str
            Node name
        file_path : str
            Path for the output file
        format : FileFormat, optional
            File format: 'csv', 'parquet', 'json', 'jsonl', 'excel', 'feather', 'pickle'
            Auto-detected from extension if not specified
        input_key : str
            Key in input data containing the DataFrame (default: 'data')
        options : dict, optional
            Additional options passed to pandas write function
        create_dirs : bool
            Create parent directories if they don't exist (default: True)
        deps : list[str], optional
            Dependency node names
        **kwargs : Any
            Additional node parameters

        Returns
        -------
        NodeSpec
            Node specification ready for execution
        """
        # Auto-detect format from file extension if not specified
        if format is None:
            format = self._detect_format(file_path)

        # Create wrapped function
        wrapped_fn = self._create_writer_function(name, file_path, format, input_key, options or {}, create_dirs)

        # Define schemas
        input_schema = {"input_data": dict}
        output_model = FileWriterOutput

        input_model = self.create_pydantic_model(f"{name}Input", input_schema)

        # Store parameters
        node_params = {
            "file_path": file_path,
            "format": format,
            "input_key": input_key,
            "options": options,
            "create_dirs": create_dirs,
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

    def _detect_format(self, file_path: str) -> str:
        """Detect file format from extension."""
        path = Path(file_path)
        ext = path.suffix.lower()

        format_map = {
            ".csv": "csv",
            ".parquet": "parquet",
            ".pq": "parquet",
            ".json": "json",
            ".jsonl": "jsonl",
            ".xlsx": "excel",
            ".feather": "feather",
            ".pickle": "pickle",
            ".pkl": "pickle",
        }

        if ext not in format_map:
            raise ValueError(f"Unknown file format for extension '{ext}'. Supported: {list(format_map.keys())}")

        return format_map[ext]

    def _create_writer_function(
        self,
        name: str,
        file_path: str,
        format: str,
        input_key: str,
        options: dict[str, Any],
        create_dirs: bool,
    ) -> Any:
        """Create the file writing function."""

        async def write_file(input_data: Any) -> dict[str, Any]:
            """Write DataFrame to file."""
            # Extract DataFrame from input
            if isinstance(input_data, dict):
                df = input_data.get(input_key)
                if df is None:
                    # Try to find a DataFrame in the input
                    for value in input_data.values():
                        if isinstance(value, pd.DataFrame):
                            df = value
                            break
                        if isinstance(value, dict) and "data" in value:
                            df = value["data"]
                            break
            elif isinstance(input_data, pd.DataFrame):
                df = input_data
            else:
                df = input_data

            if df is None:
                raise ValueError(f"No DataFrame found in input. Expected key: '{input_key}'")

            if not isinstance(df, pd.DataFrame):
                try:
                    df = pd.DataFrame(df)
                except Exception as e:
                    raise ValueError(f"Could not convert input to DataFrame: {e}")

            # Resolve file path and create directories
            path = Path(file_path)

            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            # Write based on format
            if format == "csv":
                df.to_csv(path, **options)
            elif format == "parquet":
                df.to_parquet(path, **options)
            elif format == "json":
                df.to_json(path, **options)
            elif format == "jsonl":
                df.to_json(path, orient="records", lines=True, **options)
            elif format == "excel":
                df.to_excel(path, **options)
            elif format == "feather":
                df.to_feather(path, **options)
            elif format == "pickle":
                df.to_pickle(path, **options)
            else:
                raise ValueError(f"Unsupported format: {format}")

            return {
                "file_path": str(path.absolute()),
                "rows": len(df),
                "format": format,
                "success": True,
            }

        write_file.__name__ = f"file_writer_{name}"
        write_file.__doc__ = f"Write file: {file_path}"

        return write_file
