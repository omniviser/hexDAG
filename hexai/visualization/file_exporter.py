"""File export and display functionality for DAG visualization."""

import contextlib
import logging
import pathlib
import platform
import shutil
import subprocess  # nosec B404
import tempfile
import threading
import time

logger = logging.getLogger(__name__)


class FileExporter:
    """Handle file operations and display for visualizations."""

    def render_to_file(self, dot_content: str, output_path: str, format: str = "png") -> str:
        """Render DOT content to file using Graphviz.

        Args:
            dot_content: DOT format string
            output_path: Path where to save the rendered graph (without extension)
            format: Output format ('png', 'svg', 'pdf', etc.)

        Returns:
            Path to the rendered file

        Raises:
            RuntimeError: If rendering fails.
        """
        try:
            # Create a temporary DOT file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as temp_file:
                temp_file.write(dot_content)
                temp_dot_path = temp_file.name

            # Use dot command to render
            output_file = f"{output_path}.{format}"
            # nosec B607, B603 - dot is a trusted system command for Graphviz
            subprocess.run(  # nosec B607, B603
                ["dot", "-T" + format, "-o", output_file, temp_dot_path],
                capture_output=True,
                text=True,
                check=True,
            )

            # Clean up temporary file
            with contextlib.suppress(OSError):
                pathlib.Path(temp_dot_path).unlink()

            return output_file
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to render graph: {e.stderr}") from e
        except FileNotFoundError:
            raise ImportError(
                "Graphviz 'dot' command not found. Please install Graphviz."
            ) from None
        except Exception as e:
            raise RuntimeError(f"Failed to render graph: {e}") from e

    def show(self, dot_content: str, title: str = "Pipeline DAG") -> None:
        """Display visualization in default viewer.

        Args:
            dot_content: DOT format string
            title: Title for the display window

        Raises:
            RuntimeError: If showing graph fails
        """
        try:
            # Create a temporary DOT file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as temp_file:
                temp_file.write(dot_content)
                temp_dot_path = temp_file.name

            # Use dot command to create a temporary image
            temp_image_path = temp_dot_path.replace(".dot", ".png")
            # nosec B607, B603 - dot is a trusted system command for Graphviz
            subprocess.run(  # nosec B607, B603
                ["dot", "-Tpng", "-o", temp_image_path, temp_dot_path],
                capture_output=True,
                text=True,
                check=True,
            )

            # Open the image with the default viewer
            self._open_with_default_viewer(temp_image_path)

            # Clean up files after a delay
            self._schedule_cleanup([temp_dot_path, temp_image_path])

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to show graph: {e.stderr}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to show graph: {e}") from e

    def generate_html(self, dot_content: str, title: str) -> str:
        """Generate interactive HTML visualization.

        Args:
            dot_content: DOT format string
            title: Title for the HTML page

        Returns:
            HTML content as string
        """
        # Generate SVG from DOT content
        svg_content = self._generate_svg(dot_content)

        # Create HTML with embedded SVG
        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .graph-container {{
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            margin: 20px auto;
            max-width: 95%;
            overflow: auto;
        }}
        .controls {{
            text-align: center;
            margin: 20px;
        }}
        button {{
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            margin: 0 5px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        button:hover {{
            background-color: #45a049;
        }}
        svg {{
            width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="controls">
        <button onclick="zoomIn()">Zoom In</button>
        <button onclick="zoomOut()">Zoom Out</button>
        <button onclick="resetZoom()">Reset</button>
        <button onclick="downloadSVG()">Download SVG</button>
    </div>
    <div class="graph-container" id="graphContainer">
        {svg_content}
    </div>
    <script>
        let scale = 1;
        const svg = document.querySelector('svg');

        function zoomIn() {{
            scale *= 1.2;
            svg.style.transform = `scale(${{scale}})`;
        }}

        function zoomOut() {{
            scale *= 0.8;
            svg.style.transform = `scale(${{scale}})`;
        }}

        function resetZoom() {{
            scale = 1;
            svg.style.transform = `scale(${{scale}})`;
        }}

        function downloadSVG() {{
            const svgData = svg.outerHTML;
            const blob = new Blob([svgData], {{type: 'image/svg+xml'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = '{title}.svg';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>"""

        return html_template.format(title=title, svg_content=svg_content)

    def save_html(self, html_content: str, output_path: str) -> None:
        """Save HTML content to file.

        Args:
            html_content: HTML content to save
            output_path: Path where to save the HTML file
        """
        output = pathlib.Path(output_path)
        output.write_text(html_content, encoding="utf-8")

    def cleanup_temp_files(self, files: list[str]) -> None:
        """Clean up temporary files.

        Args:
            files: List of file paths to clean up
        """
        for file_path in files:
            with contextlib.suppress(OSError):
                pathlib.Path(file_path).unlink()

    def _generate_svg(self, dot_content: str) -> str:
        """Generate SVG from DOT content.

        Args:
            dot_content: DOT format string

        Returns:
            SVG content as string
        """
        try:
            # Create a temporary DOT file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as temp_file:
                temp_file.write(dot_content)
                temp_dot_path = temp_file.name

            # Use dot command to generate SVG
            # nosec B607, B603 - dot is a trusted system command for Graphviz
            result = subprocess.run(  # nosec B607, B603
                ["dot", "-Tsvg", temp_dot_path],
                capture_output=True,
                text=True,
                check=True,
            )

            # Clean up temporary file
            with contextlib.suppress(OSError):
                pathlib.Path(temp_dot_path).unlink()

            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error("Failed to generate SVG: %s", e.stderr)
            return f"<p>Error generating visualization: {e.stderr}</p>"
        except Exception as e:
            logger.error("Failed to generate SVG: %s", e)
            return f"<p>Error generating visualization: {e}</p>"

    def _open_with_default_viewer(self, file_path: str) -> None:
        """Open file with system's default viewer.

        Args:
            file_path: Path to the file to open
        """
        # nosec B607, B603 - open/xdg-open are trusted system commands for viewing files
        system_platform = platform.system()
        if system_platform == "Darwin":
            viewer_cmd = "open"
        elif system_platform == "Linux":
            viewer_cmd = "xdg-open"
        else:
            viewer_cmd = None

        if viewer_cmd and shutil.which(viewer_cmd):
            subprocess.run([viewer_cmd, file_path], check=False)  # nosec B607, B603
        else:
            help_msg = (
                f"No default image viewer found for platform '{system_platform}'.\n"
                f"For macOS, please ensure the 'open' command is available.\n"
                f"For Linux, please ensure the 'xdg-open' command is installed.\n"
                f"You can manually open the file located at: {file_path}"
            )
            logger.error(help_msg)

    def _schedule_cleanup(self, files: list[str], delay: int = 2) -> None:
        """Schedule cleanup of temporary files after a delay.

        Args:
            files: List of file paths to clean up
            delay: Delay in seconds before cleanup
        """

        def cleanup_files() -> None:
            time.sleep(delay)
            for file_path in files:
                with contextlib.suppress(OSError):
                    pathlib.Path(file_path).unlink()

        threading.Thread(target=cleanup_files, daemon=True).start()
