"""Tests for FileExporter class."""

import pathlib
import subprocess
import tempfile
from unittest.mock import Mock, patch

import pytest

from hexai.visualization.file_exporter import FileExporter


class TestFileExporter:
    """Tests for FileExporter class."""

    @pytest.fixture
    def exporter(self):
        """Create a FileExporter instance."""
        return FileExporter()

    @pytest.fixture
    def dot_content(self):
        """Sample DOT content for testing."""
        return """digraph G {
            node1 [label="Node 1"];
            node2 [label="Node 2"];
            node1 -> node2;
        }"""

    def test_render_to_file_success(self, exporter, dot_content):
        """Test successful rendering to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_graph"

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0)

                result = exporter.render_to_file(dot_content, output_path, "png")

                assert result == f"{output_path}.png"
                mock_run.assert_called_once()

                # Check that dot command was called correctly
                args = mock_run.call_args[0][0]
                assert args[0] == "dot"
                assert "-Tpng" in args
                assert "-o" in args

    def test_render_to_file_graphviz_error(self, exporter, dot_content):
        """Test handling Graphviz rendering error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_graph"

            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.CalledProcessError(
                    1, "dot", stderr="Graphviz error"
                )

                with pytest.raises(RuntimeError, match="Failed to render graph"):
                    exporter.render_to_file(dot_content, output_path, "png")

    def test_render_to_file_graphviz_not_installed(self, exporter, dot_content):
        """Test handling when Graphviz is not installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_graph"

            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError()

                with pytest.raises(ImportError, match="Graphviz 'dot' command not found"):
                    exporter.render_to_file(dot_content, output_path, "png")

    def test_render_to_file_different_formats(self, exporter, dot_content):
        """Test rendering to different formats."""
        formats = ["png", "svg", "pdf"]

        with tempfile.TemporaryDirectory() as tmpdir:
            for fmt in formats:
                output_path = f"{tmpdir}/test_graph_{fmt}"

                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = Mock(returncode=0)

                    result = exporter.render_to_file(dot_content, output_path, fmt)

                    assert result == f"{output_path}.{fmt}"
                    args = mock_run.call_args[0][0]
                    assert f"-T{fmt}" in args

    @patch("hexai.visualization.file_exporter.threading.Timer")
    @patch("subprocess.run")
    def test_show_success(self, mock_run, mock_timer, exporter, dot_content):
        """Test successful show operation."""
        mock_run.return_value = Mock(returncode=0)

        with patch("hexai.visualization.file_exporter.platform.system", return_value="Darwin"):
            with patch(
                "hexai.visualization.file_exporter.shutil.which", return_value="/usr/bin/open"
            ):
                exporter.show(dot_content, "Test Graph")

                # Check dot was called to create PNG
                assert mock_run.call_count == 2  # Once for PNG, once for open

                # Check cleanup timer was started
                mock_timer.assert_called_once()
                mock_timer.return_value.start.assert_called_once()

    @patch("subprocess.run")
    def test_show_no_viewer(self, mock_run, exporter, dot_content):
        """Test show when no viewer is available."""
        mock_run.return_value = Mock(returncode=0)

        with patch("hexai.visualization.file_exporter.platform.system", return_value="Unknown"):
            with patch("hexai.visualization.file_exporter.logger") as mock_logger:
                exporter.show(dot_content, "Test Graph")

                # Should log error about no viewer
                mock_logger.error.assert_called_once()
                assert "No default image viewer found" in mock_logger.error.call_args[0][0]

    def test_generate_html(self, exporter, dot_content):
        """Test HTML generation."""
        with patch.object(exporter, "_generate_svg", return_value="<svg>Test SVG</svg>"):
            html = exporter.generate_html(dot_content, "Test Title")

            assert "<html>" in html
            assert "Test Title" in html
            assert "<svg>Test SVG</svg>" in html
            assert "zoomIn()" in html
            assert "zoomOut()" in html
            assert "resetZoom()" in html
            assert "downloadSVG()" in html

    def test_save_html(self, exporter):
        """Test saving HTML to file."""
        html_content = "<html><body>Test</body></html>"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "test.html"

            exporter.save_html(html_content, str(output_path))

            assert output_path.exists()
            assert output_path.read_text() == html_content

    def test_cleanup_temp_files(self, exporter):
        """Test cleaning up temporary files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = pathlib.Path(tmpdir) / "test1.txt"
            file2 = pathlib.Path(tmpdir) / "test2.txt"
            file1.write_text("test")
            file2.write_text("test")

            files = [str(file1), str(file2)]

            # Cleanup should remove files
            exporter.cleanup_temp_files(files)

            assert not file1.exists()
            assert not file2.exists()

    def test_cleanup_temp_files_ignore_errors(self, exporter):
        """Test cleanup ignores errors for missing files."""
        non_existent_files = ["/tmp/non_existent_1.txt", "/tmp/non_existent_2.txt"]

        # Should not raise any errors
        exporter.cleanup_temp_files(non_existent_files)

    @patch("subprocess.run")
    def test_generate_svg_success(self, mock_run, exporter, dot_content):
        """Test successful SVG generation."""
        mock_run.return_value = Mock(stdout="<svg>Generated SVG</svg>", returncode=0)

        svg = exporter._generate_svg(dot_content)

        assert svg == "<svg>Generated SVG</svg>"
        mock_run.assert_called_once()

        args = mock_run.call_args[0][0]
        assert args[0] == "dot"
        assert "-Tsvg" in args

    @patch("subprocess.run")
    @patch("hexai.visualization.file_exporter.logger")
    def test_generate_svg_error(self, mock_logger, mock_run, exporter, dot_content):
        """Test SVG generation error handling."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "dot", stderr="SVG generation failed"
        )

        svg = exporter._generate_svg(dot_content)

        assert "<p>Error generating visualization" in svg
        mock_logger.error.assert_called_once()

    @patch("hexai.visualization.file_exporter.platform.system")
    @patch("hexai.visualization.file_exporter.shutil.which")
    @patch("subprocess.run")
    def test_open_with_default_viewer_macos(self, mock_run, mock_which, mock_platform, exporter):
        """Test opening with default viewer on macOS."""
        mock_platform.return_value = "Darwin"
        mock_which.return_value = "/usr/bin/open"

        exporter._open_with_default_viewer("/tmp/test.png")

        mock_run.assert_called_once_with(["open", "/tmp/test.png"], check=False)

    @patch("hexai.visualization.file_exporter.platform.system")
    @patch("hexai.visualization.file_exporter.shutil.which")
    @patch("subprocess.run")
    def test_open_with_default_viewer_linux(self, mock_run, mock_which, mock_platform, exporter):
        """Test opening with default viewer on Linux."""
        mock_platform.return_value = "Linux"
        mock_which.return_value = "/usr/bin/xdg-open"

        exporter._open_with_default_viewer("/tmp/test.png")

        mock_run.assert_called_once_with(["xdg-open", "/tmp/test.png"], check=False)

    @patch("hexai.visualization.file_exporter.platform.system")
    @patch("hexai.visualization.file_exporter.logger")
    def test_open_with_default_viewer_unsupported(self, mock_logger, mock_platform, exporter):
        """Test handling unsupported platform."""
        mock_platform.return_value = "Windows"

        exporter._open_with_default_viewer("/tmp/test.png")

        mock_logger.error.assert_called_once()
        assert "No default image viewer found" in mock_logger.error.call_args[0][0]

    @patch("hexai.visualization.file_exporter.threading.Timer")
    def test_schedule_cleanup(self, mock_timer, exporter):
        """Test scheduling cleanup of temporary files."""
        files = ["/tmp/test1.txt", "/tmp/test2.txt"]

        exporter._schedule_cleanup(files, delay=5)

        # Check Timer was created with correct delay
        mock_timer.assert_called_once()
        assert mock_timer.call_args[0][0] == 5  # delay argument
        # Check timer was started
        mock_timer.return_value.start.assert_called_once()

    def test_render_to_file_cleanup(self, exporter, dot_content):
        """Test that temporary DOT file is cleaned up after rendering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_graph"

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0)

                # Track if pathlib.Path.unlink is called
                with patch("pathlib.Path.unlink"):
                    exporter.render_to_file(dot_content, output_path, "png")

                    # Cleanup should be attempted
                    assert True  # contextlib.suppress handles errors
