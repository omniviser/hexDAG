"""Tests for the node_timer utility."""

import time

from hexdag.core.utils.node_timer import Timer, node_timer


class TestTimer:
    def test_duration_ms_increases(self) -> None:
        t = Timer()
        time.sleep(0.01)
        assert t.duration_ms >= 5  # at least 5ms

    def test_duration_str_format(self) -> None:
        t = Timer()
        result = t.duration_str
        assert "." in result  # has decimal
        float(result)  # parseable as float


class TestNodeTimer:
    def test_yields_timer(self) -> None:
        with node_timer() as t:
            assert isinstance(t, Timer)

    def test_duration_available_inside_block(self) -> None:
        with node_timer() as t:
            time.sleep(0.01)
            assert t.duration_ms >= 5

    def test_duration_available_after_block(self) -> None:
        with node_timer() as t:
            time.sleep(0.01)
        assert t.duration_ms >= 5
