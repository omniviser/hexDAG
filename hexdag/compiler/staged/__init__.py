"""Staged compiler package — the single front door is :func:`compile`."""

from hexdag.compiler.staged.compile import CompileResult, compile

__all__ = ["CompileResult", "compile"]
