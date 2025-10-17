# 📓 hexDAG Notebooks (Future Enhancement)

## Status: Infrastructure Ready

The notebook infrastructure is fully configured and ready for use:

✅ **Validation script** - `scripts/check_notebooks.py`
✅ **Pre-commit hooks** - Auto-formatting, validation, output stripping
✅ **CI/CD integration** - Azure pipeline stage
✅ **Dependencies** - `uv sync --all-extras` includes notebook support

## Why No Notebooks Yet?

We prioritized:
1. **Working code first** - See `examples/` directory for 40+ working examples
2. **Integration tests** - See `tests/integration/` for comprehensive test scenarios
3. **YAML-first docs** - See `docs/` for YAML pipeline guides

Notebooks require significant effort to maintain API alignment and we want to ensure they're production-quality and fully working before including them.

## Future Notebook Ideas

When ready, consider creating:

**Getting Started:**
- `01_hello_hexdag.ipynb` - Minimal working example (based on `examples/01_basic_dag.py`)
- `02_yaml_pipeline.ipynb` - Simple YAML workflow (based on `examples/13_yaml_pipelines.py`)

**Real-World Use Cases:**
- Customer support automation
- Document intelligence
- Research assistant
- Data pipeline orchestration

**Production Patterns:**
- Multi-agent collaboration
- Environment management
- Monitoring and observability
- CI/CD integration

## For Now

**Learn hexDAG through:**
- 📝 **Examples** - `examples/` directory (40+ working scripts)
- 🧪 **Tests** - `tests/integration/` (production patterns)
- 📚 **Docs** - `docs/` (comprehensive guides)
- 💻 **CLI** - `hexdag --help` (command-line tools)

## Contributing Notebooks

Want to create notebooks? Follow these guidelines:

1. **Start with working examples** - Base on `examples/` directory
2. **Use correct API** - Check `hexdag/core/` for current signatures
3. **Test execution** - `uv run python scripts/check_notebooks.py`
4. **YAML-first** - Emphasize declarative approach
5. **Keep it simple** - Working code > comprehensive coverage

See [CLAUDE.md](../CLAUDE.md) for YAML-first philosophy and development guidelines.

---

**Infrastructure ready. Notebooks coming soon. Learn through examples and docs for now!**
