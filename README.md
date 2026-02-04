# 🤖 hexDAG - AI Agent Orchestration Framework

[![PyPI version](https://img.shields.io/pypi/v/hexdag.svg)](https://pypi.org/project/hexdag/)
[![Python 3.12](https://img.shields.io/badge/python-3.12.*-blue.svg)](https://www.python.org/downloads/)
[![uv: Python package manager](https://img.shields.io/badge/uv-fastest--python--installer-blueviolet?logo=python&logoColor=white)](https://github.com/astral-sh/uv)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

> **Enterprise-ready AI agent orchestration with low-code declarative workflows and powerful macro system**

hexDAG revolutionizes AI development by making agent orchestration and data science workflows accessible through declarative YAML configurations, reusable macro templates, and advanced conversation patterns, while maintaining the power and flexibility needed for enterprise deployments.

## ✨ Why hexDAG?

Traditional AI frameworks force you to choose between simplicity and power. hexDAG delivers both through:

- **🤖 Agent-First Design**: Build complex multi-agent systems with simple YAML
- **📊 Data Science Ready**: Mix AI agents with traditional data processing seamlessly
- **🌊 Real-Time Streaming**: See agent thoughts and memory operations as they happen
- **🔧 Low-Code Development**: Non-technical users can create sophisticated workflows
- **🏢 Enterprise Grade**: Production-ready with comprehensive monitoring and control
- **🎭 Macro System**: Reusable pipeline templates that expand into full workflows
- **💬 Conversation Patterns**: Built-in support for multi-turn conversations with memory

## 🎯 The Six Pillars

1. **Async-First Architecture** - Non-blocking execution for maximum performance
2. **Event-Driven Observability** - Real-time monitoring of agent actions
3. **Pydantic Validation Everywhere** - Type safety and runtime validation
4. **Hexagonal Architecture** - Clean separation of business logic and infrastructure
5. **Composable Declarative Files** - Build complex workflows from simple components
6. **DAG-Based Orchestration** - Intelligent dependency management and parallelization

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install hexdag

# Or with uv (recommended)
uv pip install hexdag

# With optional dependencies
pip install hexdag[openai]      # OpenAI LLM support
pip install hexdag[anthropic]   # Anthropic Claude support
pip install hexdag[all]         # All optional dependencies
```

#### Development Installation

```bash
# Clone and install for development
git clone https://github.com/omniviser/hexdag.git
cd hexdag
uv sync
```

### MCP Server for LLM Editors

hexDAG includes a built-in MCP (Model Context Protocol) server that exposes pipeline building capabilities to Claude Code, Cursor, and other LLM-powered editors:

```bash
# Development: Install MCP dependencies
uv sync --extra mcp

# Production: Install from PyPI with MCP support
uv pip install "hexdag[mcp]"

# Configure in Claude Desktop (~/Library/Application Support/Claude/claude_desktop_config.json)
{
  "mcpServers": {
    "hexdag": {
      "command": "uv",
      "args": ["run", "python", "-m", "hexdag", "--mcp"]
    }
  }
}
```

The MCP server provides LLMs with tools to:
- List available nodes, adapters, tools, and macros from your registry
- Build and validate YAML pipelines interactively
- Get component schemas and documentation
- Auto-discover custom plugins from your `pyproject.toml`

See [examples/mcp/](examples/mcp/) for detailed configuration guides.

### Your First Agent Workflow

Create a simple AI agent workflow with YAML:

```yaml
# research_agent.yaml
name: research_workflow
description: AI-powered research assistant

nodes:
  - type: agent
    id: researcher
    params:
      initial_prompt_template: "Research the topic: {{topic}}"
      max_steps: 5
      available_tools: ["web_search", "summarize"]
    depends_on: []

  - type: agent
    id: analyst
    params:
      initial_prompt_template: |
        Analyze the research findings: {{researcher.results}}
        Provide actionable insights.
      max_steps: 3
    depends_on: [researcher]

  - type: function
    id: formatter
    params:
      fn: format_report
      input_mapping:
        title: "researcher.topic"
        findings: "researcher.results"
        insights: "analyst.insights"
    depends_on: [researcher, analyst]
```

Run it with Python:

```python
from hexdag import Orchestrator, YamlPipelineBuilder

# Load and execute the workflow
builder = YamlPipelineBuilder()
graph, metadata = builder.build_from_yaml_file("research_agent.yaml")

orchestrator = Orchestrator()
result = await orchestrator.run(graph, {"topic": "AI trends 2024"})

## 📚 Documentation & Learning

### 📓 Interactive Notebooks (Recommended Start)
Learn hexDAG through comprehensive, working Jupyter notebooks:

**Core Concepts:**
- **[01. Introduction](notebooks/01_introduction.ipynb)** - Your first pipeline (15 min)
- **[02. YAML Pipelines](notebooks/02_yaml_pipelines.ipynb)** - Declarative workflows (25 min)
- **[03. Practical Workflow](notebooks/03_practical_workflow.ipynb)** - Real-world patterns (30 min)

**Advanced Features:**
- **[06. Dynamic Reasoning Agent](notebooks/06_dynamic_reasoning_agent.ipynb)** - Advanced agent patterns
- **[Advanced Few-shot & Retry](notebooks/advanced_fewshot_and_retry.ipynb)** - Error handling and examples
- **[Composable LLM Architecture](notebooks/composable_llm_architecture.ipynb)** - Modular AI systems

**All notebooks execute successfully:** `✅ All notebook(s) validated successfully!`

### 📚 Complete Documentation
- **[📖 Documentation Hub](docs/README.md)** - Complete navigation with learning paths
- **[🤔 Philosophy & Design](docs/PHILOSOPHY.md)** - Six pillars and design principles
- **[🔧 Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)** - Production-ready workflows
- **[⌨️ CLI Reference](docs/CLI_REFERENCE.md)** - Complete CLI documentation
- **[🔌 Plugin System](docs/PLUGIN_SYSTEM.md)** - Custom component development
- **[🗺️ Roadmap](docs/ROADMAP.md)** - Future vision and features

### 📝 Additional Resources
- **[Demo Directory](examples/demo/)** - Live demonstration scripts
- **[Integration Tests](tests/integration/)** - Production test scenarios

## 🎪 Interactive Notebooks

Explore comprehensive Jupyter notebooks for hands-on learning:

```bash
# Start Jupyter to explore notebooks
jupyter notebook notebooks/

# Or run specific notebooks
jupyter notebook notebooks/01_introduction.ipynb           # Getting started
jupyter notebook notebooks/02_yaml_pipelines.ipynb         # YAML workflows
jupyter notebook notebooks/03_practical_workflow.ipynb     # Real-world patterns
jupyter notebook notebooks/06_dynamic_reasoning_agent.ipynb # Advanced agents
```

### Running the Demo

```bash
# Run the startup pitch demo
uv run python examples/demo/run_demo_pitch.py

# Or explore the YAML configuration
cat examples/demo/demo_startup_pitch.yaml
```

## 🛠️ Development

```bash
# Setup development environment
uv run pre-commit install

# Run tests
uv run pytest

# Code quality checks
uv run pre-commit run --all-files

# Build documentation
uv run docs-build        # Build HTML documentation
uv run docs-clean        # Clean build directory
uv run docs-rebuild      # Clean and rebuild
uv run docs-check        # Build with warnings as errors
uv run docs-autobuild    # Auto-rebuild on file changes

# Documentation will be in docs/build/html/
```

## 🌟 Key Features

### 🤖 Multi-Agent Orchestration
- Sequential agent chains for complex reasoning
- Parallel specialist agents for diverse perspectives
- Hierarchical agent networks with supervisor patterns

### 📊 Data Science Integration
- Mix AI agents with traditional data processing
- Real-time streaming for Jupyter notebooks
- Built-in support for popular ML frameworks

### 🌊 Real-Time Streaming
- WebSocket-based agent action streaming
- Memory operation visualization
- Interactive debugging and control

### 🔧 Low-Code Development
- YAML-based workflow definitions
- Template system for reusable patterns
- Automatic field mapping between nodes
- Visual workflow editor (coming soon)

### 🔄 Smart Data Mapping
- **Automatic Input Mapping**: Define how data flows between nodes with simple mappings
- **Nested Field Extraction**: Access deeply nested data with dot notation
- **Type Inference**: Automatic type detection from Pydantic models
- **Flexible Patterns**: Support for passthrough, rename, and prefixed mappings

### 🎭 Powerful Macro System
- **Reusable Templates**: Define pipeline patterns once, use everywhere
- **Built-in Macros**: ConversationMacro, LLMMacro, ToolMacro, ReasoningAgentMacro
- **YAML Integration**: Seamlessly use macros in declarative pipelines
- **Dynamic Expansion**: Macros expand at runtime into full DAG subgraphs
- **Configuration Inheritance**: Override macro defaults per invocation

## 🔒 Production Security

### Docker Build Command

The `hexdag build` command generates containerized deployments from YAML pipelines.

⚠️ **IMPORTANT**: This command is designed for **development and trusted pipelines only**.

**Production Safety:**
```bash
# Disable build command in production environments
export HEXDAG_DISABLE_BUILD=1
```

**For detailed documentation**, including security threat model, hardening checklist, and Docker Compose patterns, see the [CLI Reference](docs/CLI_REFERENCE.md#build---build-docker-containers).

## 🤝 Community

- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)


## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

**Built with ❤️ for the AI community by the hexDAG team**
