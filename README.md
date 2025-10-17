# 🤖 hexDAG - AI Agent Orchestration Framework

[![Python 3.12](https://img.shields.io/badge/python-3.12.*-blue.svg)](https://www.python.org/downloads/)
[![uv: Python package manager](https://img.shields.io/badge/uv-fastest--python--installer-blueviolet?logo=python&logoColor=white)](https://github.com/astral-sh/uv)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Enterprise-ready AI agent orchestration with low-code declarative workflows**

hexDAG revolutionizes AI development by making agent orchestration and data science workflows accessible through declarative YAML configurations, while maintaining the power and flexibility needed for enterprise deployments.

## ✨ Why hexDAG?

Traditional AI frameworks force you to choose between simplicity and power. hexDAG delivers both through:

- **🤖 Agent-First Design**: Build complex multi-agent systems with simple YAML
- **📊 Data Science Ready**: Mix AI agents with traditional data processing seamlessly
- **🌊 Real-Time Streaming**: See agent thoughts and memory operations as they happen
- **🔧 Low-Code Development**: Non-technical users can create sophisticated workflows
- **🏢 Enterprise Grade**: Production-ready with comprehensive monitoring and control

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
# Clone and install


git clone https://omniviser@dev.azure.com/omniviser/hexDAG/_git/hexDAG
cd hexDAG
uv sync # optional, uv checks packages in each run
````

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
from hexai import Orchestrator
from hexai.agent_factory import YamlPipelineBuilder

# Load and execute the workflow
builder = YamlPipelineBuilder()
graph, metadata = builder.build_from_yaml_file("research_agent.yaml")

orchestrator = Orchestrator()
result = await orchestrator.run(graph, {"topic": "AI trends 2024"})

## 📚 Documentation & Learning

### 📓 Interactive Notebooks (Recommended Start)
Learn hexDAG through 3 comprehensive, working Jupyter notebooks:

- **[01. Introduction](notebooks/01_introduction.ipynb)** - Your first pipeline (15 min)
- **[02. YAML Pipelines](notebooks/02_yaml_pipelines.ipynb)** - Declarative workflows (25 min)
- **[03. Practical Workflow](notebooks/03_practical_workflow.ipynb)** - Real-world patterns (30 min)

**All notebooks execute successfully:** `✅ All 3 notebook(s) validated successfully!`

### 📝 Examples & More
- **[Examples Directory](examples/)** - 40+ working Python scripts
- **[Integration Tests](tests/integration/)** - Production test scenarios

### Core Concepts
- **[🤔 Philosophy & Design](docs/PHILOSOPHY.md)** - The six pillars and design principles
- **[🏗️ Framework Architecture](docs/HEXAI_FRAMEWORK.md)** - Technical architecture deep dive

### Implementation
- **[🔧 Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)** - Build production-ready AI workflows
- **[🎯 Agent Patterns](docs/PIPELINES_GUIDE.md)** - Multi-agent coordination patterns

### CLI & Tools
- **[⌨️ CLI Reference](docs/CLI_REFERENCE.md)** - Complete command-line interface documentation (includes Docker build guide)

### Planning
- **[🗺️ Roadmap](docs/ROADMAP.md)** - Our vision for the future of AI orchestration

### Navigation
- **[📚 Documentation Guide](docs/DOCUMENTATION_GUIDE.md)** - Navigate the documentation ecosystem

## 🎪 Examples

Explore 20+ examples covering everything from basic to advanced patterns:

```bash
# Run all examples
uv run examples/run_all.py

# Try specific examples
uv run examples/01_basic_dag.py           # DAG fundamentals
uv run examples/10_agent_nodes.py         # AI agents with tools
uv run examples/13_yaml_pipelines.py      # Declarative workflows
uv run examples/19_complex_workflow.py     # Enterprise patterns
```

## 🛠️ Development

```bash
# Setup development environment
uv run pre-commit install

# Run tests
uv run pytest

# Code quality checks
uv run pre-commit run --all-files
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

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with ❤️ for the AI community by the hexDAG team**
