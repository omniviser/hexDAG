# 🤖 hexDAG - AI Agent Orchestration Framework

[![Python 3.12.8](https://img.shields.io/badge/python-3.12.8-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20manager-poetry-blue.svg)](https://python-poetry.org/)
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
poetry install
```

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
```

## 📚 Documentation

### Core Concepts
- **[🤔 Philosophy & Design](docs/PHILOSOPHY.md)** - The six pillars and design principles
- **[🏗️ Framework Architecture](docs/HEXAI_FRAMEWORK.md)** - Technical architecture deep dive

### Implementation
- **[🔧 Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)** - Build production-ready AI workflows
- **[🎯 Agent Patterns](docs/PIPELINES_GUIDE.md)** - Multi-agent coordination patterns

### Planning
- **[🗺️ Roadmap](docs/ROADMAP.md)** - Our vision for the future of AI orchestration

### Navigation
- **[📚 Documentation Guide](docs/DOCUMENTATION_GUIDE.md)** - Navigate the documentation ecosystem

## 🎪 Examples

Explore 20+ examples covering everything from basic to advanced patterns:

```bash
# Run all examples
poetry run python examples/run_all.py

# Try specific examples
poetry run python examples/01_basic_dag.py           # DAG fundamentals
poetry run python examples/10_agent_nodes.py         # AI agents with tools
poetry run python examples/13_yaml_pipelines.py      # Declarative workflows
poetry run python examples/19_complex_workflow.py     # Enterprise patterns
```

## 🛠️ Development

```bash
# Setup development environment
poetry install
poetry run pre-commit install

# Run tests
poetry run pytest

# Code quality checks
poetry run pre-commit run --all-files
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
- Visual workflow editor (coming soon)

## 🤝 Community

- **Discord**: [Join our community](https://discord.gg/hexdag)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Issues**: [GitHub Issues](https://github.com/yourusername/hexdag/issues)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with ❤️ for the AI community by the hexDAG team**
