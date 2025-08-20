# HexDAG Installation & Features

HexDAG is a lightweight DAG orchestration framework with enterprise pipeline capabilities.
It provides optional modules for CLI usage, DAG visualization, and LLM adapters.

---

## 🔧 Base Installation

Install the core HexDAG framework without optional features:

```bash
pip install hexdag
```
## 🎛 Optional Features

You can install additional features depending on your use case.

### CLI

Command line utilities for running and managing pipelines:

```bash
pip install hexdag[cli]
```

Requires:

 - click

 - pyyaml

### Visualization

Graph visualization of pipelines using Graphviz:

```bash
pip install hexdag[viz]
```

Requires:

 - Python package: graphviz

 - System binary: dot from Graphviz

### On Ubuntu/WSL install Graphviz system package with:

```bash
sudo apt-get update && sudo apt-get install graphviz
```

### Check installation:
```bash
dot -V
```
## LLM Adapters

Adapters for large language models:

### OpenAI:

```bash
pip install hexdag[adapters-openai]
```

### Anthropic:

```bash
pip install hexdag[adapters-anthropic]
```

### All Features

Install everything at once:

```bash
pip install hexdag[all]
```
## 📊 Compatibility Matrix
| Feature            | Requires        | Compatible With | Notes                    |
| ------------------ |-----------------| --------------- | ------------------------ |
| cli                | click, pyyaml, rich | all             | Command line utilities   |
| viz                | graphviz, dot   | cli, adapters   | DAG visualization        |
| adapters-openai    | openai          | viz, cli        | LLM functionality        |
| adapters-anthropic | anthropic       | viz, cli        | Alternative LLM provider |
