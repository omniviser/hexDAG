# 📚 hexDAG Documentation

**Complete guide to building enterprise AI agent workflows with hexDAG**

Welcome to the hexDAG documentation! This guide will help you navigate the documentation ecosystem and find what you need quickly.

## 🎯 Start Here

### New to hexDAG?
1. **[📓 Interactive Notebooks](../notebooks/)** - Best way to learn! Three hands-on tutorials (1 hour total)
   - [01. Introduction](../notebooks/01_introduction.ipynb) - Your first pipeline (15 min)
   - [02. YAML Pipelines](../notebooks/02_yaml_pipelines.ipynb) - Declarative workflows (25 min)
   - [03. Practical Workflow](../notebooks/03_practical_workflow.ipynb) - Real-world patterns (30 min)

2. **[Getting Started Guide]()** - Quick setup and first steps
   - [Installation](installation.md) - Set up hexDAG
   - [Quick Start](quickstart.md) - Build your first workflow
   - [Core Concepts](concepts.md) - Understand the architecture

### Building Production Workflows?
- **[🔧 Implementation Guide](IMPLEMENTATION_GUIDE.md)** - Comprehensive production guide (36KB)
- **[⌨️ CLI Reference](CLI_REFERENCE.md)** - Complete CLI documentation (22KB)
- **[📊 Examples Directory](../examples/)** - 40+ working code examples

## 📖 Documentation Structure

### Core Concepts

#### **[🤔 Philosophy & Design](PHILOSOPHY.md)**
Understanding hexDAG's design principles and positioning in the AI framework landscape.
- The Six Pillars (Async-First, Event-Driven, etc.)
- Comparison with LangChain, LangGraph, CrewAI, AutoGen
- Why hexDAG exists and what problems it solves

#### **[🏗️ System Architecture](ARCHITECTURE.md)** ⭐ NEW
Complete system architecture with diagrams showing how hexDAG works end-to-end.
- High-level architecture diagrams
- Component interactions and data flow
- Execution pipeline visualization
- Lifecycle management
- Extension points and customization

#### **[🏗️ Core Concepts](concepts.md)**
Essential hexDAG concepts and architecture patterns.
- DirectedGraph and NodeSpec
- Orchestrator and execution model
- Event system and observability
- Validation and type safety

### User Guides

#### **[📝 Node Types](node-types.md)**
Comprehensive guide to all available node types:
- FunctionNode - Execute Python functions
- LLMNode - Language model interactions
- AgentNode - ReAct pattern agents with tools
- LoopNode - Iterative processing
- ConditionalNode - Branching logic

#### **📊 YAML Pipelines** *(See notebooks and examples)*
Declarative workflow configuration guide:
- **[02. YAML Pipelines Notebook](../notebooks/02_yaml_pipelines.ipynb)** - Interactive tutorial
- **[YAML Examples](../examples/13_yaml_pipelines.py)** - Working code examples
- **[Implementation Guide](IMPLEMENTATION_GUIDE.md)** - YAML Pipelines section for YAML details

### Implementation

#### **[🔧 Implementation Guide](IMPLEMENTATION_GUIDE.md)** (36KB)
Complete guide for production deployments:
- Project structure and organization
- Adapter configuration (LLM, Memory, Database)
- Agent factory and YAML pipelines
- Error handling and retry strategies
- Performance optimization
- Production deployment patterns

#### **[⌨️ CLI Reference](CLI_REFERENCE.md)** (22KB)
Command-line interface documentation:
- `hexdag run` - Execute pipelines
- `hexdag validate` - Validate configurations
- `hexdag build` - Docker containerization (dev only)
- Configuration management
- Environment variables
- Security best practices

### Advanced Topics

#### **[🔌 Plugin System](PLUGIN_SYSTEM.md)** (18KB)
Extending hexDAG with custom components:
- Plugin architecture and discovery
- Creating custom adapters
- Custom node types
- Tool development
- Policy implementation
- Registry system
- **Quick Start section** for rapid plugin development

### Reference Documentation

#### **[Component Reference]()**
Auto-generated documentation for all registered components:
- **[Nodes](nodes.md)** - All available node types
- **[Adapters](adapters.md)** - LLM, memory, database adapters
- **[Tools](tools.md)** - Built-in and custom tools
- **[Ports](ports.md)** - Interface definitions

#### **[Namespaces]()**
Components organized by namespace:
- **[Core Namespace](core.md)** - Built-in components
- **[Plugin Namespace](plugin.md)** - Plugin-provided components

### Planning & Development

#### **[🗺️ Roadmap](ROADMAP.md)** (13KB)
Our vision for the future of hexDAG:
- Upcoming features
- Long-term goals
- Community contributions
- Version milestones

#### **[YAML Pipelines Roadmap](YAML_PIPELINES_ROADMAP.md)**
Planned enhancements to YAML pipeline system:
- Advanced node types
- Improved validation
- Enhanced templates
- Performance optimizations

#### **[YAML Pipelines Guide](YAML_PIPELINES_ARCHITECTURE.md)**
Technical documentation for YAML pipelines internals:
- YamlPipelineBuilder architecture
- Node factory system
- Graph compilation
- Validation framework

### Development

#### **[Contributing Guide](../CONTRIBUTING.md)**
How to contribute to hexDAG:
- Development setup
- Code quality standards
- Testing requirements
- Pull request process

#### **[Doctest Guidelines](doctest_guidelines.md)**
Writing effective doctests:
- Doctest conventions
- Testing patterns
- Common pitfalls
- Best practices

## 🔗 Quick Links

### External Resources
- **[GitHub Repository](https://dev.azure.com/omniviser/hexDAG/_git/hexDAG)** - Source code and issues
- **[Changelog](../CHANGELOG.md)** - Version history and release notes
- **[Architecture Roadmap](../ARCHITECTURE_ROADMAP.md)** - Technical evolution plans

### Internal Navigation
- **[Main README](../README.md)** - Project overview and quick start
- **[Examples](../examples/)** - Code examples and patterns
- **[Tests](../tests/)** - Integration tests and test patterns
- **[Notebooks](../notebooks/)** - Interactive learning materials

## 🎓 Learning Paths

### Path 1: Low-Code YAML Development
**Goal**: Build AI workflows using YAML without deep Python knowledge

1. Start with [02. YAML Pipelines Notebook](../notebooks/02_yaml_pipelines.ipynb)
2. Read [Implementation Guide - YAML Pipelines](IMPLEMENTATION_GUIDE.md#agent-factory)
3. Explore [YAML Examples](../examples/13_yaml_pipelines.py)
4. Reference [Node Types Guide](node-types.md)
5. Use [CLI Reference](CLI_REFERENCE.md) for execution

### Path 2: Python Developer
**Goal**: Build complex workflows programmatically

1. Start with [01. Introduction Notebook](../notebooks/01_introduction.ipynb)
2. Read [Core Concepts](concepts.md)
3. Explore [Basic Examples](../examples/01_basic_dag.py)
4. Study [Node Types](node-types.md)
5. Build with [Implementation Guide](IMPLEMENTATION_GUIDE.md)

### Path 3: Enterprise Deployment
**Goal**: Deploy hexDAG in production environments

1. Review [Philosophy](PHILOSOPHY.md) for architecture understanding
2. Read [Implementation Guide](IMPLEMENTATION_GUIDE.md) fully
3. Study [CLI Reference](CLI_REFERENCE.md) for deployment options
4. Set up monitoring with [Event System](concepts.md#event-system)
5. Implement [Plugin System](PLUGIN_SYSTEM.md) for custom components

### Path 4: Plugin Developer
**Goal**: Extend hexDAG with custom components

1. Understand [Plugin System](PLUGIN_SYSTEM.md)
2. Follow Quick Start in [Plugin System](PLUGIN_SYSTEM.md)
3. Reference [Component Reference]()
4. Study [Built-in Adapters](../hexdag/builtin/adapters/)
5. Contribute via [Contributing Guide](../CONTRIBUTING.md)

## 📊 Documentation Statistics

- **Total Documentation**: ~150KB of comprehensive guides
- **Code Examples**: 40+ working Python scripts
- **Interactive Notebooks**: 3 validated Jupyter notebooks
- **Integration Tests**: 100+ test scenarios
- **Reference Docs**: Auto-generated from component registry

## 🆘 Getting Help

### Documentation Issues
- **Broken Link**: Report in [GitHub Issues](https://dev.azure.com/omniviser/hexDAG/_git/hexDAG/pullrequests)
- **Unclear Guide**: Request clarification via pull request
- **Missing Topic**: Suggest new documentation

### Common Questions
1. **"How do I...?"** → Check [Implementation Guide](IMPLEMENTATION_GUIDE.md)
2. **"What's the command for...?"** → See [CLI Reference](CLI_REFERENCE.md)
3. **"How does X work?"** → Read [Core Concepts](concepts.md)
4. **"Can I extend...?"** → Read [Plugin System](PLUGIN_SYSTEM.md)

## 📝 Documentation Conventions

### Symbols Used
- 🎯 **Important concept** - Core framework ideas
- ⚠️ **Warning** - Security or breaking changes
- 💡 **Tip** - Best practices and optimization
- 🔍 **Deep Dive** - Advanced technical details
- 📊 **Example** - Code samples and patterns

### File Naming
- `UPPERCASE.md` - Major documentation files (e.g., `PHILOSOPHY.md`)
- `lowercase-hyphenated.md` - Section files (e.g., `node-types.md`)
- `TitleCase.md` - Reference docs (e.g., `YAML_PIPELINES_ARCHITECTURE.md`)

---

**Last Updated**: October 2024
**hexDAG Version**: 0.3.0-a3
**Documentation Status**: ✅ Complete and Validated
