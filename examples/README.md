# 🎓 hexAI Learning Examples

> **Progressive examples that teach you hexAI from basics to advanced usage**

## 📚 Learning Path

This directory contains examples that progressively teach you hexAI concepts. Follow them in order for the best learning experience.

### 🏁 Getting Started (Bare DAG API)
1. **[01_basic_dag.py](01_basic_dag.py)** - Your first DAG with simple functions
2. **[02_dependencies.py](02_dependencies.py)** - Understanding node dependencies and execution waves
3. **[03_validation_basics.py](03_validation_basics.py)** - Input/output validation and type conversion

### 🔧 Core Features
4. **[04_validation_strategies.py](04_validation_strategies.py)** - Strict, coerce, and passthrough validation
5. **[05_event_system.py](05_event_system.py)** - Event observers and monitoring
6. **[06_ports_and_adapters.py](06_ports_and_adapters.py)** - Using ports for external services
7. **[07_error_handling.py](07_error_handling.py)** - Graceful error handling and recovery

### 🎭 Node Types
8. **[08_function_nodes.py](08_function_nodes.py)** - Working with function nodes and factories
9. **[09_llm_nodes.py](09_llm_nodes.py)** - LLM interactions and structured outputs
10. **[10_agent_nodes.py](10_agent_nodes.py)** - Multi-step reasoning with tools

### 📊 Visualization & Debugging
11. **[11_dag_visualization.py](11_dag_visualization.py)** - DAG visualization and debugging
12. **[12_streaming_execution.py](12_streaming_execution.py)** - Real-time pipeline monitoring

### 🏢 Enterprise Features (Pipelines API)
13. **[13_yaml_pipelines.py](13_yaml_pipelines.py)** - YAML-based pipeline definitions
14. **[14_pipeline_compilation.py](14_pipeline_compilation.py)** - Pipeline compilation for performance
15. **[15_pipeline_catalog.py](15_pipeline_catalog.py)** - Managing pipeline versions

### ⚠️ Common Issues & Solutions
16. **[16_validation_errors.py](16_validation_errors.py)** - Common validation problems and fixes
17. **[17_circular_dependencies.py](17_circular_dependencies.py)** - Detecting and fixing cycles
18. **[18_schema_compatibility.py](18_schema_compatibility.py)** - Schema compatibility issues

### 🚀 Advanced Patterns
19. **[19_complex_workflow.py](19_complex_workflow.py)** - Real-world complex pipeline
20. **[20_performance_optimization.py](20_performance_optimization.py)** - Optimization techniques

---

## 🏃‍♂️ Quick Start

```bash
# Run basic examples
python examples/01_basic_dag.py
python examples/02_dependencies.py

# Try validation examples
python examples/03_validation_basics.py
python examples/04_validation_strategies.py

# Explore enterprise features
python examples/13_yaml_pipelines.py
python examples/14_pipeline_compilation.py
```

## 📋 Example Categories

### **Bare DAG API** (Examples 1-12)
- Direct use of `DirectedGraph`, `NodeSpec`, `Orchestrator`
- Core hexAI framework features
- MockPorts for external services
- Manual graph construction

### **Pipelines API** (Examples 13-15)
- YAML-based pipeline definitions
- Pipeline compilation and optimization
- Enterprise features
- Pipeline catalog management

### **Problem Solving** (Examples 16-18)
- Common issues you'll encounter
- Debugging techniques
- Error resolution patterns

### **Advanced Usage** (Examples 19-20)
- Complex real-world scenarios
- Performance optimization
- Best practices

---

## 🛠️ Running Examples

### Prerequisites
```bash
cd fastapi_app
pip install -e .
```

### Individual Examples
```bash
# Basic usage
python examples/01_basic_dag.py

# With debug output
HEXAI_DEBUG=1 python examples/05_event_system.py

# With visualization
python examples/11_dag_visualization.py
```

### All Examples
```bash
# Run learning sequence
python examples/run_all.py

# Run specific category
python examples/run_all.py --category=validation
python examples/run_all.py --category=enterprise
```

---

## 🎯 Learning Objectives

By the end of these examples, you'll understand:

✅ **Core Concepts**
- DAG construction and execution
- Node types and dependencies
- Validation strategies
- Event system and monitoring

✅ **Development Patterns**
- Port/adapter architecture
- Error handling strategies
- Testing approaches
- Performance optimization

✅ **Enterprise Features**
- YAML pipeline definitions
- Pipeline compilation
- Catalog management
- Advanced monitoring

✅ **Troubleshooting**
- Common validation errors
- Dependency issues
- Schema compatibility
- Performance bottlenecks

---

## 📖 Additional Resources

- [hexAI Documentation](../src/hexai/README.md)
- [Implementation Guide](../src/hexai/HEXAI_IMPLEMENTATION_GUIDE.md)
- [Validation Framework](../src/hexai/VALIDATION_FRAMEWORK.md)
- [Development Roadmap](../src/hexai/ROADMAP.md)

---

## 🤝 Contributing Examples

Found a missing use case? Want to add an example?

1. Follow the naming convention: `##_descriptive_name.py`
2. Include comprehensive docstrings
3. Use MockPorts for external dependencies
4. Add to the appropriate category in this README
5. Test your example thoroughly

---

**Happy Learning! 🚀**
