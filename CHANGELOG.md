## Version Release: 0.1.0
Merged PR 2166: feat: add cz

feat: add cz

## Version Release: 0.1.1
Merged PR 2168: fix: add uv

fix: add uv

## Version Release: 0.1.2
Merged PR 2170: test: JEAN PLS MERGE ME TO TEST THE CI PIPELINE

If you merge PR there will be bump based on the prefix in PR title
```java
bump_map = {
  ".*!" = "MAJOR",
  "ci" = "PATCH",
  "docs" = "PATCH",
  "experiment" = "PATCH",
  "feat" = "MINOR",
  "fix" = "PATCH",
  "refactor" = "PATCH",
  "test" = "PATCH"
}
```
And the `CHANGELOG.md` file will be updated with the PR description and version that was bumped

In ci-pipeline.yaml there is a place to put logic for jobs triggered once per version update of the project

Related work items: #2082

## Version Release: 0.2.0
Merged PR 2101: feat: function-level decorators

Related work items: #1902

## v0.3.0-a3 (2025-10-19) - Latest

### Features

#### Macro System
- **ConversationMacro**: Multi-turn conversation support with configurable memory port
- **LLMMacro**: Simplified LLM invocation with YAML schema conversion
- **ToolMacro**: Tool integration macro for function calling
- **ReasoningAgentMacro**: Advanced reasoning patterns for complex decision-making
- Macro expansion strategies: STATIC, DYNAMIC, and LAZY
- Full YAML integration with `macro_invocation` support
- Smart dependency validation for macro-generated nodes

#### Node Improvements
- **ToolNode**: New node type for tool execution
- **ConfigurableNode**: Base class for custom node development
- **ConfigurablePolicy**: Base class for custom policies
- Custom node support without `_node` suffix requirement

#### YAML Pipeline Enhancements
- Simplified YAML builder with better error messages
- Plugin node support in YAML validator
- Environment-based configuration management
- Improved template system with Jinja2 support

#### Infrastructure
- Local executor support for simplified deployment
- Storage plugin system with PostgreSQL support
- Azure Pipelines CI/CD integration
- Comprehensive notebook examples (6 interactive notebooks)
- Improved test coverage (73+ tests for macro system alone)

### Bug Fixes
- Fixed YAML validator to accept plugin nodes without `_node` suffix
- Added YAML schema conversion to LLMMacroConfig
- Resolved namespace conflicts in registry

### Code Refactoring
- Simplified LLMMacroConfig by removing unnecessary overrides
- Removed over-engineered registry patterns
- Consolidated storage as part of main package
- Code quality improvements and deduplication
- Removed legacy namespace requirements

### Documentation
- Added comprehensive macro system documentation
- Updated notebooks with advanced patterns
- Enhanced YAML pipeline examples
- Improved API documentation

## v0.3.0-a2 (2025-10-10)

### Features
- Enhanced plugin system architecture
- Improved registry patterns

## v0.3.0-a1 (2025-10-09)

### Features

- Added the version

## v0.2.0 (2025-10-09)

### Bug Fixes

- removed girffe

### Code Refactoring

- refactor
- Optimized the code
- Removed the overengineered refactor of the registry/
- Added the execution context to not to respawn the ports.
- Refactored the orchestrator god class

### Features

- Added more CLI
- refactored the bugs
- Added the optimizations
- Observers and Controlers
- Added the docker buid
- Added localSecrets adapter
- mkdocs and CLi for yaml
- Added the yaml builder
- Added the schema generator and adapter component
- Added azure pipelines
- Added the test coverage
- Added doctest and type annotation
- Added doctest and type annotation
- Added logguru
- DAG runtime improvements
- Centralized logging
- removed even more code smells
- Fixed bugs
- Added the Protocols and Exceptions
- Added pre and post DAG hooks
- Added the enchanced point builder
- Added the checkpoints
- added no cov for integration tests
- added the async warnings
- Added the async methods to the tool router
- Added the asyncio timeout
- Added integration tests
- Removed the god class for the registry.
- initial hexDAG standalone repository
