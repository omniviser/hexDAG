# hexDAG CLI Reference

This reference is auto-generated from `hexdag --help`.

## Overview

```
Usage: hexdag [OPTIONS] COMMAND [ARGS]...

 HexDAG - Lightweight DAG orchestration framework with hexagonal architecture.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --version             -v        Show version and exit                        │
│ --install-completion            Install completion for the current shell.    │
│ --show-completion               Show completion for the current shell, to    │
│                                 copy it or customize the installation.       │
│ --help                          Show this message and exit.                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ build           Build Docker containers for pipelines                        │
│ generate-types  Generate type stubs from YAML pipelines                      │
│ lint            Lint YAML pipeline files for best practices                  │
│ validate        Validate YAML pipeline files                                 │
│ create          Create pipeline templates from schemas                       │
│ docs            Generate and serve documentation                             │
│ init            Initialize a new HexDAG project                              │
│ pipeline        Pipeline validation and execution                            │
│ plugin          Plugin development commands                                  │
│ plugins         Manage plugins and adapters                                  │
│ studio          Visual editor for pipelines                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Commands

### `hexdag build`

Build Docker containers for pipelines

```
Usage: hexdag build [OPTIONS] PIPELINE...

 Build Docker containers for pipelines

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    pipeline      PIPELINE...  Pipeline YAML file(s) to build [required]    │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --output          -o                    TEXT  Output directory for Docker    │
│                                               files (default: ./build)       │
│                                               [default: ./build]             │
│ --image           -i                    TEXT  Docker image name (default:    │
│                                               hexdag-<pipeline-name>)        │
│ --python-version  -p                    TEXT  Python version for base image  │
│                                               [default: 3.12]                │
│ --base-image      -b                    TEXT  Custom base Docker image       │
│ --compose         -c  --no-compose  -C        Generate docker-compose.yml    │
│                                               for multi-pipeline             │
│                                               orchestration                  │
│                                               [default: compose]             │
│ --local           -l                          Install hexdag from local      │
│                                               source (copies hexdag/         │
│                                               directory to build context)    │
│ --extras          -e                    TEXT  Comma-separated list of extras │
│                                               to install (e.g.,              │
│                                               'yaml,openai,anthropic,cli').  │
│                                               Available: yaml, viz, openai,  │
│                                               anthropic, database, cli,      │
│                                               docs, all                      │
│                                               [default:                      │
│                                               yaml,openai,anthropic,cli]     │
│ --help                                        Show this message and exit.    │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### `hexdag create`

Create pipeline templates from schemas

```
Usage: hexdag create [OPTIONS] COMMAND [ARGS]...

 Create pipeline templates from schemas

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ pipeline  Create a new pipeline YAML file from a template.                   │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### `hexdag docs`

Generate and serve documentation

```
Usage: hexdag docs [OPTIONS] COMMAND [ARGS]...

 Generate and serve documentation

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ build   Build documentation site with MkDocs.                                │
│ serve   Serve documentation with live reload.                                │
│ deploy  Deploy documentation to GitHub Pages.                                │
│ new     Create a new documentation page.                                     │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### `hexdag generate-types`

Generate type stubs from YAML pipelines

```
Usage: hexdag generate-types [OPTIONS] YAML_PATH

 Generate type stubs from YAML pipelines

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    yaml_path      FILE  Path to YAML pipeline file [required]              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --output-dir  -o      PATH  Output directory for stub files (default:        │
│                             current directory)                               │
│                             [default: .]                                     │
│ --prefix      -p      TEXT  Prefix for generated file names                  │
│ --help                      Show this message and exit.                      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### `hexdag init`

Initialize a new HexDAG project

```
Usage: hexdag init [OPTIONS] [PATH] COMMAND [ARGS]...

 Initialize a new HexDAG project

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│   path      [PATH]  Directory to initialize (defaults to current directory)  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --with           TEXT  Comma-separated list of adapters to include (e.g.,    │
│                        openai,anthropic,local)                               │
│ --force  -f            Overwrite existing configuration                      │
│ --help                 Show this message and exit.                           │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### `hexdag lint`

Lint YAML pipeline files for best practices

```
Usage: hexdag lint [OPTIONS] YAML_FILE

 Lint YAML pipeline files for best practices

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    yaml_file      FILE  Path to YAML pipeline file to lint [required]      │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --severity  -s      TEXT  Minimum severity to report (error, warning, info)  │
│                           [default: info]                                    │
│ --format    -f      TEXT  Output format (text, json) [default: text]         │
│ --disable   -d      TEXT  Comma-separated rule IDs to skip (e.g., W200,W201) │
│ --help                    Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### `hexdag pipeline`

Pipeline validation and execution

```
Usage: hexdag pipeline [OPTIONS] COMMAND [ARGS]...

 Pipeline validation and execution

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ validate  Validate pipeline file (schema + DAG validation).                  │
│ plan      Show execution plan (waves, concurrency, expected I/O).            │
│ run       Execute a pipeline with optional input data.                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### `hexdag plugin`

Plugin development commands

```
Usage: hexdag plugin [OPTIONS] COMMAND [ARGS]...

 Plugin development commands

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ new          Create a new plugin from template.                              │
│ list         List all available plugins.                                     │
│ lint         Lint a plugin's code.                                           │
│ format       Format a plugin's code.                                         │
│ test         Run tests for a plugin.                                         │
│ install      Install a plugin in development mode.                           │
│ conventions  Run convention checks (exception hierarchy, protocols, timer,   │
│              init params).                                                   │
│ check-all    Run lint, test, and convention checks for all plugins.          │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### `hexdag plugins`

Manage plugins and adapters

```
Usage: hexdag plugins [OPTIONS] COMMAND [ARGS]...

 Manage plugins and adapters

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ list     List all available plugins and adapters.                            │
│ check    Check plugin dependencies and suggest installation commands.        │
│ install  Install a plugin or adapter (wrapper around package manager).       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### `hexdag studio`

Visual editor for pipelines

```
Usage: hexdag studio [OPTIONS] [PATH] COMMAND [ARGS]...

 Visual editor for pipelines

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│   path      [PATH]  Directory containing pipeline YAML files [default: .]    │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --host        -h      TEXT     Host to bind to [default: 127.0.0.1]          │
│ --port        -p      INTEGER  Port to bind to [default: 3141]               │
│ --no-browser                   Don't open browser automatically              │
│ --help                         Show this message and exit.                   │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### `hexdag validate`

Validate YAML pipeline files

```
Usage: hexdag validate [OPTIONS] YAML_FILE

 Validate YAML pipeline files

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    yaml_file      FILE  Path to YAML pipeline file to validate [required]  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --explain  -e        Show detailed explanation of validation process         │
│ --help               Show this message and exit.                             │
╰──────────────────────────────────────────────────────────────────────────────╯
```
