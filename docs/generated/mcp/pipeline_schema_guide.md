# hexDAG Pipeline Schema Reference

This reference is auto-generated from the pipeline JSON schema.

## Overview

hexDAG pipelines are defined in YAML using a Kubernetes-like structure.
The schema provides validation and IDE autocompletion support.

## Pipeline Structure

```yaml
apiVersion: hexdag/v1
kind: Pipeline
metadata:
  name: my-pipeline
  description: Pipeline description
spec:
  ports: {}     # Adapter configurations
  nodes: []     # Processing nodes
  events: {}    # Event handlers
```

## IDE Setup

### VS Code

Add to `.vscode/settings.json`:

```json
{
  "yaml.schemas": {
    "./schemas/pipeline-schema.json": ["*.yaml", "pipelines/*.yaml"]
  }
}
```

### Schema Location

The schema file is at `schemas/pipeline-schema.json` and is auto-generated from node `_yaml_schema` attributes.
