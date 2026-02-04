# hexDAG Variable Reference Syntax

## 1. Initial Input Reference: $input

Use `$input.field` in `input_mapping` to access the original pipeline input.

```yaml
nodes:
  - kind: function_node
    metadata:
      name: processor
    spec:
      fn: myapp.process
      input_mapping:
        load_id: $input.load_id
        carrier: $input.carrier_mc
    dependencies: [extractor]
```

## 2. Node Output Reference: {{node.field}}

Use Jinja2 syntax in prompt templates to reference previous node outputs.

```yaml
- kind: llm_node
  metadata:
    name: analyzer
  spec:
    prompt_template: |
      Analyze this data:
      {{extractor.result}}
```

## 3. Environment Variables: ${VAR}

```yaml
spec:
  ports:
    llm:
      config:
        model: ${MODEL}              # Resolved at build time
        api_key: ${OPENAI_API_KEY}   # Secret - resolved at runtime
```

**Secret Patterns (deferred to runtime):**
- `*_API_KEY`, `*_SECRET`, `*_TOKEN`, `*_PASSWORD`, `*_CREDENTIAL`, `SECRET_*`

## 4. Input Mapping

```yaml
- kind: function_node
  metadata:
    name: merger
  spec:
    fn: myapp.merge_results
    input_mapping:
      request_id: $input.id          # From initial input
      analysis: analyzer.result       # From dependency
  dependencies: [analyzer]
```

## Quick Reference

| Syntax | Location | Purpose |
|--------|----------|---------|
| `$input.field` | input_mapping | Access initial pipeline input |
| `{{node.field}}` | prompt_template | Jinja2 template reference |
| `${VAR}` | Any string | Environment variable |
| `${VAR:default}` | Any string | Env var with default |
| `node.path` | input_mapping | Dependency output extraction |
