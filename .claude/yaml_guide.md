# YAML Pipeline Syntax and Templating Guide

YAML configuration files are compiled by the `agent_factory` into executable DirectedGraph objects.

## I. Full Pipeline Structure Example
(Based on README.md and example 13)

```yaml
nodes:
  - type: agent
    id: researcher
    params:
      initial_prompt_template: "Research: {{variables.research_topic}}"
      available_tools: ["web_search"]
    depends_on: [] # Start node

  - type: llm
    id: analyst
    params:
      prompt_template: "Analyze findings: {{researcher.results}}"
    depends_on: [researcher]