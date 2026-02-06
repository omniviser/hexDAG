import type { NodeTemplate } from '../types'

// Built-in hexDAG node templates
export const nodeTemplates: NodeTemplate[] = [
  {
    kind: 'function_node',
    label: 'Function',
    description: 'Execute Python functions',
    icon: 'Code',
    color: '#3b82f6', // blue-500
    defaultSpec: {
      fn: '',
    },
  },
  {
    kind: 'llm_node',
    label: 'LLM',
    description: 'Language model interactions',
    icon: 'Brain',
    color: '#8b5cf6', // violet-500
    defaultSpec: {
      prompt_template: '',
    },
    requiredPorts: ['llm'],
  },
  {
    kind: 'agent_node',
    label: 'Agent',
    description: 'ReAct agent with tools',
    icon: 'Bot',
    color: '#ec4899', // pink-500
    defaultSpec: {
      initial_prompt_template: '',
      max_steps: 5,
    },
    requiredPorts: ['llm', 'tool_router'],
  },
  {
    kind: 'conditional_node',
    label: 'Conditional',
    description: 'Conditional branching',
    icon: 'GitBranch',
    color: '#f59e0b', // amber-500
    defaultSpec: {
      condition: '',
      if_true: null,
      if_false: null,
    },
  },
  {
    kind: 'loop_node',
    label: 'Loop',
    description: 'Iterative processing',
    icon: 'Repeat',
    color: '#10b981', // emerald-500
    defaultSpec: {
      iterations: 1,
      body: null,
    },
  },
  {
    kind: 'input_node',
    label: 'Input',
    description: 'Pipeline input',
    icon: 'FileText',
    color: '#06b6d4', // cyan-500
    defaultSpec: {
      schema: {},
    },
  },
  {
    kind: 'output_node',
    label: 'Output',
    description: 'Pipeline output',
    icon: 'FileText',
    color: '#14b8a6', // teal-500
    defaultSpec: {
      schema: {},
    },
  },
  {
    kind: 'transform_node',
    label: 'Transform',
    description: 'Data transformation',
    icon: 'Scissors',
    color: '#f97316', // orange-500
    defaultSpec: {
      transform_fn: '',
    },
  },
  {
    kind: 'parallel_node',
    label: 'Parallel',
    description: 'Parallel execution',
    icon: 'Cpu',
    color: '#6366f1', // indigo-500
    defaultSpec: {
      branches: [],
    },
  },
]

// Map of kind to template for quick lookup
const templateMap = new Map<string, NodeTemplate>(
  nodeTemplates.map((t) => [t.kind, t])
)

// Default colors for unknown node kinds
const defaultColors: Record<string, string> = {
  function: '#3b82f6',
  llm: '#8b5cf6',
  agent: '#ec4899',
  conditional: '#f59e0b',
  loop: '#10b981',
  input: '#06b6d4',
  output: '#14b8a6',
  transform: '#f97316',
  parallel: '#6366f1',
  default: '#6b7280',
}

/**
 * Get a node template by kind
 */
export function getNodeTemplate(kind: string): NodeTemplate | undefined {
  // Direct lookup
  if (templateMap.has(kind)) {
    return templateMap.get(kind)
  }

  // Try with _node suffix
  if (templateMap.has(`${kind}_node`)) {
    return templateMap.get(`${kind}_node`)
  }

  // Handle namespaced kinds (e.g., 'etl:file_reader_node')
  if (kind.includes(':')) {
    const baseName = kind.split(':').pop()
    if (baseName && templateMap.has(baseName)) {
      return templateMap.get(baseName)
    }
  }

  return undefined
}

/**
 * Get the color for a node kind
 */
export function getNodeColor(kind: string): string {
  // Try to get from template
  const template = getNodeTemplate(kind)
  if (template) {
    return template.color
  }

  // Try to match based on kind name
  const kindLower = kind.toLowerCase()
  for (const [key, color] of Object.entries(defaultColors)) {
    if (kindLower.includes(key)) {
      return color
    }
  }

  return defaultColors.default
}

/**
 * Get the icon name for a node kind
 */
export function getNodeIcon(kind: string): string {
  const template = getNodeTemplate(kind)
  if (template) {
    return template.icon
  }

  // Default icons based on kind name
  const kindLower = kind.toLowerCase()
  if (kindLower.includes('function')) return 'Code'
  if (kindLower.includes('llm')) return 'Brain'
  if (kindLower.includes('agent')) return 'Bot'
  if (kindLower.includes('conditional') || kindLower.includes('branch'))
    return 'GitBranch'
  if (kindLower.includes('loop') || kindLower.includes('repeat')) return 'Repeat'
  if (kindLower.includes('input') || kindLower.includes('output'))
    return 'FileText'
  if (kindLower.includes('transform')) return 'Scissors'
  if (kindLower.includes('parallel')) return 'Cpu'

  return 'Box'
}

/**
 * Get the default spec for a node kind
 */
export function getDefaultSpec(kind: string): Record<string, unknown> {
  const template = getNodeTemplate(kind)
  return template?.defaultSpec || {}
}

/**
 * Get required ports for a node kind
 */
export function getRequiredPorts(kind: string): string[] {
  const template = getNodeTemplate(kind)
  return template?.requiredPorts || []
}
