import type { NodeTemplate } from '../types'
import {
  Code,
  Brain,
  Bot,
  Plug,
  Package,
  FileInput,
  FileOutput,
  Table,
  Mail,
  Send,
  Layers,
  Calculator,
  Database,
  Wrench,
  type LucideIcon,
} from 'lucide-react'

// Built-in hexDAG node templates (fallback - actual data fetched from API)
// This is used for color/icon lookups when rendering nodes from existing YAML
export const nodeTemplates: NodeTemplate[] = [
  // === Core Processing Nodes ===
  {
    kind: 'function_node',
    label: 'Function',
    description: 'Execute Python functions with optional validation',
    icon: 'Code',
    color: '#3b82f6', // blue-500
    defaultSpec: { fn: '' },
  },
  {
    kind: 'llm_node',
    label: 'LLM',
    description: 'Language model interactions with prompts',
    icon: 'Brain',
    color: '#8b5cf6', // violet-500
    defaultSpec: { prompt_template: '' },
    requiredPorts: ['llm'],
  },
  {
    kind: 're_act_agent_node',
    label: 'ReAct Agent',
    description: 'Multi-step reasoning agent with tool calling',
    icon: 'Bot',
    color: '#ec4899', // pink-500
    defaultSpec: { main_prompt: '', config: { max_steps: 5 } },
    requiredPorts: ['llm', 'tool_router'],
  },

  // === Control Flow Nodes ===
  {
    kind: 'composite_node',
    label: 'Composite',
    description: 'Unified control flow (while, for-each, if-else, switch)',
    icon: 'Layers',
    color: '#6366f1', // indigo-500
    defaultSpec: { mode: 'for-each', items: '', body: [] },
  },

  // === Data Nodes ===
  {
    kind: 'expression_node',
    label: 'Expression',
    description: 'Compute values using safe expressions and merge data',
    icon: 'Calculator',
    color: '#06b6d4', // cyan-500
    defaultSpec: { expressions: {} },
  },

  // === Integration Nodes ===
  {
    kind: 'tool_call_node',
    label: 'Tool Call',
    description: 'Execute a tool function as a node',
    icon: 'Wrench',
    color: '#f97316', // orange-500
    defaultSpec: { tool_name: '', arguments: {} },
  },
  {
    kind: 'port_call_node',
    label: 'Port Call',
    description: 'Call a method on a configured port/adapter',
    icon: 'Plug',
    color: '#84cc16', // lime-500
    defaultSpec: { port: '', method: '', input_mapping: {} },
  },
]

// Map of kind to template for quick lookup
const templateMap = new Map<string, NodeTemplate>(
  nodeTemplates.map((t) => [t.kind, t])
)

// Plugin node metadata registry - populated by API calls
// Stores config schemas and other metadata from loaded plugins
export interface PluginNodeMetadata {
  kind: string
  name: string
  description: string
  icon: string
  color: string
  configSchema: Record<string, unknown>
  plugin: string
}

const pluginNodeRegistry = new Map<string, PluginNodeMetadata>()

/**
 * Register plugin node metadata (called when plugins are loaded from API)
 */
export function registerPluginNode(metadata: PluginNodeMetadata): void {
  pluginNodeRegistry.set(metadata.kind, metadata)
}

/**
 * Register multiple plugin nodes at once
 */
export function registerPluginNodes(nodes: PluginNodeMetadata[]): void {
  for (const node of nodes) {
    pluginNodeRegistry.set(node.kind, node)
  }
  // Debug: log registry contents
  console.log('[NodeRegistry] Registered nodes:', Array.from(pluginNodeRegistry.keys()))
  // Log one example with schema
  const firstWithSchema = nodes.find(n => n.configSchema && Object.keys(n.configSchema).length > 0)
  if (firstWithSchema) {
    console.log('[NodeRegistry] Example node with schema:', firstWithSchema.kind, 'properties:', Object.keys((firstWithSchema.configSchema as any)?.properties || {}))
  }
}

/**
 * Get plugin node metadata by kind
 */
export function getPluginNodeMetadata(kind: string): PluginNodeMetadata | undefined {
  return pluginNodeRegistry.get(kind)
}

/**
 * Get config schema for a node kind (checks both builtin and plugin nodes)
 */
export function getNodeConfigSchema(kind: string): Record<string, unknown> | undefined {
  // Check plugin nodes first (direct lookup)
  let pluginMeta = pluginNodeRegistry.get(kind)

  // If not found and kind has namespace prefix, try without prefix
  if (!pluginMeta && kind.includes(':')) {
    const baseName = kind.split(':').pop()
    if (baseName) {
      pluginMeta = pluginNodeRegistry.get(baseName)
    }
  }

  console.log('[NodeRegistry] getNodeConfigSchema for:', kind, 'found:', !!pluginMeta, 'hasSchema:', !!(pluginMeta?.configSchema))
  if (pluginMeta?.configSchema) {
    const props = (pluginMeta.configSchema as any)?.properties
    console.log('[NodeRegistry] Schema properties:', props ? Object.keys(props) : 'none')
    return pluginMeta.configSchema
  }
  return undefined
}

// Add aliases for common alternative names
const kindAliases: Record<string, string> = {
  // Legacy aliases (for backward compatibility with existing YAML files)
  'agent_node': 're_act_agent_node',
  'conditional_node': 'composite_node',
  'loop_node': 'composite_node',
  'data_node': 'expression_node',
  'static_node': 'expression_node',
  // Short aliases (core: prefix)
  'core:function': 'function_node',
  'core:function_node': 'function_node',
  'core:llm': 'llm_node',
  'core:llm_node': 'llm_node',
  'core:agent': 're_act_agent_node',
  'core:re_act_agent': 're_act_agent_node',
  'core:re_act_agent_node': 're_act_agent_node',
  'core:composite': 'composite_node',
  'core:composite_node': 'composite_node',
  'core:expression': 'expression_node',
  'core:expression_node': 'expression_node',
  'core:tool_call': 'tool_call_node',
  'core:tool_call_node': 'tool_call_node',
  'core:port_call': 'port_call_node',
  'core:port_call_node': 'port_call_node',
}

// Default colors for unknown node kinds
const defaultColors: Record<string, string> = {
  function: '#3b82f6',
  llm: '#8b5cf6',
  agent: '#ec4899',
  composite: '#6366f1',
  expression: '#06b6d4',
  tool: '#f97316',
  port: '#84cc16',
  // Legacy - map to new colors
  conditional: '#6366f1',
  loop: '#6366f1',
  data: '#06b6d4',
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

  // Check aliases
  const aliasedKind = kindAliases[kind]
  if (aliasedKind && templateMap.has(aliasedKind)) {
    return templateMap.get(aliasedKind)
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
    // Also try alias lookup for namespaced kinds
    if (baseName) {
      const aliasedBase = kindAliases[kind] || kindAliases[baseName]
      if (aliasedBase && templateMap.has(aliasedBase)) {
        return templateMap.get(aliasedBase)
      }
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
  if (kindLower.includes('composite') || kindLower.includes('control')) return 'Layers'
  if (kindLower.includes('expression') || kindLower.includes('calc')) return 'Calculator'
  if (kindLower.includes('tool')) return 'Wrench'
  if (kindLower.includes('port')) return 'Plug'
  // Legacy mappings
  if (kindLower.includes('conditional') || kindLower.includes('branch')) return 'Layers'
  if (kindLower.includes('loop') || kindLower.includes('repeat')) return 'Layers'
  if (kindLower.includes('data') || kindLower.includes('static')) return 'Calculator'

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

// Icon component mapping for core node types
const coreIconMap: Record<string, LucideIcon> = {
  function_node: Code,
  llm_node: Brain,
  re_act_agent_node: Bot,
  agent_node: Bot, // legacy alias
  composite_node: Layers,
  conditional_node: Layers, // legacy, now uses composite
  loop_node: Layers, // legacy, now uses composite
  expression_node: Calculator,
  data_node: Calculator, // legacy alias
  tool_call_node: Wrench,
  port_call_node: Plug,
}

/**
 * Get the icon component for a plugin node based on its kind.
 * Uses heuristics to match node kind to appropriate icon.
 */
export function getPluginNodeIcon(kind: string): LucideIcon {
  const kindLower = kind.toLowerCase()
  if (kindLower.includes('file_reader') || kindLower.includes('input')) return FileInput
  if (kindLower.includes('file_writer') || kindLower.includes('output')) return FileOutput
  if (kindLower.includes('outlook_reader') || kindLower.includes('mail_reader')) return Mail
  if (kindLower.includes('outlook_sender') || kindLower.includes('mail_sender')) return Send
  if (kindLower.includes('transform') || kindLower.includes('pandas')) return Table
  if (kindLower.includes('llm') || kindLower.includes('openai')) return Brain
  if (kindLower.includes('database') || kindLower.includes('sql') || kindLower.includes('cosmos')) return Database
  if (kindLower.includes('tool')) return Wrench
  if (kindLower.includes('port') || kindLower.includes('adapter')) return Plug
  if (kindLower.includes('expression') || kindLower.includes('calc')) return Calculator
  if (kindLower.includes('composite') || kindLower.includes('control')) return Layers
  return Package
}

/**
 * Get the Lucide icon component for a node kind.
 * Works for both core nodes (via direct mapping) and plugin nodes (via heuristics).
 */
export function getNodeIconComponent(kind: string): LucideIcon {
  // Direct lookup for core nodes
  if (coreIconMap[kind]) {
    return coreIconMap[kind]
  }

  // Handle namespaced kinds (e.g., 'etl:file_reader_node')
  if (kind.includes(':')) {
    const baseName = kind.split(':').pop()
    if (baseName && coreIconMap[baseName]) {
      return coreIconMap[baseName]
    }
  }

  // Use plugin node heuristics
  return getPluginNodeIcon(kind)
}
