import type { FileInfo, FileContent, ValidationResult } from '../types'
import { registerPluginNodes, type PluginNodeMetadata } from './nodeTemplates'

const API_BASE = '/api'

// Plugin types
export interface PluginNode {
  kind: string
  name: string
  description?: string
  plugin: string
  color: string
  icon?: string
  config_schema?: Record<string, unknown>
  defaultSpec?: Record<string, unknown>
}

export interface PluginAdapter {
  name: string
  port_type: string
  plugin: string
  description?: string
  config_schema?: Record<string, unknown>
  secrets: string[]
}

export interface PluginInfo {
  name: string
  version: string
  description?: string
  author?: string
  adapters: PluginAdapter[]
  nodes: PluginNode[]
  enabled: boolean
}

// Execution types
export interface ExecutionResult {
  success: boolean
  outputs?: Record<string, unknown>
  error?: string
  execution_time?: number
  duration_ms: number
  node_results?: Record<string, unknown>
}

// Helper function for API requests
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${endpoint}`
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(`API error: ${response.status} - ${error}`)
  }

  return response.json()
}

// File operations
export async function listFiles(path: string = '.'): Promise<FileInfo[]> {
  const response = await apiRequest<FileInfo[] | { files: FileInfo[]; root?: string }>(`/files?path=${encodeURIComponent(path)}`)
  // Handle both array response and object with files property
  if (Array.isArray(response)) {
    return response
  }
  if (response && typeof response === 'object' && 'files' in response && Array.isArray(response.files)) {
    return response.files
  }
  console.warn('Unexpected files response format:', response)
  return []
}

export async function readFile(path: string): Promise<FileContent> {
  return apiRequest<FileContent>(`/files/${encodeURIComponent(path)}`)
}

export async function saveFile(path: string, content: string): Promise<void> {
  await apiRequest(`/files/${encodeURIComponent(path)}`, {
    method: 'PUT',
    body: JSON.stringify({ content }),
  })
}

export async function deleteFile(path: string): Promise<void> {
  await apiRequest(`/files/${encodeURIComponent(path)}`, {
    method: 'DELETE',
  })
}

// Validation
export async function validateYaml(
  content: string,
  filePath?: string
): Promise<ValidationResult> {
  return apiRequest<ValidationResult>('/validate', {
    method: 'POST',
    body: JSON.stringify({ content, file_path: filePath }),
  })
}

// Execution
export async function executePipeline(
  content: string,
  inputs: Record<string, unknown> = {},
  dryRun: boolean = false
): Promise<ExecutionResult> {
  const endpoint = dryRun ? '/execute/dry-run' : '/execute'
  return apiRequest<ExecutionResult>(endpoint, {
    method: 'POST',
    body: JSON.stringify({
      content,
      inputs,
      use_mocks: true,
      timeout: 30,
    }),
  })
}

// Project operations
export async function downloadProject(
  yamlContent?: string,
  filename?: string,
  includeAssets?: boolean
): Promise<void> {
  const response = await fetch(`${API_BASE}/project/download`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      yaml: yamlContent,
      filename,
      include_assets: includeAssets,
    }),
  })
  if (!response.ok) {
    throw new Error(`Failed to download project: ${response.status}`)
  }

  const blob = await response.blob()
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename || 'hexdag-project.zip'
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  window.URL.revokeObjectURL(url)
}

// Plugin discovery
export async function getAllPluginNodes(): Promise<PluginNode[]> {
  try {
    console.log('[NodeRegistry] Fetching plugin nodes...')
    const plugins = await listPlugins()
    console.log('[NodeRegistry] Got', plugins.length, 'plugins from API')
    const nodes: PluginNode[] = []

    for (const plugin of plugins) {
      const pluginNodes = Array.isArray(plugin.nodes) ? plugin.nodes : []
      console.log('[NodeRegistry] Plugin', plugin.name, 'has', pluginNodes.length, 'nodes')
      for (const node of pluginNodes) {
        // Debug: log raw node data from API
        console.log('[NodeRegistry] Raw node from API:', node.kind, 'config_schema:', node.config_schema ? Object.keys(node.config_schema) : 'MISSING')
        nodes.push({
          ...node,
          plugin: plugin.name,
        })
      }
    }

    // Register plugin nodes for use by NodeInspector (schema lookup)
    const metadata: PluginNodeMetadata[] = nodes.map((node) => ({
      kind: node.kind,
      name: node.name,
      description: node.description || '',
      icon: node.icon || 'Package',
      color: node.color,
      configSchema: node.config_schema || {},
      plugin: node.plugin,
    }))

    // Log sample plugin schema
    const withSchema = metadata.find(n => n.configSchema && Object.keys(n.configSchema).length > 0)
    if (withSchema) {
      console.log('[NodeRegistry] Plugin node sample schema:', withSchema.kind, (withSchema.configSchema as any)?.properties ? Object.keys((withSchema.configSchema as any).properties) : 'no properties')
    }

    registerPluginNodes(metadata)

    return nodes
  } catch (error) {
    console.error('[NodeRegistry] Failed to get plugin nodes:', error)
    return []
  }
}

export async function getAllPluginAdapters(): Promise<PluginAdapter[]> {
  try {
    // Use registry/adapters endpoint which includes both built-in and plugin adapters
    const response = await apiRequest<PluginAdapter[]>('/registry/adapters')
    return Array.isArray(response) ? response : []
  } catch (error) {
    console.error('Failed to get adapters:', error)
    return []
  }
}

export async function listPlugins(): Promise<PluginInfo[]> {
  try {
    const response = await apiRequest<PluginInfo[] | { plugins: PluginInfo[] }>('/plugins')
    // Handle both array response and object with plugins property
    if (Array.isArray(response)) {
      return response
    }
    if (response && typeof response === 'object' && 'plugins' in response && Array.isArray(response.plugins)) {
      return response.plugins
    }
    console.warn('Unexpected plugins response format:', response)
    return []
  } catch (error) {
    console.error('Failed to list plugins:', error)
    return []
  }
}

// Registry endpoints - built-in hexDAG node types
export interface BuiltinNodeType {
  kind: string
  name: string
  description: string
  namespace: string
  color: string
  icon: string
  default_spec: Record<string, unknown>
  required_ports: string[]
  config_schema: Record<string, unknown>
}

export async function getBuiltinNodes(includeDeprecated: boolean = false): Promise<BuiltinNodeType[]> {
  try {
    const response = await apiRequest<{ nodes: BuiltinNodeType[] }>(
      `/registry/nodes?include_deprecated=${includeDeprecated}`
    )
    return response.nodes || []
  } catch (error) {
    console.error('Failed to get builtin nodes:', error)
    return []
  }
}

/**
 * Fetch and register all core (builtin) node types with their schemas.
 * This populates the node registry for use by NodeInspector.
 */
export async function getAllCoreNodes(): Promise<PluginNodeMetadata[]> {
  try {
    console.log('[NodeRegistry] Fetching core nodes...')
    const nodes = await getBuiltinNodes()
    console.log('[NodeRegistry] Got', nodes.length, 'core nodes from API')

    const metadata: PluginNodeMetadata[] = nodes.map((node) => ({
      kind: node.kind,
      name: node.name,
      description: node.description || '',
      icon: node.icon || 'Package',
      color: node.color,
      configSchema: node.config_schema || {},
      plugin: 'builtin',
    }))

    // Log sample schema
    const withSchema = metadata.find(n => n.configSchema && Object.keys(n.configSchema).length > 0)
    if (withSchema) {
      console.log('[NodeRegistry] Core node sample schema:', withSchema.kind, (withSchema.configSchema as any)?.properties ? Object.keys((withSchema.configSchema as any).properties) : 'no properties')
    }

    registerPluginNodes(metadata)
    return metadata
  } catch (error) {
    console.error('[NodeRegistry] Failed to get core nodes:', error)
    return []
  }
}

/**
 * Initialize the node registry with both core and plugin nodes.
 * Call this once on app startup to populate schemas for all node types.
 */
export async function initializeNodeRegistry(): Promise<void> {
  console.log('[NodeRegistry] Initializing node registry...')
  try {
    const [coreNodes, pluginNodes] = await Promise.all([getAllCoreNodes(), getAllPluginNodes()])
    console.log('[NodeRegistry] Initialized:', coreNodes.length, 'core nodes,', pluginNodes.length, 'plugin nodes')
  } catch (error) {
    console.error('[NodeRegistry] Failed to initialize:', error)
  }
}

// Legacy - keeping for backward compatibility
export async function getRegisteredNodes(): Promise<
  Array<{ kind: string; name: string; namespace: string }>
> {
  const nodes = await getBuiltinNodes()
  return nodes.map((n) => ({ kind: n.kind, name: n.name, namespace: n.namespace }))
}

export async function getRegisteredAdapters(): Promise<
  Array<{ name: string; port_type: string; namespace: string }>
> {
  try {
    return await apiRequest('/registry/adapters')
  } catch {
    return []
  }
}
