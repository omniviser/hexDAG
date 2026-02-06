import type { FileInfo, FileContent, ValidationResult } from '../types'

const API_BASE = '/api'

// Plugin types
export interface PluginNode {
  kind: string
  name: string
  description?: string
  plugin: string
  color: string
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
  return apiRequest<FileInfo[]>(`/files?path=${encodeURIComponent(path)}`)
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
  yaml: string,
  inputs: Record<string, unknown> = {},
  dryRun: boolean = false
): Promise<ExecutionResult> {
  return apiRequest<ExecutionResult>('/execute', {
    method: 'POST',
    body: JSON.stringify({
      yaml,
      inputs,
      dry_run: dryRun,
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
    const plugins = await listPlugins()
    const nodes: PluginNode[] = []

    for (const plugin of plugins) {
      const pluginNodes = Array.isArray(plugin.nodes) ? plugin.nodes : []
      for (const node of pluginNodes) {
        nodes.push({
          ...node,
          plugin: plugin.name,
        })
      }
    }

    return nodes
  } catch (error) {
    console.error('Failed to get plugin nodes:', error)
    return []
  }
}

export async function getAllPluginAdapters(): Promise<PluginAdapter[]> {
  try {
    const plugins = await listPlugins()
    const adapters: PluginAdapter[] = []

    for (const plugin of plugins) {
      const pluginAdapters = Array.isArray(plugin.adapters) ? plugin.adapters : []
      for (const adapter of pluginAdapters) {
        adapters.push({
          ...adapter,
          plugin: plugin.name,
        })
      }
    }

    return adapters
  } catch (error) {
    console.error('Failed to get plugin adapters:', error)
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

// Registry endpoints
export async function getRegisteredNodes(): Promise<
  Array<{ kind: string; name: string; namespace: string }>
> {
  try {
    return await apiRequest('/registry/nodes')
  } catch {
    return []
  }
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
