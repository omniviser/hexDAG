import type { Node, Edge } from '@xyflow/react'

// File types
export interface FileInfo {
  name: string
  path: string
  is_directory: boolean
  size?: number
  modified?: number
}

export interface FileContent {
  path: string
  content: string
  modified: number
}

// Validation types
export interface ValidationError {
  line?: number
  column?: number
  message: string
  severity: 'error' | 'warning' | 'info'
}

export interface ValidationResult {
  valid: boolean
  errors: ValidationError[]
  node_count?: number
  nodes?: string[]
}

// Pipeline types
export interface PipelineMetadata {
  name: string
  description?: string
}

export interface NodeSpec {
  kind: string
  metadata: {
    name: string
    description?: string
  }
  spec: Record<string, unknown>
  dependencies: string[]
}

export interface PipelineSpec {
  ports?: Record<string, unknown>
  nodes: NodeSpec[]
}

export interface Pipeline {
  apiVersion: string
  kind: string
  metadata: PipelineMetadata
  spec: PipelineSpec
}

// Canvas node types - must have index signature for React Flow compatibility
export interface HexdagNodeData extends Record<string, unknown> {
  kind: string
  label: string
  spec: Record<string, unknown>
  isValid: boolean
  errors: string[]
}

export type HexdagNode = Node<HexdagNodeData>
export type HexdagEdge = Edge

// Node palette types
export interface NodeTemplate {
  kind: string
  label: string
  description: string
  icon: string
  color: string
  defaultSpec: Record<string, unknown>
  requiredPorts?: string[]  // Ports this node type requires (e.g., ['llm', 'tool_router'])
}

// Port configuration for a node
export interface NodePortConfig {
  adapter: string  // e.g., 'core:openai' or 'plugin:azure_openai'
  config?: Record<string, unknown>
}

// Execution status for individual nodes
export type NodeExecutionStatus = 'idle' | 'pending' | 'running' | 'completed' | 'failed'

export interface NodeExecutionState {
  status: NodeExecutionStatus
  output?: unknown
  error?: string
  duration_ms?: number
}

// Store types
export interface StudioState {
  // Files
  files: FileInfo[]
  currentFile: string | null
  yamlContent: string

  // Canvas
  nodes: HexdagNode[]
  edges: HexdagEdge[]

  // Validation
  validation: ValidationResult | null

  // Execution state - tracks live execution status for canvas nodes
  nodeExecutionStatus: Map<string, NodeExecutionState>
  isExecuting: boolean

  // UI state
  isLoading: boolean
  isSaving: boolean
  isDirty: boolean

  // Actions
  setFiles: (files: FileInfo[]) => void
  setCurrentFile: (path: string | null) => void
  setYamlContent: (content: string) => void
  setNodes: (nodes: HexdagNode[]) => void
  setEdges: (edges: HexdagEdge[]) => void
  setValidation: (result: ValidationResult | null) => void
  setIsLoading: (loading: boolean) => void
  setIsSaving: (saving: boolean) => void
  setIsDirty: (dirty: boolean) => void

  // Execution actions
  setIsExecuting: (executing: boolean) => void
  setNodeExecutionStatus: (nodeName: string, state: NodeExecutionState) => void
  resetExecutionStatus: () => void
  initializeExecutionStatus: (nodeNames: string[]) => void

  // Complex actions
  syncYamlToCanvas: () => void
  syncCanvasToYaml: () => void
}
