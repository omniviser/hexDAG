import { create } from 'zustand'
import yaml from 'yaml'
import type {
  FileInfo,
  HexdagNode,
  HexdagEdge,
  ValidationResult,
  StudioState,
  Pipeline,
  NodeSpec,
  NodeExecutionState,
} from '../types'

// Helper to calculate node positions in a HORIZONTAL DAG layout (left-to-right)
function calculateNodePositions(
  nodes: NodeSpec[]
): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>()

  // Build dependency graph
  const dependencyMap = new Map<string, string[]>()
  const dependentMap = new Map<string, string[]>()

  nodes.forEach((node) => {
    dependencyMap.set(node.metadata.name, node.dependencies || [])
    dependentMap.set(node.metadata.name, [])
  })

  // Build reverse dependency map
  nodes.forEach((node) => {
    ;(node.dependencies || []).forEach((dep) => {
      const dependents = dependentMap.get(dep) || []
      dependents.push(node.metadata.name)
      dependentMap.set(dep, dependents)
    })
  })

  // Calculate levels (topological sort with levels)
  const levels = new Map<string, number>()
  const visited = new Set<string>()

  function calculateLevel(nodeName: string): number {
    if (levels.has(nodeName)) {
      return levels.get(nodeName)!
    }
    if (visited.has(nodeName)) {
      return 0 // Cycle detected, return 0
    }

    visited.add(nodeName)
    const deps = dependencyMap.get(nodeName) || []
    let maxDepLevel = -1

    deps.forEach((dep) => {
      const depLevel = calculateLevel(dep)
      maxDepLevel = Math.max(maxDepLevel, depLevel)
    })

    const level = maxDepLevel + 1
    levels.set(nodeName, level)
    return level
  }

  nodes.forEach((node) => calculateLevel(node.metadata.name))

  // Group nodes by level
  const levelGroups = new Map<number, string[]>()
  levels.forEach((level, nodeName) => {
    const group = levelGroups.get(level) || []
    group.push(nodeName)
    levelGroups.set(level, group)
  })

  // Calculate positions with HORIZONTAL layout (left-to-right flow)
  // Each "level" is now a column (x position), nodes at same level stack vertically
  const nodeWidth = 220
  const nodeHeight = 80
  const horizontalGap = 180  // Gap between columns (levels)
  const verticalGap = 100    // Gap between nodes in same column

  levelGroups.forEach((nodesAtLevel, level) => {
    // Calculate vertical centering for nodes at this level
    const totalHeight =
      nodesAtLevel.length * nodeHeight + (nodesAtLevel.length - 1) * verticalGap
    const startY = -totalHeight / 2 + nodeHeight / 2

    nodesAtLevel.forEach((nodeName, index) => {
      positions.set(nodeName, {
        // X position based on level (column) - flows left to right
        x: level * (nodeWidth + horizontalGap) + 80,
        // Y position stacks nodes vertically within their column
        y: startY + index * (nodeHeight + verticalGap),
      })
    })
  })

  return positions
}

export const useStudioStore = create<StudioState>((set, get) => ({
  // Files
  files: [],
  currentFile: null,
  yamlContent: '',

  // Canvas
  nodes: [],
  edges: [],

  // Validation
  validation: null,

  // Execution state
  nodeExecutionStatus: new Map<string, NodeExecutionState>(),
  isExecuting: false,

  // UI state
  isLoading: false,
  isSaving: false,
  isDirty: false,

  // Actions
  setFiles: (files: FileInfo[]) => set({ files }),
  setCurrentFile: (path: string | null) => set({ currentFile: path }),
  setYamlContent: (content: string) => set({ yamlContent: content, isDirty: true }),
  setNodes: (nodes: HexdagNode[]) => set({ nodes }),
  setEdges: (edges: HexdagEdge[]) => set({ edges }),
  setValidation: (result: ValidationResult | null) => set({ validation: result }),
  setIsLoading: (loading: boolean) => set({ isLoading: loading }),
  setIsSaving: (saving: boolean) => set({ isSaving: saving }),
  setIsDirty: (dirty: boolean) => set({ isDirty: dirty }),

  // Execution actions
  setIsExecuting: (executing: boolean) => set({ isExecuting: executing }),

  setNodeExecutionStatus: (nodeName: string, state: NodeExecutionState) =>
    set((prev) => {
      const newMap = new Map(prev.nodeExecutionStatus)
      newMap.set(nodeName, state)
      return { nodeExecutionStatus: newMap }
    }),

  resetExecutionStatus: () =>
    set({ nodeExecutionStatus: new Map(), isExecuting: false }),

  initializeExecutionStatus: (nodeNames: string[]) =>
    set({
      nodeExecutionStatus: new Map(
        nodeNames.map((name) => [name, { status: 'pending' as const }])
      ),
    }),

  // Complex actions
  syncYamlToCanvas: () => {
    const { yamlContent } = get()

    if (!yamlContent.trim()) {
      set({ nodes: [], edges: [] })
      return
    }

    try {
      const parsed = yaml.parse(yamlContent) as Pipeline | null

      if (!parsed || !parsed.spec || !parsed.spec.nodes) {
        set({ nodes: [], edges: [] })
        return
      }

      const pipelineNodes = parsed.spec.nodes
      const positions = calculateNodePositions(pipelineNodes)

      // Convert pipeline nodes to canvas nodes
      // Always use calculated positions for clean layout on file open
      const canvasNodes: HexdagNode[] = pipelineNodes.map((node) => {
        const calculatedPos = positions.get(node.metadata.name) || { x: 0, y: 0 }

        // Process spec for function_node: convert body to code for editing
        let spec = node.spec || {}
        if (node.kind === 'function_node' && spec.body && typeof spec.body === 'string') {
          spec = { ...spec, code: spec.body }
          delete (spec as Record<string, unknown>).body
        }

        return {
          id: node.metadata.name,
          type: 'hexdagNode',
          position: calculatedPos,
          data: {
            kind: node.kind,
            label: node.metadata.name,
            spec,
            isValid: true,
            errors: [],
          },
        }
      })

      // Build edges from dependencies
      const canvasEdges: HexdagEdge[] = []
      pipelineNodes.forEach((node) => {
        ;(node.dependencies || []).forEach((dep) => {
          canvasEdges.push({
            id: `${dep}-${node.metadata.name}`,
            source: dep,
            target: node.metadata.name,
            type: 'smoothstep',
            animated: false,
          })
        })
      })

      set({ nodes: canvasNodes, edges: canvasEdges })
    } catch (error) {
      console.error('Failed to parse YAML:', error)
      // Keep existing nodes/edges on parse error
    }
  },

  syncCanvasToYaml: () => {
    const { nodes, edges, yamlContent } = get()

    if (nodes.length === 0) {
      // Don't clear YAML if there are no nodes - could be a parse error state
      return
    }

    try {
      // Parse existing YAML to preserve metadata and ports
      let existingPipeline: Pipeline | null = null
      try {
        existingPipeline = yaml.parse(yamlContent) as Pipeline | null
      } catch {
        // If YAML is invalid, start fresh
      }

      // Build dependency map from edges
      const dependencyMap = new Map<string, string[]>()
      edges.forEach((edge) => {
        const deps = dependencyMap.get(edge.target) || []
        deps.push(edge.source)
        dependencyMap.set(edge.target, deps)
      })

      // Convert canvas nodes to pipeline nodes
      const pipelineNodes: NodeSpec[] = nodes.map((node) => {
        const spec = { ...node.data.spec } as Record<string, unknown>

        // For function_node with inline code, convert code to body field
        // The backend will process this as executable Python code
        if (node.data.kind === 'function_node' && spec.code && typeof spec.code === 'string') {
          // Move code to 'body' which is the standard field for inline Python
          spec.body = spec.code
          delete spec.code
          // Remove inline_code if present (legacy field)
          delete spec.inline_code
        }

        return {
          kind: node.data.kind,
          metadata: {
            name: node.id,
          },
          spec,
          dependencies: dependencyMap.get(node.id) || [],
        }
      })

      // Build the pipeline object
      const pipeline: Pipeline = {
        apiVersion: existingPipeline?.apiVersion || 'hexdag/v1',
        kind: existingPipeline?.kind || 'Pipeline',
        metadata: existingPipeline?.metadata || {
          name: 'untitled-pipeline',
        },
        spec: {
          ports: existingPipeline?.spec?.ports,
          nodes: pipelineNodes,
        },
      }

      // Serialize to YAML
      const newYamlContent = yaml.stringify(pipeline, {
        indent: 2,
        lineWidth: 120,
      })

      set({ yamlContent: newYamlContent, isDirty: true })
    } catch (error) {
      console.error('Failed to serialize to YAML:', error)
    }
  },
}))
