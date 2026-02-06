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
} from '../types'

// Helper to calculate node positions in a DAG layout
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

  // Calculate positions
  const nodeWidth = 200
  const nodeHeight = 80
  const horizontalGap = 100
  const verticalGap = 50

  levelGroups.forEach((nodesAtLevel, level) => {
    const totalWidth =
      nodesAtLevel.length * nodeWidth + (nodesAtLevel.length - 1) * horizontalGap
    const startX = -totalWidth / 2

    nodesAtLevel.forEach((nodeName, index) => {
      positions.set(nodeName, {
        x: startX + index * (nodeWidth + horizontalGap) + 100,
        y: level * (nodeHeight + verticalGap) + 50,
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
      const existingNodes = get().nodes
      const existingPositions = new Map(
        existingNodes.map((n) => [n.id, n.position])
      )

      // Convert pipeline nodes to canvas nodes
      const canvasNodes: HexdagNode[] = pipelineNodes.map((node) => {
        const existingPos = existingPositions.get(node.metadata.name)
        const calculatedPos = positions.get(node.metadata.name) || { x: 0, y: 0 }

        return {
          id: node.metadata.name,
          type: 'hexdagNode',
          position: existingPos || calculatedPos,
          data: {
            kind: node.kind,
            label: node.metadata.name,
            spec: node.spec || {},
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
      const pipelineNodes: NodeSpec[] = nodes.map((node) => ({
        kind: node.data.kind,
        metadata: {
          name: node.id,
        },
        spec: node.data.spec || {},
        dependencies: dependencyMap.get(node.id) || [],
      }))

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
