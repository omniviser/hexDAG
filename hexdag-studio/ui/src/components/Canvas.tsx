import { useCallback, useRef, useEffect, useState } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  type Connection,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  type Node,
  type ReactFlowInstance,
  type NodeChange,
  BackgroundVariant,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import HexdagNode from './HexdagNode'
import ContextMenu, { createNodeContextMenuItems } from './ContextMenu'
import { useStudioStore } from '../lib/store'
import { getNodeTemplate, getNodeColor } from '../lib/nodeTemplates'
import { generateUniqueName } from '../lib/formatValue'
import type { HexdagNode as HexdagNodeType, HexdagNodeData } from '../types'

const nodeTypes = {
  hexdagNode: HexdagNode,
} as const

interface CanvasProps {
  onNodeSelect?: (nodeId: string | null) => void
  selectedNodeId?: string | null
}

export default function Canvas({ onNodeSelect, selectedNodeId }: CanvasProps) {
  const reactFlowWrapper = useRef<HTMLDivElement>(null)
  const reactFlowInstance = useRef<ReactFlowInstance<HexdagNodeType> | null>(null)

  // Context menu state
  const [contextMenu, setContextMenu] = useState<{
    x: number
    y: number
    nodeId: string
  } | null>(null)

  const {
    nodes: storeNodes,
    edges: storeEdges,
    setNodes: setStoreNodes,
    setEdges: setStoreEdges,
    syncCanvasToYaml,
  } = useStudioStore()

  const [nodes, setNodes, onNodesChange] = useNodesState<HexdagNodeType>(storeNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(storeEdges)

  // Sync from store when store changes externally (e.g., YAML edit)
  useEffect(() => {
    setNodes(storeNodes)
    setEdges(storeEdges)
  }, [storeNodes, storeEdges, setNodes, setEdges])

  // Update selected state when selectedNodeId changes
  useEffect(() => {
    setNodes((nds) =>
      nds.map((node) => ({
        ...node,
        selected: node.id === selectedNodeId,
      }))
    )
  }, [selectedNodeId, setNodes])

  const handleNodesChange: OnNodesChange<HexdagNodeType> = useCallback(
    (changes: NodeChange<HexdagNodeType>[]) => {
      onNodesChange(changes)

      // Handle selection changes
      const selectionChange = changes.find((c) => c.type === 'select')
      if (selectionChange && 'selected' in selectionChange) {
        if (selectionChange.selected) {
          onNodeSelect?.(selectionChange.id)
        }
      }

      // Handle deletions - use the nodes from changes directly, not from store
      const deletions = changes.filter((c) => c.type === 'remove')
      if (deletions.length > 0) {
        const deletedIds = deletions.map((d) => d.id)
        // Use setNodes callback to get current local state and update store synchronously
        setNodes((currentNodes) => {
          const newNodes = currentNodes.filter((n) => !deletedIds.includes(n.id))
          setStoreNodes(newNodes)
          // Sync to YAML after store is updated
          queueMicrotask(() => syncCanvasToYaml())
          return newNodes
        })

        // Deselect if deleted node was selected
        if (selectedNodeId && deletedIds.includes(selectedNodeId)) {
          onNodeSelect?.(null)
        }
        return
      }

      // Sync position changes when drag ends
      const hasPositionChange = changes.some(
        (c) => c.type === 'position' && 'dragging' in c && c.dragging === false
      )
      if (hasPositionChange) {
        // Use setNodes callback to get current local state synchronously
        setNodes((currentNodes) => {
          setStoreNodes(currentNodes)
          // Use queueMicrotask for YAML sync after store update
          queueMicrotask(() => syncCanvasToYaml())
          return currentNodes
        })
      }
    },
    [onNodesChange, nodes, setStoreNodes, syncCanvasToYaml, onNodeSelect, selectedNodeId]
  )

  const handleEdgesChange: OnEdgesChange = useCallback(
    (changes) => {
      onEdgesChange(changes)
      const hasNonSelectChange = changes.some((c) => c.type !== 'select')
      if (hasNonSelectChange) {
        setTimeout(() => {
          setStoreEdges(edges)
          syncCanvasToYaml()
        }, 100)
      }
    },
    [onEdgesChange, edges, setStoreEdges, syncCanvasToYaml]
  )

  const onConnect: OnConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) => {
        const newEdges = addEdge(
          { ...connection, type: 'smoothstep', animated: false },
          eds
        )
        setTimeout(() => {
          setStoreEdges(newEdges)
          syncCanvasToYaml()
        }, 100)
        return newEdges
      })
    },
    [setEdges, setStoreEdges, syncCanvasToYaml]
  )

  const onPaneClick = useCallback(() => {
    onNodeSelect?.(null)
  }, [onNodeSelect])

  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      onNodeSelect?.(node.id)
    },
    [onNodeSelect]
  )

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault()

      const kind = event.dataTransfer.getData('application/hexdag-node')
      if (!kind || !reactFlowInstance.current || !reactFlowWrapper.current) {
        return
      }

      const bounds = reactFlowWrapper.current.getBoundingClientRect()
      const position = reactFlowInstance.current.screenToFlowPosition({
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      })

      // Get template for builtin nodes, or create default spec for plugin nodes
      const template = getNodeTemplate(kind)
      const defaultSpec = template?.defaultSpec || {}

      // Generate unique name - handle namespaced kinds (e.g., etl:file_reader_node)
      const existingNames = nodes.map((n) => n.id)
      // Extract base name: "etl:file_reader_node" -> "file_reader", "function_node" -> "function"
      const kindParts = kind.split(':')
      const nodeKind = kindParts[kindParts.length - 1] // Get last part after ':'
      const baseName = nodeKind.replace('_node', '')
      const name = generateUniqueName(baseName, existingNames)

      const newNode: HexdagNodeType = {
        id: name,
        type: 'hexdagNode',
        position,
        data: {
          kind,
          label: name,
          spec: { ...defaultSpec },
          isValid: true,
          errors: [],
        },
      }

      // Update both local React state and store, then sync to YAML
      const newNodes = [...nodes, newNode]
      setNodes(newNodes)
      setStoreNodes(newNodes)

      // Use setTimeout to ensure store is updated, then call sync from store
      setTimeout(() => {
        useStudioStore.getState().syncCanvasToYaml()
      }, 50)

      // Select the new node
      onNodeSelect?.(name)
    },
    [nodes, setNodes, setStoreNodes, syncCanvasToYaml, onNodeSelect]
  )

  const onInit = useCallback((instance: ReactFlowInstance<HexdagNodeType>) => {
    reactFlowInstance.current = instance
  }, [])

  // Node context menu handlers
  const onNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: Node) => {
      event.preventDefault()
      setContextMenu({
        x: event.clientX,
        y: event.clientY,
        nodeId: node.id,
      })
    },
    []
  )

  const closeContextMenu = useCallback(() => {
    setContextMenu(null)
  }, [])

  const duplicateNode = useCallback(
    (nodeId: string) => {
      const node = nodes.find((n) => n.id === nodeId)
      if (!node) return

      const existingNames = nodes.map((n) => n.id)
      const newName = generateUniqueName(nodeId, existingNames, '_copy')

      const newNode: HexdagNodeType = {
        ...node,
        id: newName,
        position: {
          x: node.position.x + 50,
          y: node.position.y + 50,
        },
        data: {
          ...node.data,
          label: newName,
        } as HexdagNodeData,
      }

      const newNodes = [...nodes, newNode]
      setNodes(newNodes)
      setStoreNodes(newNodes)
      setTimeout(() => {
        useStudioStore.getState().syncCanvasToYaml()
      }, 50)
    },
    [nodes, setNodes, setStoreNodes]
  )

  const deleteNode = useCallback(
    (nodeId: string) => {
      const newNodes = nodes.filter((n) => n.id !== nodeId)
      const newEdges = edges.filter((e) => e.source !== nodeId && e.target !== nodeId)

      setNodes(newNodes)
      setEdges(newEdges)
      setStoreNodes(newNodes)
      setStoreEdges(newEdges)

      setTimeout(() => {
        useStudioStore.getState().syncCanvasToYaml()
      }, 50)

      if (selectedNodeId === nodeId) {
        onNodeSelect?.(null)
      }
    },
    [nodes, edges, setNodes, setEdges, setStoreNodes, setStoreEdges, selectedNodeId, onNodeSelect]
  )

  const disconnectNode = useCallback(
    (nodeId: string) => {
      const newEdges = edges.filter((e) => e.source !== nodeId && e.target !== nodeId)
      setEdges(newEdges)
      setStoreEdges(newEdges)
      setTimeout(() => {
        useStudioStore.getState().syncCanvasToYaml()
      }, 50)
    },
    [edges, setEdges, setStoreEdges]
  )

  // Get context menu items for the selected node
  const getContextMenuItems = useCallback(
    (nodeId: string) => {
      return createNodeContextMenuItems(nodeId, {
        onEdit: () => {
          onNodeSelect?.(nodeId)
        },
        onDuplicate: () => {
          duplicateNode(nodeId)
        },
        onDelete: () => {
          deleteNode(nodeId)
        },
        onDisconnect: () => {
          disconnectNode(nodeId)
        },
      })
    },
    [onNodeSelect, duplicateNode, deleteNode, disconnectNode]
  )

  return (
    <div ref={reactFlowWrapper} className="w-full h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={handleNodesChange}
        onEdgesChange={handleEdgesChange}
        onConnect={onConnect}
        onInit={onInit}
        onDragOver={onDragOver}
        onDrop={onDrop}
        onPaneClick={onPaneClick}
        onNodeClick={onNodeClick}
        onNodeContextMenu={onNodeContextMenu}
        nodeTypes={nodeTypes}
        fitView
        snapToGrid
        snapGrid={[20, 20]}
        deleteKeyCode={['Backspace', 'Delete']}
        multiSelectionKeyCode={['Shift']}
        defaultEdgeOptions={{
          type: 'smoothstep',
          animated: false,
          style: { stroke: '#6366f1', strokeWidth: 2 },
        }}
        proOptions={{ hideAttribution: true }}
      >
        <Background
          color="#2a2a3e"
          gap={20}
          variant={BackgroundVariant.Dots}
        />
        <Controls
          showZoom={true}
          showFitView={true}
          showInteractive={false}
        />
        <MiniMap
          nodeColor={(node) => {
            const data = node.data as HexdagNodeData | undefined
            return getNodeColor(data?.kind || '')
          }}
          maskColor="rgba(15, 15, 26, 0.8)"
          style={{
            backgroundColor: '#1a1a2e',
          }}
        />
      </ReactFlow>

      {/* Context Menu */}
      {contextMenu && (
        <ContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          items={getContextMenuItems(contextMenu.nodeId)}
          onClose={closeContextMenu}
        />
      )}
    </div>
  )
}
