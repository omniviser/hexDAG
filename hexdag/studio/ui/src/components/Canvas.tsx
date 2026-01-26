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
  BackgroundVariant,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import HexdagNode from './HexdagNode'
import ContextMenu, { createNodeContextMenuItems } from './ContextMenu'
import { useStudioStore } from '../lib/store'
import { getNodeTemplate, getNodeColor } from '../lib/nodeTemplates'
import type { HexdagNode as HexdagNodeType } from '../types'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const nodeTypes: Record<string, any> = {
  hexdagNode: HexdagNode,
}

interface CanvasProps {
  onNodeSelect?: (nodeId: string | null) => void
  selectedNodeId?: string | null
}

export default function Canvas({ onNodeSelect, selectedNodeId }: CanvasProps) {
  const reactFlowWrapper = useRef<HTMLDivElement>(null)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const reactFlowInstance = useRef<any>(null)

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

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [nodes, setNodes, onNodesChange] = useNodesState(storeNodes as any)
  const [edges, setEdges, onEdgesChange] = useEdgesState(storeEdges)

  // Sync from store when store changes externally (e.g., YAML edit)
  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    setNodes(storeNodes as any)
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

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleNodesChange: OnNodesChange<any> = useCallback(
    (changes) => {
      onNodesChange(changes)

      // Handle selection changes
      const selectionChange = changes.find((c) => c.type === 'select')
      if (selectionChange && 'selected' in selectionChange) {
        if (selectionChange.selected) {
          onNodeSelect?.(selectionChange.id)
        }
      }

      // Handle deletions
      const deletions = changes.filter((c) => c.type === 'remove')
      if (deletions.length > 0) {
        setTimeout(() => {
          const currentNodes = useStudioStore.getState().nodes
          const deletedIds = deletions.map((d) => d.id)
          const newNodes = currentNodes.filter((n) => !deletedIds.includes(n.id))
          setStoreNodes(newNodes)
          syncCanvasToYaml()

          // Deselect if deleted node was selected
          if (selectedNodeId && deletedIds.includes(selectedNodeId)) {
            onNodeSelect?.(null)
          }
        }, 100)
        return
      }

      // Debounce sync for position changes
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const hasPositionChange = changes.some(
        (c: any) => c.type === 'position' && c.dragging === false
      )
      if (hasPositionChange) {
        setTimeout(() => {
          // Get current nodes from React state via callback
          setNodes((currentNodes) => {
            setStoreNodes(currentNodes as HexdagNodeType[])
            setTimeout(() => useStudioStore.getState().syncCanvasToYaml(), 10)
            return currentNodes
          })
        }, 100)
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
      let baseName = nodeKind.replace('_node', '')
      let name = baseName
      let counter = 1
      while (existingNames.includes(name)) {
        name = `${baseName}_${counter}`
        counter++
      }

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
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const newNodes = [...nodes, newNode] as any
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

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const onInit = useCallback((instance: any) => {
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

      let newName = `${nodeId}_copy`
      let counter = 1
      while (nodes.some((n) => n.id === newName)) {
        newName = `${nodeId}_copy_${counter}`
        counter++
      }

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
        },
      } as HexdagNodeType

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const newNodes = [...nodes, newNode] as any
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

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      setNodes(newNodes as any)
      setEdges(newEdges)
      setStoreNodes(newNodes as HexdagNodeType[])
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
          nodeColor={(node: Node) => {
            const data = node.data as { kind?: string }
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
