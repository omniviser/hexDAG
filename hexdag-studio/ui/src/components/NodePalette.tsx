import { useState, useEffect } from 'react'
import {
  Code,
  Brain,
  Bot,
  Box,
  Boxes,
  GripVertical,
  Plug,
  ChevronDown,
  ChevronRight,
  Loader2,
  Package,
  Layers,
  Calculator,
  Database,
  DatabaseBackup,
  Wrench,
  FileInput,
  FileOutput,
  FolderOpen,
  Table,
  Mail,
  Send,
  Globe,
  HardDrive,
  KeyRound,
  type LucideIcon,
} from 'lucide-react'
import { getAllPluginNodes, getBuiltinNodes, type PluginNode, type BuiltinNodeType } from '../lib/api'
import { getPluginNodeIcon } from '../lib/nodeTemplates'

// Icon mapping for node icon names (from API)
const iconMap: Record<string, LucideIcon> = {
  Code,
  Brain,
  Bot,
  Box,
  Boxes,
  Plug,
  Package,
  Layers,
  Calculator,
  Database,
  DatabaseBackup,
  Wrench,
  FileInput,
  FileOutput,
  FolderOpen,
  Table,
  Mail,
  Send,
  Globe,
  HardDrive,
  KeyRound,
}

interface PluginNodeSection {
  plugin: string
  nodes: PluginNode[]
  expanded: boolean
}

export default function NodePalette() {
  const [builtinNodes, setBuiltinNodes] = useState<BuiltinNodeType[]>([])
  const [pluginNodeSections, setPluginNodeSections] = useState<PluginNodeSection[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [builtinExpanded, setBuiltinExpanded] = useState(true)

  useEffect(() => {
    loadNodes()
  }, [])

  const loadNodes = async () => {
    try {
      setIsLoading(true)

      // Load built-in nodes from API
      const builtin = await getBuiltinNodes(false) // Don't include deprecated
      setBuiltinNodes(builtin)

      // Load plugin nodes
      const pluginNodes = await getAllPluginNodes()

      // Group plugin nodes by plugin
      const groupedNodes = pluginNodes.reduce(
        (acc: Record<string, PluginNode[]>, node: PluginNode) => {
          const plugin = node.plugin || 'unknown'
          if (!acc[plugin]) {
            acc[plugin] = []
          }
          acc[plugin].push(node)
          return acc
        },
        {} as Record<string, PluginNode[]>
      )

      setPluginNodeSections(
        Object.entries(groupedNodes).map(([plugin, nodes]: [string, PluginNode[]]) => ({
          plugin,
          nodes,
          expanded: true,
        }))
      )
    } catch (error) {
      console.error('Failed to load nodes:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const toggleNodeSection = (plugin: string) => {
    setPluginNodeSections((sections) =>
      sections.map((s) => (s.plugin === plugin ? { ...s, expanded: !s.expanded } : s))
    )
  }

  const onDragStart = (event: React.DragEvent, kind: string) => {
    event.dataTransfer.setData('application/hexdag-node', kind)
    event.dataTransfer.effectAllowed = 'move'
  }

  return (
    <div className="h-full flex flex-col">
      <div className="p-3 border-b border-hex-border">
        <h2 className="text-xs font-semibold uppercase text-hex-text-muted tracking-wider">
          Node Palette
        </h2>
      </div>
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="flex items-center justify-center p-4">
            <Loader2 size={16} className="animate-spin text-hex-text-muted" />
            <span className="ml-2 text-xs text-hex-text-muted">Loading nodes...</span>
          </div>
        ) : (
          <>
            {/* Built-in Nodes Section */}
            <div className="border-b border-hex-border">
              <button
                onClick={() => setBuiltinExpanded(!builtinExpanded)}
                className="w-full flex items-center gap-2 p-2 hover:bg-hex-border/30 transition-colors"
              >
                {builtinExpanded ? (
                  <ChevronDown size={14} className="text-hex-text-muted" />
                ) : (
                  <ChevronRight size={14} className="text-hex-text-muted" />
                )}
                <Box size={14} className="text-hex-accent" />
                <span className="text-xs font-medium text-hex-text">Core Nodes</span>
                <span className="ml-auto text-[10px] text-hex-text-muted">{builtinNodes.length}</span>
              </button>
              {builtinExpanded && (
                <div className="px-2 pb-2 space-y-1">
                  {builtinNodes.map((node) => {
                    const Icon = iconMap[node.icon] || Box
                    return (
                      <div
                        key={node.kind}
                        draggable
                        onDragStart={(e) => onDragStart(e, node.kind)}
                        className="
                          flex items-center gap-2 p-2 rounded-md cursor-grab
                          bg-hex-bg hover:bg-hex-border/50 transition-colors
                          border border-transparent hover:border-hex-border
                          group
                        "
                      >
                        <div className="text-hex-text-muted opacity-0 group-hover:opacity-100 transition-opacity">
                          <GripVertical size={12} />
                        </div>
                        <div
                          className="w-7 h-7 rounded flex items-center justify-center flex-shrink-0"
                          style={{ backgroundColor: `${node.color}20` }}
                        >
                          <Icon size={14} style={{ color: node.color }} />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="text-xs font-medium text-hex-text truncate">
                            {node.name}
                          </div>
                          <div className="text-[10px] text-hex-text-muted truncate">
                            {node.description}
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>

            {/* Plugin Nodes Sections */}
            {pluginNodeSections.map((section) => (
              <div key={section.plugin} className="border-b border-hex-border">
                <button
                  onClick={() => toggleNodeSection(section.plugin)}
                  className="w-full flex items-center gap-2 p-2 hover:bg-hex-border/30 transition-colors"
                >
                  {section.expanded ? (
                    <ChevronDown size={14} className="text-hex-text-muted" />
                  ) : (
                    <ChevronRight size={14} className="text-hex-text-muted" />
                  )}
                  <Plug size={14} className="text-green-500" />
                  <span className="text-xs font-medium text-hex-text truncate">
                    {section.plugin}
                  </span>
                  <span className="ml-auto text-[10px] text-hex-text-muted">
                    {section.nodes.length}
                  </span>
                </button>
                {section.expanded && (
                  <div className="px-2 pb-2 space-y-1">
                    {section.nodes.map((node) => {
                      // Use icon from API if available, otherwise fall back to heuristics
                      const Icon = node.icon && iconMap[node.icon] ? iconMap[node.icon] : getPluginNodeIcon(node.kind)
                      return (
                        <div
                          key={`${section.plugin}:${node.kind}`}
                          draggable
                          onDragStart={(e) => onDragStart(e, node.kind)}
                          className="
                            flex items-center gap-2 p-2 rounded-md cursor-grab
                            bg-hex-bg hover:bg-hex-border/50 transition-colors
                            border border-transparent hover:border-hex-border
                            group
                          "
                        >
                          <div className="text-hex-text-muted opacity-0 group-hover:opacity-100 transition-opacity">
                            <GripVertical size={12} />
                          </div>
                          <div
                            className="w-7 h-7 rounded flex items-center justify-center flex-shrink-0"
                            style={{ backgroundColor: `${node.color}20` }}
                          >
                            <Icon size={14} style={{ color: node.color }} />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="text-xs font-medium text-hex-text truncate">
                              {node.name}
                            </div>
                            <div className="text-[10px] text-hex-text-muted truncate">
                              {node.description || `From ${section.plugin}`}
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            ))}

            {pluginNodeSections.length === 0 && (
              <div className="p-4 text-center">
                <Plug size={24} className="mx-auto mb-2 text-hex-text-muted opacity-50" />
                <p className="text-xs text-hex-text-muted">No plugin nodes found</p>
                <p className="text-[10px] text-hex-text-muted mt-1">
                  Add node plugins to hexdag_plugins/
                </p>
              </div>
            )}
          </>
        )}
      </div>
      <div className="p-3 border-t border-hex-border">
        <p className="text-[10px] text-hex-text-muted text-center">
          Drag nodes onto the canvas
        </p>
      </div>
    </div>
  )
}
