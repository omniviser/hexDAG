import { useState, useEffect } from 'react'
import {
  Plug,
  Package,
  Loader2,
  CheckCircle2,
  XCircle,
  ChevronDown,
  ChevronRight,
  Key,
  Database,
  HardDrive,
  Brain,
  RefreshCw,
  X,
} from 'lucide-react'
import { listPlugins, type PluginInfo, type PluginAdapter, type PluginNode } from '../lib/api'

interface PluginManagerProps {
  isOpen: boolean
  onClose: () => void
}

export default function PluginManager({ isOpen, onClose }: PluginManagerProps) {
  const [plugins, setPlugins] = useState<PluginInfo[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [expandedPlugins, setExpandedPlugins] = useState<Set<string>>(new Set())
  const [activeTab, setActiveTab] = useState<'all' | 'adapters' | 'nodes'>('all')

  useEffect(() => {
    if (isOpen) {
      loadPlugins()
    }
  }, [isOpen])

  const loadPlugins = async () => {
    try {
      setIsLoading(true)
      const data = await listPlugins()
      setPlugins(data)
      // Expand first plugin by default
      if (data.length > 0) {
        setExpandedPlugins(new Set([data[0].name]))
      }
    } catch (error) {
      console.error('Failed to load plugins:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const togglePlugin = (name: string) => {
    setExpandedPlugins((prev) => {
      const next = new Set(prev)
      if (next.has(name)) {
        next.delete(name)
      } else {
        next.add(name)
      }
      return next
    })
  }

  const getAdapterIcon = (adapter: PluginAdapter) => {
    const portType = adapter.port_type.toLowerCase()
    if (portType === 'llm') return Brain
    if (portType === 'secret') return Key
    if (portType === 'memory' || portType === 'database') return Database
    if (portType === 'storage') return HardDrive
    return Package
  }

  const getPortTypeColor = (portType: string) => {
    const type = portType.toLowerCase()
    if (type === 'llm') return '#6366f1'
    if (type === 'secret') return '#f59e0b'
    if (type === 'memory') return '#14b8a6'
    if (type === 'database') return '#ec4899'
    if (type === 'storage') return '#22c55e'
    return '#6b7280'
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-hex-surface border border-hex-border rounded-lg shadow-2xl w-[800px] max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-hex-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-hex-accent/20 flex items-center justify-center">
              <Plug size={20} className="text-hex-accent" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-hex-text">Plugin Manager</h2>
              <p className="text-xs text-hex-text-muted">
                Manage hexdag plugins and their components
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={loadPlugins}
              disabled={isLoading}
              className="p-2 rounded hover:bg-hex-border/50 transition-colors text-hex-text-muted hover:text-hex-text"
            >
              <RefreshCw size={16} className={isLoading ? 'animate-spin' : ''} />
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded hover:bg-hex-border/50 transition-colors text-hex-text-muted hover:text-hex-text"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-hex-border px-4">
          {(['all', 'adapters', 'nodes'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 text-xs font-medium border-b-2 transition-colors ${
                activeTab === tab
                  ? 'text-hex-accent border-hex-accent'
                  : 'text-hex-text-muted border-transparent hover:text-hex-text'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {isLoading ? (
            <div className="flex items-center justify-center h-40">
              <Loader2 size={24} className="animate-spin text-hex-accent" />
              <span className="ml-2 text-hex-text-muted">Loading plugins...</span>
            </div>
          ) : plugins.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-40 text-center">
              <Package size={48} className="text-hex-text-muted opacity-30 mb-4" />
              <p className="text-hex-text-muted">No plugins found</p>
              <p className="text-xs text-hex-text-muted mt-1">
                Add plugins to the hexdag_plugins/ directory
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {plugins.map((plugin) => (
                <PluginCard
                  key={plugin.name}
                  plugin={plugin}
                  isExpanded={expandedPlugins.has(plugin.name)}
                  onToggle={() => togglePlugin(plugin.name)}
                  activeTab={activeTab}
                  getAdapterIcon={getAdapterIcon}
                  getPortTypeColor={getPortTypeColor}
                />
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-hex-border bg-hex-bg/50">
          <p className="text-[10px] text-hex-text-muted text-center">
            Plugins are automatically discovered from hexdag_plugins/ and installed packages
          </p>
        </div>
      </div>
    </div>
  )
}

interface PluginCardProps {
  plugin: PluginInfo
  isExpanded: boolean
  onToggle: () => void
  activeTab: 'all' | 'adapters' | 'nodes'
  getAdapterIcon: (adapter: PluginAdapter) => typeof Package
  getPortTypeColor: (portType: string) => string
}

function PluginCard({
  plugin,
  isExpanded,
  onToggle,
  activeTab,
  getAdapterIcon,
  getPortTypeColor,
}: PluginCardProps) {
  const showAdapters = activeTab === 'all' || activeTab === 'adapters'
  const showNodes = activeTab === 'all' || activeTab === 'nodes'

  return (
    <div className="border border-hex-border rounded-lg overflow-hidden">
      {/* Plugin Header */}
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 p-3 hover:bg-hex-border/30 transition-colors"
      >
        {isExpanded ? (
          <ChevronDown size={14} className="text-hex-text-muted" />
        ) : (
          <ChevronRight size={14} className="text-hex-text-muted" />
        )}
        <div className="w-8 h-8 rounded bg-green-500/20 flex items-center justify-center">
          <Plug size={16} className="text-green-500" />
        </div>
        <div className="flex-1 text-left">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-hex-text">{plugin.name}</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-hex-border text-hex-text-muted">
              v{plugin.version}
            </span>
          </div>
          <p className="text-xs text-hex-text-muted truncate">{plugin.description}</p>
        </div>
        <div className="flex items-center gap-4 text-hex-text-muted">
          <div className="flex items-center gap-1 text-xs">
            <Key size={12} />
            <span>{plugin.adapters.length}</span>
          </div>
          <div className="flex items-center gap-1 text-xs">
            <Package size={12} />
            <span>{plugin.nodes.length}</span>
          </div>
          {plugin.enabled ? (
            <CheckCircle2 size={16} className="text-green-500" />
          ) : (
            <XCircle size={16} className="text-red-500" />
          )}
        </div>
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="border-t border-hex-border bg-hex-bg/50 p-3 space-y-4">
          {/* Adapters */}
          {showAdapters && plugin.adapters.length > 0 && (
            <div>
              <h4 className="text-xs font-medium text-hex-text-muted mb-2 flex items-center gap-2">
                <Key size={12} />
                Adapters ({plugin.adapters.length})
              </h4>
              <div className="grid grid-cols-2 gap-2">
                {plugin.adapters.map((adapter, idx) => {
                  const Icon = getAdapterIcon(adapter)
                  const color = getPortTypeColor(adapter.port_type)
                  return (
                    <div
                      key={idx}
                      className="flex items-start gap-2 p-2 rounded bg-hex-surface border border-hex-border"
                    >
                      <div
                        className="w-7 h-7 rounded flex items-center justify-center flex-shrink-0"
                        style={{ backgroundColor: `${color}20` }}
                      >
                        <Icon size={14} style={{ color }} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-medium text-hex-text truncate">
                          {adapter.name}
                        </div>
                        <div className="text-[10px] text-hex-text-muted truncate">
                          {adapter.description}
                        </div>
                        <div className="flex items-center gap-2 mt-1">
                          <span
                            className="text-[9px] px-1 py-0.5 rounded"
                            style={{
                              backgroundColor: `${color}20`,
                              color: color,
                            }}
                          >
                            {adapter.port_type}
                          </span>
                          {adapter.secrets.length > 0 && (
                            <span className="text-[9px] px-1 py-0.5 rounded bg-amber-500/20 text-amber-500">
                              {adapter.secrets.length} secrets
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Nodes */}
          {showNodes && plugin.nodes.length > 0 && (
            <div>
              <h4 className="text-xs font-medium text-hex-text-muted mb-2 flex items-center gap-2">
                <Package size={12} />
                Nodes ({plugin.nodes.length})
              </h4>
              <div className="grid grid-cols-2 gap-2">
                {plugin.nodes.map((node, idx) => (
                  <NodeCard key={idx} node={node} />
                ))}
              </div>
            </div>
          )}

          {/* No content message */}
          {((activeTab === 'adapters' && plugin.adapters.length === 0) ||
            (activeTab === 'nodes' && plugin.nodes.length === 0)) && (
            <p className="text-xs text-hex-text-muted text-center py-4">
              No {activeTab} in this plugin
            </p>
          )}
        </div>
      )}
    </div>
  )
}

function NodeCard({ node }: { node: PluginNode }) {
  return (
    <div className="flex items-start gap-2 p-2 rounded bg-hex-surface border border-hex-border">
      <div
        className="w-7 h-7 rounded flex items-center justify-center flex-shrink-0"
        style={{ backgroundColor: `${node.color}20` }}
      >
        <Package size={14} style={{ color: node.color }} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-xs font-medium text-hex-text truncate">{node.name}</div>
        <div className="text-[10px] text-hex-text-muted truncate">{node.description}</div>
        <span
          className="inline-block text-[9px] px-1 py-0.5 rounded mt-1"
          style={{
            backgroundColor: `${node.color}20`,
            color: node.color,
          }}
        >
          {node.kind}
        </span>
      </div>
    </div>
  )
}
