import { useState, useEffect } from 'react'
import {
  Brain,
  Key,
  Database,
  HardDrive,
  Plug,
  ChevronDown,
  ChevronRight,
  Plus,
  X,
  Loader2,
  Info,
} from 'lucide-react'
import { useStudioStore } from '../lib/store'
import { getAllPluginAdapters, type PluginAdapter } from '../lib/api'
import yaml from 'yaml'

interface PortConfig {
  adapter: string
  config: Record<string, unknown>
}

interface PortsConfig {
  llm?: PortConfig
  memory?: PortConfig
  database?: PortConfig
  storage?: PortConfig
  secret?: PortConfig
  [key: string]: PortConfig | undefined
}

const PORT_TYPES = ['llm', 'memory', 'database', 'storage', 'secret'] as const

const PORT_INFO: Record<string, { icon: typeof Brain; color: string; description: string }> = {
  llm: {
    icon: Brain,
    color: '#6366f1',
    description: 'Language model for AI inference (OpenAI, Azure OpenAI, Anthropic)',
  },
  memory: {
    icon: Database,
    color: '#14b8a6',
    description: 'Persistent memory for agent state and conversations',
  },
  database: {
    icon: Database,
    color: '#ec4899',
    description: 'Database connection for data persistence',
  },
  storage: {
    icon: HardDrive,
    color: '#22c55e',
    description: 'File storage for documents and artifacts',
  },
  secret: {
    icon: Key,
    color: '#f59e0b',
    description: 'Secret management for API keys and credentials',
  },
}

export default function PortsEditor() {
  const { yamlContent, setYamlContent, setIsDirty } = useStudioStore()
  const [ports, setPorts] = useState<PortsConfig>({})
  const [adapters, setAdapters] = useState<PluginAdapter[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [expandedPorts, setExpandedPorts] = useState<Set<string>>(new Set(['llm']))

  useEffect(() => {
    loadAdapters()
  }, [])

  useEffect(() => {
    parsePortsFromYaml()
  }, [yamlContent])

  const loadAdapters = async () => {
    try {
      setIsLoading(true)
      const data = await getAllPluginAdapters()
      setAdapters(data)
    } catch (error) {
      console.error('Failed to load adapters:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const parsePortsFromYaml = () => {
    try {
      if (!yamlContent) {
        setPorts({})
        return
      }
      const parsed = yaml.parse(yamlContent) as { spec?: { ports?: PortsConfig } }
      setPorts(parsed?.spec?.ports || {})
    } catch {
      setPorts({})
    }
  }

  const updatePortsInYaml = (newPorts: PortsConfig) => {
    try {
      // Parse existing YAML or create default structure
      let parsed: Record<string, unknown> | null = null
      try {
        parsed = yamlContent ? yaml.parse(yamlContent) as Record<string, unknown> : null
      } catch {
        parsed = null
      }

      // Create default pipeline structure if needed
      if (!parsed || typeof parsed !== 'object') {
        parsed = {
          apiVersion: 'hexdag/v1',
          kind: 'Pipeline',
          metadata: { name: 'untitled' },
          spec: { nodes: [] },
        }
      }

      // Ensure spec exists
      if (!parsed.spec || typeof parsed.spec !== 'object') {
        parsed.spec = { nodes: [] }
      }

      const spec = parsed.spec as { ports?: PortsConfig; nodes?: unknown[] }

      // Ensure nodes array exists
      if (!spec.nodes) {
        spec.nodes = []
      }

      // Remove empty ports
      const cleanedPorts: PortsConfig = {}
      for (const [key, value] of Object.entries(newPorts)) {
        if (value && value.adapter) {
          cleanedPorts[key] = value
        }
      }

      if (Object.keys(cleanedPorts).length > 0) {
        spec.ports = cleanedPorts
      } else {
        delete spec.ports
      }

      const newYaml = yaml.stringify(parsed, {
        indent: 2,
        lineWidth: 0,
      })

      setYamlContent(newYaml)
      setIsDirty(true)
      setPorts(cleanedPorts)
    } catch (error) {
      console.error('Failed to update YAML:', error)
    }
  }

  const togglePort = (portType: string) => {
    setExpandedPorts((prev) => {
      const next = new Set(prev)
      if (next.has(portType)) {
        next.delete(portType)
      } else {
        next.add(portType)
      }
      return next
    })
  }

  const setPortAdapter = (portType: string, adapterName: string) => {
    const adapter = adapters.find((a) => a.name === adapterName)
    const newPorts = { ...ports }

    if (!adapterName) {
      delete newPorts[portType]
    } else {
      newPorts[portType] = {
        adapter: adapterName,
        config: adapter ? getDefaultConfig(adapter) : {},
      }
    }

    updatePortsInYaml(newPorts)
  }

  const updatePortConfig = (portType: string, key: string, value: unknown) => {
    const newPorts = { ...ports }
    if (newPorts[portType]) {
      newPorts[portType] = {
        ...newPorts[portType]!,
        config: {
          ...newPorts[portType]!.config,
          [key]: value,
        },
      }
      updatePortsInYaml(newPorts)
    }
  }

  const deletePortConfig = (portType: string, key: string) => {
    const newPorts = { ...ports }
    if (newPorts[portType]) {
      const newConfig = { ...newPorts[portType]!.config }
      delete newConfig[key]
      newPorts[portType] = {
        ...newPorts[portType]!,
        config: newConfig,
      }
      updatePortsInYaml(newPorts)
    }
  }

  const getDefaultConfig = (adapter: PluginAdapter): Record<string, unknown> => {
    const config: Record<string, unknown> = {}
    const schema = adapter.config_schema as {
      properties?: Record<string, { default?: unknown }>
      required?: string[]
    }

    if (schema?.properties) {
      for (const [key, prop] of Object.entries(schema.properties)) {
        if (prop.default !== undefined && prop.default !== null) {
          config[key] = prop.default
        } else if (schema.required?.includes(key)) {
          // Add placeholder for required fields
          config[key] = `\${${key.toUpperCase()}}`
        }
      }
    }

    return config
  }

  const getAdaptersForPort = (portType: string) => {
    return adapters.filter((a) => a.port_type === portType)
  }

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 size={24} className="animate-spin text-hex-text-muted" />
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-hex-surface">
      {/* Header */}
      <div className="p-3 border-b border-hex-border">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-hex-accent/20 flex items-center justify-center">
            <Plug size={16} className="text-hex-accent" />
          </div>
          <div>
            <h3 className="text-sm font-medium text-hex-text">Pipeline Ports</h3>
            <p className="text-[10px] text-hex-text-muted">Configure adapters for your pipeline</p>
          </div>
        </div>
      </div>

      {/* Port List */}
      <div className="flex-1 overflow-y-auto">
        {PORT_TYPES.map((portType) => {
          const portInfo = PORT_INFO[portType]
          const Icon = portInfo.icon
          const availableAdapters = getAdaptersForPort(portType)
          const currentPort = ports[portType]
          const isExpanded = expandedPorts.has(portType)
          const isConfigured = !!currentPort?.adapter

          return (
            <div key={portType} className="border-b border-hex-border">
              <button
                onClick={() => togglePort(portType)}
                className="w-full flex items-center gap-2 p-3 hover:bg-hex-border/30 transition-colors"
              >
                {isExpanded ? (
                  <ChevronDown size={14} className="text-hex-text-muted" />
                ) : (
                  <ChevronRight size={14} className="text-hex-text-muted" />
                )}
                <div
                  className="w-6 h-6 rounded flex items-center justify-center"
                  style={{ backgroundColor: `${portInfo.color}20` }}
                >
                  <Icon size={12} style={{ color: portInfo.color }} />
                </div>
                <span className="flex-1 text-left text-xs font-medium text-hex-text capitalize">
                  {portType}
                </span>
                {isConfigured ? (
                  <span
                    className="text-[10px] px-1.5 py-0.5 rounded"
                    style={{ backgroundColor: `${portInfo.color}20`, color: portInfo.color }}
                  >
                    {currentPort.adapter}
                  </span>
                ) : (
                  <span className="text-[10px] text-hex-text-muted">Not configured</span>
                )}
              </button>

              {isExpanded && (
                <div className="px-3 pb-3 space-y-3">
                  {/* Description */}
                  <div className="flex items-start gap-2 p-2 bg-hex-bg/50 rounded text-[10px] text-hex-text-muted">
                    <Info size={12} className="flex-shrink-0 mt-0.5" />
                    <span>{portInfo.description}</span>
                  </div>

                  {/* Adapter Selection */}
                  <div>
                    <label className="block text-[10px] font-medium text-hex-text-muted mb-1 uppercase tracking-wider">
                      Adapter
                    </label>
                    {availableAdapters.length === 0 ? (
                      <p className="text-[10px] text-hex-text-muted italic">
                        No adapters available for this port type
                      </p>
                    ) : (
                      <select
                        value={currentPort?.adapter || ''}
                        onChange={(e) => setPortAdapter(portType, e.target.value)}
                        className="w-full bg-hex-bg border border-hex-border rounded px-2 py-1.5 text-xs text-hex-text focus:border-hex-accent focus:outline-none"
                      >
                        <option value="">-- Select adapter --</option>
                        {availableAdapters.map((adapter) => (
                          <option key={adapter.name} value={adapter.name}>
                            {adapter.name} ({adapter.plugin})
                          </option>
                        ))}
                      </select>
                    )}
                  </div>

                  {/* Adapter Config */}
                  {currentPort && currentPort.config && (
                    <div className="space-y-2">
                      <label className="block text-[10px] font-medium text-hex-text-muted uppercase tracking-wider">
                        Configuration
                      </label>
                      {Object.entries(currentPort.config).map(([key, value]) => (
                        <ConfigField
                          key={key}
                          name={key}
                          value={value}
                          onChange={(v) => updatePortConfig(portType, key, v)}
                          onDelete={() => deletePortConfig(portType, key)}
                        />
                      ))}
                      <AddConfigButton
                        onAdd={(key) => updatePortConfig(portType, key, '')}
                        existingKeys={Object.keys(currentPort.config)}
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-hex-border bg-hex-bg/50">
        <p className="text-[9px] text-hex-text-muted text-center">
          Ports connect your pipeline to external services like LLMs and databases
        </p>
      </div>
    </div>
  )
}

function ConfigField({
  name,
  value,
  onChange,
  onDelete,
}: {
  name: string
  value: unknown
  onChange: (value: unknown) => void
  onDelete: () => void
}) {
  const isSecret = name.toLowerCase().includes('key') ||
    name.toLowerCase().includes('secret') ||
    name.toLowerCase().includes('password') ||
    name.toLowerCase().includes('token')

  const isEnvVar = typeof value === 'string' && value.startsWith('${') && value.endsWith('}')

  return (
    <div className="group flex items-start gap-2">
      <div className="flex-1">
        <div className="flex items-center justify-between mb-1">
          <label className="text-[10px] text-hex-text-muted">{name}</label>
          <button
            onClick={onDelete}
            className="opacity-0 group-hover:opacity-100 p-0.5 hover:bg-hex-error/20 rounded transition-all"
          >
            <X size={10} className="text-hex-error" />
          </button>
        </div>
        <div className="relative">
          <input
            type={isSecret && !isEnvVar ? 'password' : 'text'}
            value={String(value)}
            onChange={(e) => onChange(e.target.value)}
            className="w-full bg-hex-bg border border-hex-border rounded px-2 py-1 text-[11px] text-hex-text focus:border-hex-accent focus:outline-none font-mono"
            placeholder={isSecret ? '${ENV_VAR_NAME}' : ''}
          />
          {isEnvVar && (
            <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[9px] px-1 py-0.5 rounded bg-amber-500/20 text-amber-500">
              env
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

function AddConfigButton({
  onAdd,
  existingKeys,
}: {
  onAdd: (key: string) => void
  existingKeys: string[]
}) {
  const [isAdding, setIsAdding] = useState(false)
  const [newKey, setNewKey] = useState('')

  const handleAdd = () => {
    if (!newKey || existingKeys.includes(newKey)) return
    onAdd(newKey)
    setNewKey('')
    setIsAdding(false)
  }

  if (isAdding) {
    return (
      <div className="flex gap-1">
        <input
          type="text"
          value={newKey}
          onChange={(e) => setNewKey(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
          placeholder="Config key"
          autoFocus
          className="flex-1 bg-hex-bg border border-hex-border rounded px-2 py-1 text-[11px] text-hex-text focus:border-hex-accent focus:outline-none"
        />
        <button
          onClick={handleAdd}
          className="px-2 py-1 text-[10px] bg-hex-accent text-white rounded hover:bg-hex-accent-hover"
        >
          Add
        </button>
        <button
          onClick={() => setIsAdding(false)}
          className="px-2 py-1 text-[10px] bg-hex-border text-hex-text rounded"
        >
          Cancel
        </button>
      </div>
    )
  }

  return (
    <button
      onClick={() => setIsAdding(true)}
      className="w-full flex items-center justify-center gap-1 py-1 text-[10px] text-hex-text-muted hover:text-hex-text border border-dashed border-hex-border rounded hover:border-hex-accent transition-colors"
    >
      <Plus size={10} />
      Add config
    </button>
  )
}
