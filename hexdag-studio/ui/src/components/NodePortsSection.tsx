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
  Wrench,
} from 'lucide-react'
import { getAllPluginAdapters, type PluginAdapter } from '../lib/api'

interface PortConfig {
  adapter: string
  config: Record<string, unknown>
}

type PortsConfig = Record<string, PortConfig | undefined>

interface NodePortsSectionProps {
  nodeId: string
  requiredPorts: string[]
  nodePorts: PortsConfig
  onPortsChange: (ports: PortsConfig) => void
}

const PORT_INFO: Record<string, { icon: typeof Brain; color: string; description: string }> = {
  llm: {
    icon: Brain,
    color: '#6366f1',
    description: 'Language model for this node',
  },
  memory: {
    icon: Database,
    color: '#14b8a6',
    description: 'Memory for this node',
  },
  database: {
    icon: Database,
    color: '#ec4899',
    description: 'Database for this node',
  },
  storage: {
    icon: HardDrive,
    color: '#22c55e',
    description: 'Storage for this node',
  },
  secret: {
    icon: Key,
    color: '#f59e0b',
    description: 'Secrets for this node',
  },
  tool_router: {
    icon: Wrench,
    color: '#8b5cf6',
    description: 'Tool router for agent tools',
  },
}

export default function NodePortsSection({
  nodeId: _nodeId,  // For future use (per-node tracking)
  requiredPorts,
  nodePorts,
  onPortsChange,
}: NodePortsSectionProps) {
  const [adapters, setAdapters] = useState<PluginAdapter[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [expandedPorts, setExpandedPorts] = useState<Set<string>>(new Set(requiredPorts))

  useEffect(() => {
    loadAdapters()
  }, [])

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
    const newPorts = { ...nodePorts }

    if (!adapterName) {
      delete newPorts[portType]
    } else {
      newPorts[portType] = {
        adapter: adapterName,
        config: adapter ? getDefaultConfig(adapter) : {},
      }
    }

    onPortsChange(newPorts)
  }

  const updatePortConfig = (portType: string, key: string, value: unknown) => {
    const newPorts = { ...nodePorts }
    if (newPorts[portType]) {
      newPorts[portType] = {
        ...newPorts[portType]!,
        config: {
          ...newPorts[portType]!.config,
          [key]: value,
        },
      }
      onPortsChange(newPorts)
    }
  }

  const deletePortConfig = (portType: string, key: string) => {
    const newPorts = { ...nodePorts }
    if (newPorts[portType]) {
      const newConfig = { ...newPorts[portType]!.config }
      delete newConfig[key]
      newPorts[portType] = {
        ...newPorts[portType]!,
        config: newConfig,
      }
      onPortsChange(newPorts)
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
          config[key] = `\${${key.toUpperCase()}}`
        }
      }
    }

    return config
  }

  const getAdaptersForPort = (portType: string) => {
    return adapters.filter((a) => a.port_type === portType)
  }

  if (requiredPorts.length === 0) {
    return null
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-4">
        <Loader2 size={16} className="animate-spin text-hex-text-muted" />
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {/* Info banner */}
      <div className="flex items-start gap-2 p-2 bg-hex-accent/10 rounded text-[10px] text-hex-accent">
        <Info size={12} className="flex-shrink-0 mt-0.5" />
        <span>
          Override pipeline-level ports for this node. Leave empty to use global configuration.
        </span>
      </div>

      {/* Port List */}
      {requiredPorts.map((portType) => {
        const portInfo = PORT_INFO[portType] || {
          icon: Plug,
          color: '#6b7280',
          description: `${portType} port`,
        }
        const Icon = portInfo.icon
        const availableAdapters = getAdaptersForPort(portType)
        const currentPort = nodePorts[portType]
        const isExpanded = expandedPorts.has(portType)
        const isConfigured = !!currentPort?.adapter

        return (
          <div key={portType} className="border border-hex-border rounded overflow-hidden">
            <button
              onClick={() => togglePort(portType)}
              className="w-full flex items-center gap-2 p-2 hover:bg-hex-border/30 transition-colors"
            >
              {isExpanded ? (
                <ChevronDown size={12} className="text-hex-text-muted" />
              ) : (
                <ChevronRight size={12} className="text-hex-text-muted" />
              )}
              <div
                className="w-5 h-5 rounded flex items-center justify-center"
                style={{ backgroundColor: `${portInfo.color}20` }}
              >
                <Icon size={10} style={{ color: portInfo.color }} />
              </div>
              <span className="flex-1 text-left text-[11px] font-medium text-hex-text capitalize">
                {portType.replace('_', ' ')}
              </span>
              {isConfigured ? (
                <span
                  className="text-[9px] px-1.5 py-0.5 rounded"
                  style={{ backgroundColor: `${portInfo.color}20`, color: portInfo.color }}
                >
                  {currentPort.adapter}
                </span>
              ) : (
                <span className="text-[9px] text-hex-text-muted italic">Global</span>
              )}
            </button>

            {isExpanded && (
              <div className="px-2 pb-2 space-y-2 border-t border-hex-border bg-hex-bg/30">
                {/* Description */}
                <div className="flex items-start gap-2 p-1.5 text-[9px] text-hex-text-muted">
                  <span>{portInfo.description}</span>
                </div>

                {/* Adapter Selection */}
                <div>
                  <label className="block text-[9px] font-medium text-hex-text-muted mb-1 uppercase tracking-wider">
                    Adapter Override
                  </label>
                  {availableAdapters.length === 0 ? (
                    <p className="text-[9px] text-hex-text-muted italic">
                      No adapters available for {portType}
                    </p>
                  ) : (
                    <select
                      value={currentPort?.adapter || ''}
                      onChange={(e) => setPortAdapter(portType, e.target.value)}
                      className="w-full bg-hex-bg border border-hex-border rounded px-2 py-1 text-[11px] text-hex-text focus:border-hex-accent focus:outline-none"
                    >
                      <option value="">-- Use global config --</option>
                      {availableAdapters.map((adapter) => (
                        <option key={adapter.name} value={adapter.name}>
                          {adapter.name} ({adapter.plugin})
                        </option>
                      ))}
                    </select>
                  )}
                </div>

                {/* Adapter Config - show fields from schema */}
                {currentPort?.adapter && (() => {
                  const selectedAdapter = availableAdapters.find(a => a.name === currentPort.adapter)
                  const schema = selectedAdapter?.config_schema as {
                    properties?: Record<string, { type?: string; default?: unknown; description?: string }>
                    required?: string[]
                  } | undefined
                  const schemaProps = schema?.properties

                  if (!schemaProps || Object.keys(schemaProps).length === 0) {
                    // No schema properties - show manual config entry
                    if (currentPort.config && Object.keys(currentPort.config).length > 0) {
                      return (
                        <div className="space-y-1.5">
                          <label className="block text-[9px] font-medium text-hex-text-muted uppercase tracking-wider">
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
                      )
                    }
                    return null
                  }

                  // Show schema-driven config fields
                  return (
                    <div className="space-y-1.5">
                      <label className="block text-[9px] font-medium text-hex-text-muted uppercase tracking-wider">
                        Configuration
                      </label>
                      {Object.entries(schemaProps).map(([key, propSchema]) => {
                        const isRequired = schema?.required?.includes(key) ?? false
                        const currentValue = currentPort.config?.[key]
                        const defaultValue = propSchema.default

                        return (
                          <div key={key} className="group">
                            <div className="flex items-center justify-between mb-0.5">
                              <label className="text-[9px] text-hex-text-muted" title={propSchema.description}>
                                {key}
                                {isRequired && <span className="text-hex-error ml-0.5">*</span>}
                              </label>
                            </div>
                            <input
                              type={propSchema.type === 'number' || propSchema.type === 'integer' ? 'number' : 'text'}
                              value={currentValue !== undefined ? String(currentValue) : (defaultValue !== undefined ? String(defaultValue) : '')}
                              onChange={(e) => {
                                const value = propSchema.type === 'number' || propSchema.type === 'integer'
                                  ? Number(e.target.value)
                                  : e.target.value
                                updatePortConfig(portType, key, value)
                              }}
                              placeholder={defaultValue !== undefined ? String(defaultValue) : propSchema.description || ''}
                              className="w-full bg-hex-bg border border-hex-border rounded px-1.5 py-0.5 text-[10px] text-hex-text font-mono focus:border-hex-accent focus:outline-none"
                            />
                          </div>
                        )
                      })}
                    </div>
                  )
                })()}
              </div>
            )}
          </div>
        )
      })}
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
  const isSecret =
    name.toLowerCase().includes('key') ||
    name.toLowerCase().includes('secret') ||
    name.toLowerCase().includes('password') ||
    name.toLowerCase().includes('token')

  const isEnvVar = typeof value === 'string' && value.startsWith('${') && value.endsWith('}')

  return (
    <div className="group flex items-start gap-1.5">
      <div className="flex-1">
        <div className="flex items-center justify-between mb-0.5">
          <label className="text-[9px] text-hex-text-muted">{name}</label>
          <button
            onClick={onDelete}
            className="opacity-0 group-hover:opacity-100 p-0.5 hover:bg-hex-error/20 rounded transition-all"
          >
            <X size={8} className="text-hex-error" />
          </button>
        </div>
        <div className="relative">
          <input
            type={isSecret && !isEnvVar ? 'password' : 'text'}
            value={String(value)}
            onChange={(e) => onChange(e.target.value)}
            className="w-full bg-hex-bg border border-hex-border rounded px-1.5 py-0.5 text-[10px] text-hex-text focus:border-hex-accent focus:outline-none font-mono"
            placeholder={isSecret ? '${ENV_VAR}' : ''}
          />
          {isEnvVar && (
            <span className="absolute right-1.5 top-1/2 -translate-y-1/2 text-[8px] px-1 py-0.5 rounded bg-amber-500/20 text-amber-500">
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
          className="flex-1 bg-hex-bg border border-hex-border rounded px-1.5 py-0.5 text-[10px] text-hex-text focus:border-hex-accent focus:outline-none"
        />
        <button
          onClick={handleAdd}
          className="px-1.5 py-0.5 text-[9px] bg-hex-accent text-white rounded hover:bg-hex-accent-hover"
        >
          Add
        </button>
        <button
          onClick={() => setIsAdding(false)}
          className="px-1.5 py-0.5 text-[9px] bg-hex-border text-hex-text rounded"
        >
          Cancel
        </button>
      </div>
    )
  }

  return (
    <button
      onClick={() => setIsAdding(true)}
      className="w-full flex items-center justify-center gap-1 py-0.5 text-[9px] text-hex-text-muted hover:text-hex-text border border-dashed border-hex-border rounded hover:border-hex-accent transition-colors"
    >
      <Plus size={8} />
      Add config
    </button>
  )
}
