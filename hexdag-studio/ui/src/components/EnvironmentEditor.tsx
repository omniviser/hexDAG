import { useState, useEffect, useMemo } from 'react'
import {
  Server,
  Plus,
  Trash2,
  ChevronDown,
  ChevronRight,
  Plug,
  Check,
  X,
  Boxes,
  RefreshCw,
} from 'lucide-react'
import yaml from 'yaml'
import { useStudioStore } from '../lib/store'

interface PortConfig {
  adapter: string
  config: Record<string, unknown>
}

interface EnvironmentConfig {
  name: string
  description: string
  ports: Record<string, PortConfig>
  node_overrides: Record<string, Record<string, PortConfig>>
}

interface NodePortsInfo {
  nodeName: string
  nodeKind: string
  ports: Record<string, PortConfig>
  requiredPorts: string[]  // Ports this node type requires
}

interface AdapterInfo {
  name: string
  port_type: string
  description: string
  config_schema?: {
    type: string
    properties?: Record<string, {
      type: string
      default?: unknown
      description?: string
    }>
    required?: string[]
  }
  secrets?: string[]
}

interface NodeTypeInfo {
  kind: string
  name: string
  required_ports: string[]
}

const API_BASE = '/api'

/**
 * Parse environments from YAML content (spec.environments section)
 */
function parseEnvironmentsFromYaml(yamlContent: string): EnvironmentConfig[] {
  if (!yamlContent?.trim()) return []

  try {
    const parsed = yaml.parse(yamlContent)
    if (!parsed?.spec?.environments) return []

    const envs = parsed.spec.environments
    if (typeof envs !== 'object') return []

    return Object.entries(envs).map(([name, envData]) => {
      const data = (envData || {}) as Record<string, unknown>

      // Parse ports
      const ports: Record<string, PortConfig> = {}
      if (data.ports && typeof data.ports === 'object') {
        for (const [portName, portData] of Object.entries(data.ports as Record<string, unknown>)) {
          if (portData && typeof portData === 'object') {
            const pd = portData as Record<string, unknown>
            ports[portName] = {
              adapter: String(pd.adapter || ''),
              config: (pd.config as Record<string, unknown>) || {},
            }
          }
        }
      }

      // Parse node_overrides
      const node_overrides: Record<string, Record<string, PortConfig>> = {}
      if (data.node_overrides && typeof data.node_overrides === 'object') {
        for (const [nodeName, nodePorts] of Object.entries(data.node_overrides as Record<string, unknown>)) {
          if (nodePorts && typeof nodePorts === 'object') {
            node_overrides[nodeName] = {}
            for (const [portName, portData] of Object.entries(nodePorts as Record<string, unknown>)) {
              if (portData && typeof portData === 'object') {
                const pd = portData as Record<string, unknown>
                node_overrides[nodeName][portName] = {
                  adapter: String(pd.adapter || ''),
                  config: (pd.config as Record<string, unknown>) || {},
                }
              }
            }
          }
        }
      }

      return {
        name,
        description: String(data.description || ''),
        ports,
        node_overrides,
      }
    })
  } catch (e) {
    console.error('Failed to parse environments from YAML:', e)
    return []
  }
}

/**
 * Parse node-level ports from YAML content (spec.nodes[].spec.ports)
 * Also includes nodes that require ports but don't have them configured yet.
 */
function parseNodePortsFromYaml(yamlContent: string, nodeTypes: NodeTypeInfo[]): NodePortsInfo[] {
  if (!yamlContent?.trim()) return []

  try {
    const parsed = yaml.parse(yamlContent)
    if (!parsed?.spec?.nodes || !Array.isArray(parsed.spec.nodes)) return []

    const nodePorts: NodePortsInfo[] = []

    // Build a map of node kind -> required ports
    const requiredPortsMap = new Map<string, string[]>()
    for (const nodeType of nodeTypes) {
      requiredPortsMap.set(nodeType.kind, nodeType.required_ports || [])
    }

    for (const node of parsed.spec.nodes) {
      if (!node?.metadata?.name) continue

      const nodeKind = node.kind || 'unknown'
      const requiredPorts = requiredPortsMap.get(nodeKind) || []

      const ports: Record<string, PortConfig> = {}
      if (node.spec?.ports && typeof node.spec.ports === 'object') {
        for (const [portName, portData] of Object.entries(node.spec.ports as Record<string, unknown>)) {
          if (portData && typeof portData === 'object') {
            const pd = portData as Record<string, unknown>
            ports[portName] = {
              adapter: String(pd.adapter || ''),
              config: (pd.config as Record<string, unknown>) || {},
            }
          }
        }
      }

      // Include node if it has ports configured OR if it requires ports
      if (Object.keys(ports).length > 0 || requiredPorts.length > 0) {
        nodePorts.push({
          nodeName: node.metadata.name,
          nodeKind,
          ports,
          requiredPorts,
        })
      }
    }

    return nodePorts
  } catch (e) {
    console.error('Failed to parse node ports from YAML:', e)
    return []
  }
}

/**
 * Update environments in YAML content
 */
function updateEnvironmentsInYaml(yamlContent: string, environments: EnvironmentConfig[]): string {
  try {
    const parsed = yaml.parse(yamlContent) || {}

    // Ensure spec exists
    if (!parsed.spec) {
      parsed.spec = {}
    }

    // Build environments object
    const envObj: Record<string, unknown> = {}
    for (const env of environments) {
      const envData: Record<string, unknown> = {}

      if (env.description) {
        envData.description = env.description
      }

      if (Object.keys(env.ports).length > 0) {
        envData.ports = {}
        for (const [portName, portConfig] of Object.entries(env.ports)) {
          (envData.ports as Record<string, unknown>)[portName] = {
            adapter: portConfig.adapter,
            ...(Object.keys(portConfig.config).length > 0 ? { config: portConfig.config } : {}),
          }
        }
      }

      if (Object.keys(env.node_overrides).length > 0) {
        envData.node_overrides = {}
        for (const [nodeName, nodePorts] of Object.entries(env.node_overrides)) {
          (envData.node_overrides as Record<string, unknown>)[nodeName] = {}
          for (const [portName, portConfig] of Object.entries(nodePorts)) {
            ((envData.node_overrides as Record<string, Record<string, unknown>>)[nodeName])[portName] = {
              adapter: portConfig.adapter,
              ...(Object.keys(portConfig.config).length > 0 ? { config: portConfig.config } : {}),
            }
          }
        }
      }

      envObj[env.name] = Object.keys(envData).length > 0 ? envData : {}
    }

    // Update or remove environments section
    if (Object.keys(envObj).length > 0) {
      parsed.spec.environments = envObj
    } else {
      delete parsed.spec.environments
    }

    return yaml.stringify(parsed, { indent: 2, lineWidth: 120 })
  } catch (e) {
    console.error('Failed to update environments in YAML:', e)
    return yamlContent
  }
}

/**
 * Update a node's ports in YAML content
 */
function updateNodePortInYaml(
  yamlContent: string,
  nodeName: string,
  portType: string,
  config: PortConfig | null
): string {
  try {
    const parsed = yaml.parse(yamlContent) || {}

    if (!parsed.spec?.nodes || !Array.isArray(parsed.spec.nodes)) {
      return yamlContent
    }

    // Find the node and update its ports
    for (const node of parsed.spec.nodes) {
      if (node?.metadata?.name === nodeName) {
        if (!node.spec) node.spec = {}
        if (!node.spec.ports) node.spec.ports = {}

        if (config === null) {
          // Delete the port
          delete node.spec.ports[portType]
          if (Object.keys(node.spec.ports).length === 0) {
            delete node.spec.ports
          }
        } else {
          // Update/add the port
          node.spec.ports[portType] = {
            adapter: config.adapter,
            ...(Object.keys(config.config).length > 0 ? { config: config.config } : {}),
          }
        }
        break
      }
    }

    return yaml.stringify(parsed, { indent: 2, lineWidth: 120 })
  } catch (e) {
    console.error('Failed to update node port in YAML:', e)
    return yamlContent
  }
}

export default function EnvironmentEditor() {
  const { currentFile, yamlContent, setYamlContent } = useStudioStore()

  const [adapters, setAdapters] = useState<AdapterInfo[]>([])
  const [nodeTypes, setNodeTypes] = useState<NodeTypeInfo[]>([])
  const [selectedEnv, setSelectedEnv] = useState<string | null>(null)
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['nodePorts', 'ports', 'envs']))

  // Parse environments directly from YAML content
  const environments = useMemo(() => parseEnvironmentsFromYaml(yamlContent), [yamlContent])

  // Parse node-level ports from YAML content (depends on nodeTypes for required ports)
  const nodePorts = useMemo(() => parseNodePortsFromYaml(yamlContent, nodeTypes), [yamlContent, nodeTypes])

  // Auto-select first environment when environments change
  useEffect(() => {
    if (environments.length > 0 && !environments.find(e => e.name === selectedEnv)) {
      setSelectedEnv(environments[0].name)
    } else if (environments.length === 0) {
      setSelectedEnv(null)
    }
  }, [environments, selectedEnv])

  // Load adapters and node types from registry
  useEffect(() => {
    loadAdapters()
    loadNodeTypes()
  }, [])

  const loadAdapters = async () => {
    try {
      // Add cache-busting to ensure fresh data
      const response = await fetch(`${API_BASE}/registry/adapters?_t=${Date.now()}`)
      if (response.ok) {
        const data = await response.json()
        console.log(`[${new Date().toISOString()}] Loaded ${data?.length || 0} adapters`)

        // Debug: Show all adapters with their schema info
        if (Array.isArray(data)) {
          for (const adapter of data) {
            const schema = adapter.config_schema
            const props = schema?.properties
            const propCount = props ? Object.keys(props).length : 0
            console.log(`  - ${adapter.name} (${adapter.port_type}): ${propCount} config properties`)
          }
        }

        // Specifically debug MockLLM
        const mockLlm = data?.find((a: AdapterInfo) => a.name === 'MockLLM')
        if (mockLlm) {
          console.log('MockLLM details:', {
            name: mockLlm.name,
            port_type: mockLlm.port_type,
            config_schema: mockLlm.config_schema,
            propertyNames: mockLlm.config_schema?.properties
              ? Object.keys(mockLlm.config_schema.properties)
              : [],
          })
        } else {
          console.log('MockLLM not found! Available:', data?.map((a: AdapterInfo) => a.name))
        }

        setAdapters(Array.isArray(data) ? data : [])
      } else {
        console.error('Failed to load adapters:', response.status, await response.text())
      }
    } catch (error) {
      console.error('Failed to load adapters:', error)
    }
  }

  const loadNodeTypes = async () => {
    try {
      const response = await fetch(`${API_BASE}/registry/nodes`)
      if (response.ok) {
        const data = await response.json()
        // Response is { nodes: [...] }
        setNodeTypes(Array.isArray(data.nodes) ? data.nodes : [])
      }
    } catch (error) {
      console.error('Failed to load node types:', error)
    }
  }

  const getCurrentEnv = (): EnvironmentConfig | undefined => {
    return environments.find((e) => e.name === selectedEnv)
  }

  // Update environments in YAML
  const updateEnvironments = (newEnvs: EnvironmentConfig[]) => {
    const newYaml = updateEnvironmentsInYaml(yamlContent, newEnvs)
    setYamlContent(newYaml)
  }

  const updateEnvironment = (envName: string, updates: Partial<EnvironmentConfig>) => {
    const newEnvs = environments.map((env) =>
      env.name === envName ? { ...env, ...updates } : env
    )
    updateEnvironments(newEnvs)
  }

  const updatePort = (portType: string, config: PortConfig) => {
    const env = getCurrentEnv()
    if (!env) return

    updateEnvironment(env.name, {
      ports: { ...env.ports, [portType]: config },
    })
  }

  const deletePort = (portType: string) => {
    const env = getCurrentEnv()
    if (!env) return

    const newPorts = { ...env.ports }
    delete newPorts[portType]
    updateEnvironment(env.name, { ports: newPorts })
  }

  // Update node-level port (stored in spec.nodes[].spec.ports)
  const updateNodePort = (nodeName: string, portType: string, config: PortConfig) => {
    const newYaml = updateNodePortInYaml(yamlContent, nodeName, portType, config)
    setYamlContent(newYaml)
  }

  const deleteNodePort = (nodeName: string, portType: string) => {
    const newYaml = updateNodePortInYaml(yamlContent, nodeName, portType, null)
    setYamlContent(newYaml)
  }

  const createEnvironment = () => {
    const name = prompt('Environment name (e.g., local, dev, prod):')
    if (!name) return

    if (environments.some((e) => e.name === name)) {
      alert('Environment already exists')
      return
    }

    const newEnv: EnvironmentConfig = {
      name,
      description: '',
      ports: {},
      node_overrides: {},
    }

    updateEnvironments([...environments, newEnv])
    setSelectedEnv(name)
  }

  const deleteEnvironment = () => {
    if (!selectedEnv) return

    if (!confirm(`Delete environment "${selectedEnv}"?`)) return

    const newEnvs = environments.filter((e) => e.name !== selectedEnv)
    updateEnvironments(newEnvs)
    setSelectedEnv(newEnvs[0]?.name || null)
  }

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev)
      if (next.has(section)) {
        next.delete(section)
      } else {
        next.add(section)
      }
      return next
    })
  }

  const getAdaptersForPort = (portType: string) => {
    const filtered = adapters.filter((a) => a.port_type === portType)
    console.log('getAdaptersForPort:', { portType, totalAdapters: adapters.length, filtered: filtered.length, adapterPortTypes: adapters.map(a => a.port_type) })
    return filtered
  }

  const getPortTypes = () => {
    const types = new Set(adapters.map((a) => a.port_type))
    return Array.from(types).sort()
  }

  const env = getCurrentEnv()

  if (!currentFile && !yamlContent?.trim()) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 text-center bg-hex-surface">
        <Server size={32} className="text-hex-text-muted opacity-30 mb-2" />
        <p className="text-xs text-hex-text-muted">Open a pipeline to edit environments</p>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-hex-surface">
      {/* Header */}
      <div className="p-3 border-b border-hex-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded bg-hex-accent/20 flex items-center justify-center">
              <Server size={16} className="text-hex-accent" />
            </div>
            <div>
              <h3 className="text-sm font-medium text-hex-text">Ports & Envs</h3>
              <p className="text-[10px] text-hex-text-muted">
                Configure adapters for nodes
              </p>
            </div>
          </div>
          <button
            onClick={loadAdapters}
            className="p-1.5 hover:bg-hex-border/50 rounded transition-colors"
            title="Refresh adapters"
          >
            <RefreshCw size={12} className="text-hex-text-muted" />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Node-Level Ports Section - always visible */}
        <div className="border-b border-hex-border">
          <button
            onClick={() => toggleSection('nodePorts')}
            className="w-full flex items-center gap-2 p-3 hover:bg-hex-border/30 transition-colors"
          >
            {expandedSections.has('nodePorts') ? (
              <ChevronDown size={14} className="text-hex-text-muted" />
            ) : (
              <ChevronRight size={14} className="text-hex-text-muted" />
            )}
            <Boxes size={14} className="text-hex-warning" />
            <span className="text-xs font-medium text-hex-text">Node Ports</span>
            <span className="ml-auto text-[10px] text-hex-text-muted">
              {nodePorts.length} node{nodePorts.length !== 1 ? 's' : ''} with ports
            </span>
          </button>

          {expandedSections.has('nodePorts') && (
            <div className="px-3 pb-3 space-y-2">
              {nodePorts.length === 0 ? (
                <p className="text-[10px] text-hex-text-muted italic py-2 text-center">
                  No nodes require ports
                </p>
              ) : (
                nodePorts.map((nodeInfo) => {
                  // Get unconfigured required ports (check both node-level and global env ports)
                  const currentEnvPorts = env?.ports || {}
                  const missingPorts = nodeInfo.requiredPorts.filter(pt =>
                    !nodeInfo.ports[pt] && !currentEnvPorts[pt]
                  )

                  return (
                    <div key={nodeInfo.nodeName} className="p-2 bg-hex-bg rounded border border-hex-border">
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <span className="text-[11px] font-medium text-hex-warning">{nodeInfo.nodeName}</span>
                          <span className="text-[9px] text-hex-text-muted ml-1">({nodeInfo.nodeKind})</span>
                        </div>
                        {missingPorts.length > 0 && (
                          <span className="text-[9px] text-hex-error">
                            Missing: {missingPorts.join(', ')}
                          </span>
                        )}
                      </div>
                      {/* Show configured ports */}
                      {Object.entries(nodeInfo.ports).map(([portType, portConfig]) => (
                        <PortEditor
                          key={portType}
                          portType={portType}
                          config={portConfig}
                          adapters={getAdaptersForPort(portType)}
                          onChange={(config) => updateNodePort(nodeInfo.nodeName, portType, config)}
                          onDelete={() => deleteNodePort(nodeInfo.nodeName, portType)}
                          compact
                        />
                      ))}
                      {/* Add port button - only show required ports that aren't configured */}
                      {missingPorts.length > 0 && (
                        <AddPortButton
                          portTypes={missingPorts}
                          adapters={adapters}
                          onAdd={(portType, config) => updateNodePort(nodeInfo.nodeName, portType, config)}
                          compact
                        />
                      )}
                    </div>
                  )
                })
              )}
              <p className="text-[9px] text-hex-text-muted text-center">
                Only shows nodes that require ports (llm, memory, etc.)
              </p>
            </div>
          )}
        </div>

        {/* Environments Section */}
        <div className="border-b border-hex-border">
          <button
            onClick={() => toggleSection('envs')}
            className="w-full flex items-center gap-2 p-3 hover:bg-hex-border/30 transition-colors"
          >
            {expandedSections.has('envs') ? (
              <ChevronDown size={14} className="text-hex-text-muted" />
            ) : (
              <ChevronRight size={14} className="text-hex-text-muted" />
            )}
            <Server size={14} className="text-hex-accent" />
            <span className="text-xs font-medium text-hex-text">Environments</span>
            <span className="ml-auto text-[10px] text-hex-text-muted">
              {environments.length} env{environments.length !== 1 ? 's' : ''}
            </span>
          </button>

          {expandedSections.has('envs') && (
            <div className="px-3 pb-3">
              {/* Environment Tabs */}
              <div className="flex items-center gap-1 mb-3 overflow-x-auto pb-1">
                {environments.map((e) => (
                  <button
                    key={e.name}
                    onClick={() => setSelectedEnv(e.name)}
                    className={`flex items-center gap-1 px-2 py-1 text-[10px] font-medium rounded whitespace-nowrap transition-colors ${
                      selectedEnv === e.name
                        ? 'bg-hex-accent text-white'
                        : 'bg-hex-bg border border-hex-border text-hex-text-muted hover:text-hex-text'
                    }`}
                  >
                    <Server size={10} />
                    {e.name}
                  </button>
                ))}
                <button
                  onClick={createEnvironment}
                  className="flex items-center gap-1 px-2 py-1 text-[10px] text-hex-text-muted hover:text-hex-text border border-dashed border-hex-border rounded"
                >
                  <Plus size={10} />
                  New
                </button>
              </div>

              {/* Selected Environment Editor */}
              {env ? (
                <div className="space-y-3">
                  {/* Description */}
                  <div>
                    <label className="block text-[10px] font-medium text-hex-text-muted mb-1 uppercase tracking-wider">
                      Description
                    </label>
                    <input
                      type="text"
                      value={env.description}
                      onChange={(e) => updateEnvironment(env.name, { description: e.target.value })}
                      placeholder="Environment description..."
                      className="w-full bg-hex-bg border border-hex-border rounded px-2 py-1.5 text-xs text-hex-text focus:border-hex-accent focus:outline-none"
                    />
                  </div>

                  {/* Global Ports */}
                  <div>
                    <div className="flex items-center gap-1 text-[10px] font-medium text-hex-text-muted mb-2 uppercase tracking-wider">
                      <Plug size={10} />
                      Global Ports
                      <span className="ml-auto font-normal normal-case">
                        {Object.keys(env.ports).length} configured
                      </span>
                    </div>
                    <div className="space-y-2">
                      {Object.entries(env.ports).map(([portType, portConfig]) => (
                        <PortEditor
                          key={portType}
                          portType={portType}
                          config={portConfig}
                          adapters={getAdaptersForPort(portType)}
                          onChange={(config) => updatePort(portType, config)}
                          onDelete={() => deletePort(portType)}
                        />
                      ))}
                      <AddPortButton
                        portTypes={getPortTypes().filter((pt) => !env.ports[pt])}
                        adapters={adapters}
                        onAdd={(portType, config) => updatePort(portType, config)}
                      />
                    </div>
                  </div>

                  {/* Delete Environment */}
                  <button
                    onClick={deleteEnvironment}
                    className="w-full flex items-center justify-center gap-1 py-2 text-[10px] text-hex-error hover:bg-hex-error/10 border border-hex-error/30 rounded transition-colors"
                  >
                    <Trash2 size={10} />
                    Delete Environment
                  </button>
                </div>
              ) : environments.length === 0 ? (
                <div className="text-center py-4">
                  <p className="text-[10px] text-hex-text-muted mb-2">
                    No environments defined
                  </p>
                  <button
                    onClick={createEnvironment}
                    className="flex items-center gap-1 px-3 py-1.5 text-[10px] font-medium bg-hex-accent text-white rounded hover:bg-hex-accent/90 mx-auto"
                  >
                    <Plus size={10} />
                    Create Environment
                  </button>
                </div>
              ) : (
                <p className="text-[10px] text-hex-text-muted text-center py-2">
                  Select an environment above
                </p>
              )}

              <p className="text-[9px] text-hex-text-muted text-center mt-3">
                Environments stored in <code className="bg-hex-bg px-1 rounded">spec.environments</code>
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Port Editor Component
function PortEditor({
  portType,
  config,
  adapters,
  onChange,
  onDelete,
  compact = false,
}: {
  portType: string
  config: PortConfig
  adapters: AdapterInfo[]
  onChange: (config: PortConfig) => void
  onDelete: () => void
  compact?: boolean
}) {
  const [expanded, setExpanded] = useState(!compact)
  const adapter = adapters.find((a) => a.name === config.adapter)

  // Debug logging
  const configSchema = adapter?.config_schema
  const schemaProps = configSchema?.properties
  if (adapter) {
    console.log('PortEditor debug for', adapter.name, ':')
    console.log('  - config_schema:', configSchema)
    console.log('  - typeof config_schema:', typeof configSchema)
    console.log('  - config_schema keys:', configSchema ? Object.keys(configSchema) : 'N/A')
    console.log('  - properties value:', schemaProps)
    console.log('  - typeof properties:', typeof schemaProps)
    console.log('  - properties is truthy:', !!schemaProps)
    console.log('  - JSON.stringify(properties):', JSON.stringify(schemaProps))
    if (schemaProps && typeof schemaProps === 'object') {
      console.log('  - Object.keys(properties):', Object.keys(schemaProps))
      console.log('  - Object.keys(properties).length:', Object.keys(schemaProps).length)
    }
  }

  return (
    <div className={`${compact ? '' : 'p-2 bg-hex-bg rounded border border-hex-border'}`}>
      <div className="flex items-center gap-2">
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-hex-text-muted hover:text-hex-text"
        >
          {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        </button>
        <span className="text-[10px] font-medium text-hex-text-muted uppercase">{portType}</span>
        <select
          value={config.adapter}
          onChange={(e) => onChange({ ...config, adapter: e.target.value, config: {} })}
          className="flex-1 bg-hex-surface border border-hex-border rounded px-1.5 py-0.5 text-[10px] text-hex-text focus:border-hex-accent focus:outline-none"
        >
          <option value="">-- Select adapter --</option>
          {adapters.map((a) => (
            <option key={a.name} value={a.name}>
              {a.name}
            </option>
          ))}
        </select>
        <button onClick={onDelete} className="p-1 hover:bg-hex-error/20 rounded">
          <X size={10} className="text-hex-error" />
        </button>
      </div>

      {expanded && adapter && adapter.config_schema?.properties && (
        <div className="mt-2 pl-5 space-y-1">
          {Object.entries(adapter.config_schema.properties).map(([key, schema]) => {
            const isRequired = adapter.config_schema?.required?.includes(key) ?? false
            return (
              <div key={key} className="flex items-center gap-2">
                <label className="text-[9px] text-hex-text-muted w-20 truncate" title={key}>
                  {key}
                  {isRequired && <span className="text-hex-error">*</span>}
                </label>
                <input
                  type={schema.type === 'number' || schema.type === 'integer' ? 'number' : 'text'}
                  value={String(config.config[key] ?? schema.default ?? '')}
                  onChange={(e) =>
                    onChange({
                      ...config,
                      config: { ...config.config, [key]: e.target.value },
                    })
                  }
                  placeholder={schema.default !== undefined ? String(schema.default) : ''}
                  className="flex-1 bg-hex-surface border border-hex-border rounded px-1.5 py-0.5 text-[10px] text-hex-text font-mono focus:border-hex-accent focus:outline-none"
                />
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// Add Port Button
function AddPortButton({
  portTypes,
  adapters,
  onAdd,
  compact = false,
}: {
  portTypes: string[]
  adapters: AdapterInfo[]
  onAdd: (portType: string, config: PortConfig) => void
  compact?: boolean
}) {
  const [isAdding, setIsAdding] = useState(false)
  const [selectedType, setSelectedType] = useState('')

  const handleAdd = () => {
    if (!selectedType) return

    const defaultAdapter = adapters.find((a) => a.port_type === selectedType)
    onAdd(selectedType, {
      adapter: defaultAdapter?.name || '',
      config: {},
    })
    setSelectedType('')
    setIsAdding(false)
  }

  if (isAdding) {
    return (
      <div className={`flex gap-1 ${compact ? 'mt-1' : ''}`}>
        <select
          value={selectedType}
          onChange={(e) => setSelectedType(e.target.value)}
          className="flex-1 bg-hex-bg border border-hex-border rounded px-2 py-1 text-[10px] text-hex-text focus:border-hex-accent focus:outline-none"
        >
          <option value="">-- Select port type --</option>
          {portTypes.map((pt) => (
            <option key={pt} value={pt}>
              {pt}
            </option>
          ))}
        </select>
        <button
          onClick={handleAdd}
          disabled={!selectedType}
          className="px-2 py-1 text-[10px] bg-hex-accent text-white rounded hover:bg-hex-accent-hover disabled:opacity-50"
        >
          <Check size={10} />
        </button>
        <button
          onClick={() => setIsAdding(false)}
          className="px-2 py-1 text-[10px] bg-hex-border text-hex-text rounded"
        >
          <X size={10} />
        </button>
      </div>
    )
  }

  if (portTypes.length === 0) {
    return null
  }

  return (
    <button
      onClick={() => setIsAdding(true)}
      className={`w-full flex items-center justify-center gap-1 py-1.5 text-[10px] text-hex-text-muted hover:text-hex-text border border-dashed border-hex-border rounded hover:border-hex-accent transition-colors ${compact ? 'mt-1' : ''}`}
    >
      <Plus size={10} />
      {compact ? 'Add port' : 'Add Port'}
    </button>
  )
}
