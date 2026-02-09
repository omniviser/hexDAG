import { useState, useEffect } from 'react'
import {
  Play,
  FlaskConical,
  Loader2,
  CheckCircle,
  XCircle,
  Clock,
  ChevronDown,
  ChevronRight,
  AlertCircle,
  Zap,
  Server,
  Plug,
  FolderOpen,
  FileCode,
  Info,
} from 'lucide-react'
import { useStudioStore } from '../lib/store'

interface NodeResult {
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
  output?: unknown
  error?: string
  duration_ms?: number
}

interface ExecutionResult {
  success: boolean
  nodes: NodeResult[]
  final_output?: unknown
  error?: string
  duration_ms: number
  environment?: string
  environment_source?: string
}

interface PortConfig {
  adapter: string
  config: Record<string, unknown>
}

interface EnvironmentInfo {
  name: string
  description: string
  ports: Record<string, PortConfig>
  node_overrides: Record<string, Record<string, PortConfig>>
}

interface EnvironmentsData {
  environments: EnvironmentInfo[]
  current: string
  source: string
}

const API_BASE = '/api'

// Default environment used when no environment is configured or discovery fails
const DEFAULT_ENVIRONMENT: EnvironmentInfo = {
  name: 'local',
  description: 'Default mock environment',
  ports: { llm: { adapter: 'mock_llm', config: {} } },
  node_overrides: {}
}

export default function ExecutionPanel() {
  const {
    yamlContent,
    currentFile,
    setIsExecuting,
    setNodeExecutionStatus,
    resetExecutionStatus,
    initializeExecutionStatus,
  } = useStudioStore()
  const [inputsJson, setInputsJson] = useState('{\\n  \\n}')
  const [timeout, setTimeoutValue] = useState(30)
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState<ExecutionResult | null>(null)
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set())
  const [useStreaming, setUseStreaming] = useState(true)
  const [liveNodes, setLiveNodes] = useState<Map<string, NodeResult>>(new Map())
  const [currentWave, setCurrentWave] = useState<number | null>(null)

  // Environment state - now discovered per-pipeline
  const [environments, setEnvironments] = useState<EnvironmentInfo[]>([])
  const [currentEnv, setCurrentEnv] = useState<string>('local')
  const [envSource, setEnvSource] = useState<string>('default')
  const [showEnvPorts, setShowEnvPorts] = useState(false)
  const [isLoadingEnvs, setIsLoadingEnvs] = useState(false)

  // Load environments when file or YAML content changes (debounced)
  useEffect(() => {
    // Debounce to avoid too many API calls during typing
    const timer = window.setTimeout(() => {
      discoverEnvironments()
    }, 300)
    return () => window.clearTimeout(timer)
  }, [currentFile, yamlContent])

  const discoverEnvironments = async () => {
    // Only discover if we have content or a file path
    if (!yamlContent?.trim() && !currentFile) {
      setEnvironments([DEFAULT_ENVIRONMENT])
      setCurrentEnv('local')
      setEnvSource('default')
      return
    }

    setIsLoadingEnvs(true)
    try {
      const response = await fetch(`${API_BASE}/environments/discover`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pipeline_path: currentFile,
          yaml_content: yamlContent,
        }),
      })
      if (response.ok) {
        const data: EnvironmentsData = await response.json()
        console.log('Discovered environments:', data)
        setEnvironments(data.environments || [])
        setCurrentEnv(data.current || 'local')
        setEnvSource(data.source || 'default')
      } else {
        console.error('Environment discovery failed:', response.status)
        throw new Error(`HTTP ${response.status}`)
      }
    } catch (error) {
      console.error('Failed to discover environments:', error)
      // Set default environment on error
      setEnvironments([DEFAULT_ENVIRONMENT])
      setCurrentEnv('local')
      setEnvSource('default')
    } finally {
      setIsLoadingEnvs(false)
    }
  }

  const getCurrentEnvInfo = (): EnvironmentInfo | undefined => {
    return environments.find(e => e.name === currentEnv)
  }

  const getSourceIcon = () => {
    switch (envSource) {
      case 'inline':
        return <FileCode size={10} className="text-hex-accent" />
      case 'folder':
        return <FolderOpen size={10} className="text-green-500" />
      default:
        return <Info size={10} className="text-hex-text-muted" />
    }
  }

  const getSourceLabel = () => {
    switch (envSource) {
      case 'inline':
        return 'From YAML'
      case 'folder':
        return 'From folder'
      default:
        return 'Default'
    }
  }

  const handleRun = async (dryRun: boolean = false) => {
    if (!yamlContent.trim()) {
      setResult({
        success: false,
        nodes: [],
        error: 'No pipeline YAML content to execute',
        duration_ms: 0,
      })
      return
    }

    // Parse inputs
    let inputs: Record<string, unknown> = {}
    try {
      const trimmed = inputsJson.trim()
      if (trimmed && trimmed !== '{}') {
        inputs = JSON.parse(trimmed)
      }
    } catch (e) {
      setResult({
        success: false,
        nodes: [],
        error: `Invalid JSON in inputs: ${e instanceof Error ? e.message : 'Parse error'}`,
        duration_ms: 0,
      })
      return
    }

    setIsRunning(true)
    setResult(null)
    setLiveNodes(new Map())
    setCurrentWave(null)
    resetExecutionStatus()  // Reset canvas node states
    setIsExecuting(true)    // Tell store we're executing

    // Use streaming for real execution, regular for dry run
    if (!dryRun && useStreaming) {
      await handleStreamingRun(inputs)
    } else {
      await handleRegularRun(inputs, dryRun)
    }
  }

  const handleStreamingRun = async (inputs: Record<string, unknown>) => {
    try {
      const response = await fetch(`${API_BASE}/execute/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: yamlContent,
          inputs,
          environment: currentEnv,
          pipeline_path: currentFile,
          timeout,
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        setResult({
          success: false,
          nodes: [],
          error: `API error: ${response.status} - ${errorText}`,
          duration_ms: 0,
        })
        setIsRunning(false)
        return
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('No response body')
      }

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // Parse SSE events from buffer
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // Keep incomplete line in buffer

        let eventType = ''
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim()
          } else if (line.startsWith('data: ') && eventType) {
            try {
              const data = JSON.parse(line.slice(6))
              handleStreamEvent(eventType, data)
            } catch {
              console.error('Failed to parse SSE data:', line)
            }
            eventType = ''
          }
        }
      }
    } catch (e) {
      setResult({
        success: false,
        nodes: [],
        error: `Stream error: ${e instanceof Error ? e.message : 'Unknown error'}`,
        duration_ms: 0,
      })
      setIsExecuting(false)  // Stop canvas animations on error
    } finally {
      setIsRunning(false)
    }
  }

  const handleStreamEvent = (eventType: string, data: Record<string, unknown>) => {
    switch (eventType) {
      case 'plan':
        // Initialize all nodes as pending
        const nodeNames = data.node_names as string[]
        setLiveNodes(new Map(
          nodeNames.map(name => [name, { name, status: 'pending' as const }])
        ))
        // Also initialize the global store for canvas nodes
        initializeExecutionStatus(nodeNames)
        break

      case 'wave_start':
        setCurrentWave(data.wave_index as number)
        break

      case 'node_start': {
        const nodeName = data.name as string
        setLiveNodes(prev => {
          const next = new Map(prev)
          next.set(nodeName, {
            name: nodeName,
            status: 'running',
          })
          return next
        })
        // Update store for canvas node animation
        setNodeExecutionStatus(nodeName, { status: 'running' })
        break
      }

      case 'node_complete': {
        const nodeName = data.name as string
        const duration_ms = data.duration_ms as number | undefined
        setLiveNodes(prev => {
          const next = new Map(prev)
          next.set(nodeName, {
            name: nodeName,
            status: 'completed',
            output: data.output,
            duration_ms,
          })
          return next
        })
        // Update store for canvas node animation
        setNodeExecutionStatus(nodeName, {
          status: 'completed',
          output: data.output,
          duration_ms,
        })
        break
      }

      case 'node_failed': {
        const nodeName = data.name as string
        const error = data.error as string | undefined
        setLiveNodes(prev => {
          const next = new Map(prev)
          next.set(nodeName, {
            name: nodeName,
            status: 'failed',
            error,
          })
          return next
        })
        // Update store for canvas node animation
        setNodeExecutionStatus(nodeName, { status: 'failed', error })
        break
      }

      case 'complete':
        setCurrentWave(null)
        setIsExecuting(false)  // Stop canvas animations
        setResult({
          success: data.success as boolean,
          nodes: Array.from(liveNodes.values()).map(n => ({
            ...n,
            status: n.status === 'running' ? 'completed' : n.status,
          })) as NodeResult[],
          final_output: data.final_output,
          error: data.error as string | undefined,
          duration_ms: data.duration_ms as number,
          environment: data.environment as string | undefined,
          environment_source: data.environment_source as string | undefined,
        })
        break

      case 'error':
        setCurrentWave(null)
        setIsExecuting(false)  // Stop canvas animations
        setResult({
          success: false,
          nodes: Array.from(liveNodes.values()) as NodeResult[],
          error: data.error as string,
          duration_ms: data.duration_ms as number,
        })
        break
    }
  }

  const handleRegularRun = async (inputs: Record<string, unknown>, dryRun: boolean) => {
    try {
      const endpoint = dryRun ? `${API_BASE}/execute/dry-run` : `${API_BASE}/execute`
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: yamlContent,
          inputs,
          environment: currentEnv,
          pipeline_path: currentFile,
          timeout,
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        setResult({
          success: false,
          nodes: [],
          error: `API error: ${response.status} - ${errorText}`,
          duration_ms: 0,
        })
        return
      }

      const data = await response.json()

      if (dryRun) {
        // Dry run returns different format
        setResult({
          success: data.valid ?? true,
          nodes: (data.execution_order || []).map((name: string) => ({
            name,
            status: 'completed' as const,
            output: data.dependency_map?.[name],
          })),
          final_output: data,
          error: data.error,
          duration_ms: 0,
        })
      } else {
        setResult(data)
      }
    } catch (e) {
      setResult({
        success: false,
        nodes: [],
        error: `Network error: ${e instanceof Error ? e.message : 'Unknown error'}`,
        duration_ms: 0,
      })
    } finally {
      setIsRunning(false)
      setIsExecuting(false)  // Stop canvas animations
    }
  }

  const toggleNodeExpanded = (nodeName: string) => {
    setExpandedNodes((prev) => {
      const next = new Set(prev)
      if (next.has(nodeName)) {
        next.delete(nodeName)
      } else {
        next.add(nodeName)
      }
      return next
    })
  }

  const formatOutput = (output: unknown): string => {
    if (output === undefined || output === null) return ''
    if (typeof output === 'string') return output
    return JSON.stringify(output, null, 2)
  }

  return (
    <div className="h-full flex flex-col bg-hex-surface">
      {/* Header */}
      <div className="p-3 border-b border-hex-border">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-hex-success/20 flex items-center justify-center">
            <Zap size={16} className="text-hex-success" />
          </div>
          <div>
            <h3 className="text-sm font-medium text-hex-text">Execute Pipeline</h3>
            <p className="text-[10px] text-hex-text-muted">Run your pipeline with test inputs</p>
          </div>
        </div>
      </div>

      {/* Environment Selection - Per-Pipeline */}
      <div className="p-3 border-b border-hex-border space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <label className="text-[10px] font-medium text-hex-text-muted uppercase tracking-wider">
              Environment
            </label>
            <div className="flex items-center gap-1 px-1.5 py-0.5 bg-hex-bg rounded text-[9px] text-hex-text-muted">
              {getSourceIcon()}
              <span>{getSourceLabel()}</span>
            </div>
          </div>
          <button
            onClick={() => setShowEnvPorts(!showEnvPorts)}
            className="text-[9px] text-hex-accent hover:text-hex-accent/80"
          >
            {showEnvPorts ? 'Hide ports' : 'Show ports'}
          </button>
        </div>

        {isLoadingEnvs ? (
          <div className="flex items-center justify-center py-2">
            <Loader2 size={14} className="animate-spin text-hex-text-muted" />
            <span className="ml-2 text-[10px] text-hex-text-muted">Discovering...</span>
          </div>
        ) : (
          <div className="flex gap-1 flex-wrap">
            {environments.map((env) => (
              <button
                key={env.name}
                onClick={() => setCurrentEnv(env.name)}
                className={`flex items-center justify-center gap-1 px-2 py-1.5 text-[10px] font-medium rounded transition-colors ${
                  currentEnv === env.name
                    ? 'bg-hex-accent text-white'
                    : 'bg-hex-bg border border-hex-border text-hex-text-muted hover:text-hex-text hover:border-hex-accent/50'
                }`}
                title={env.description}
              >
                <Server size={10} />
                {env.name}
              </button>
            ))}
          </div>
        )}

        {/* Show current environment ports */}
        {showEnvPorts && getCurrentEnvInfo() && (
          <div className="mt-2 p-2 bg-hex-bg rounded border border-hex-border space-y-2">
            {/* Global ports */}
            <div>
              <div className="flex items-center gap-1 text-[9px] text-hex-text-muted mb-1">
                <Plug size={9} />
                <span className="uppercase tracking-wider">Global Adapters:</span>
              </div>
              {Object.entries(getCurrentEnvInfo()!.ports).map(([portName, portConfig]) => (
                <div key={portName} className="flex items-center justify-between text-[10px]">
                  <span className="text-hex-text-muted">{portName}:</span>
                  <span className="text-hex-accent font-mono">{portConfig.adapter}</span>
                </div>
              ))}
              {Object.keys(getCurrentEnvInfo()!.ports).length === 0 && (
                <p className="text-[9px] text-hex-text-muted italic">No ports configured</p>
              )}
            </div>

            {/* Node overrides */}
            {getCurrentEnvInfo()!.node_overrides &&
              Object.keys(getCurrentEnvInfo()!.node_overrides).length > 0 && (
              <div className="pt-2 border-t border-hex-border/50">
                <div className="flex items-center gap-1 text-[9px] text-hex-text-muted mb-1">
                  <Server size={9} />
                  <span className="uppercase tracking-wider">Node Overrides:</span>
                </div>
                {Object.entries(getCurrentEnvInfo()!.node_overrides).map(([nodeName, nodePorts]) => (
                  <div key={nodeName} className="ml-2 mb-1">
                    <span className="text-[10px] text-hex-warning font-medium">{nodeName}:</span>
                    {Object.entries(nodePorts).map(([portName, portConfig]) => (
                      <div key={portName} className="flex items-center justify-between text-[10px] ml-2">
                        <span className="text-hex-text-muted">{portName}:</span>
                        <span className="text-hex-warning font-mono">{portConfig.adapter}</span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Help text for adding environments */}
        {envSource === 'default' && (
          <div className="flex items-start gap-2 p-2 bg-hex-bg/50 rounded text-[9px] text-hex-text-muted">
            <Info size={12} className="flex-shrink-0 mt-0.5" />
            <div>
              <p>Add environments to customize adapters:</p>
              <p className="mt-1 font-mono">spec.environments.{'{env_name}'}.ports</p>
              <p className="mt-0.5">or create <span className="font-mono">environments/{'{env_name}'}.yaml</span></p>
            </div>
          </div>
        )}
      </div>

      {/* Inputs Section */}
      <div className="p-3 border-b border-hex-border space-y-3">
        <div>
          <label className="block text-[10px] font-medium text-hex-text-muted mb-1 uppercase tracking-wider">
            Pipeline Inputs (JSON)
          </label>
          <textarea
            value={inputsJson}
            onChange={(e) => setInputsJson(e.target.value)}
            placeholder='{\\n  "key": "value"\\n}'
            rows={3}
            className="w-full bg-hex-bg border border-hex-border rounded px-2 py-1.5 text-xs text-hex-text focus:border-hex-accent focus:outline-none font-mono resize-y"
          />
        </div>

        {/* Options */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-[10px] text-hex-text-muted">Timeout:</label>
            <input
              type="number"
              value={timeout}
              onChange={(e) => setTimeoutValue(Number(e.target.value))}
              min={1}
              max={300}
              className="w-12 bg-hex-bg border border-hex-border rounded px-1.5 py-0.5 text-[10px] text-hex-text focus:border-hex-accent focus:outline-none"
            />
            <span className="text-[10px] text-hex-text-muted">sec</span>
          </div>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={useStreaming}
              onChange={(e) => setUseStreaming(e.target.checked)}
              className="w-3 h-3 rounded border-hex-border bg-hex-bg accent-hex-accent"
            />
            <span className="text-[10px] text-hex-text-muted">Live updates</span>
          </label>
        </div>

        {/* Run Buttons */}
        <div className="flex gap-2">
          <button
            onClick={() => handleRun(false)}
            disabled={isRunning || !yamlContent.trim()}
            className="flex-1 flex items-center justify-center gap-1.5 py-2 text-xs font-medium rounded bg-hex-success text-white hover:bg-hex-success/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isRunning ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Play size={14} />
            )}
            Run Pipeline
          </button>
          <button
            onClick={() => handleRun(true)}
            disabled={isRunning || !yamlContent.trim()}
            className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium rounded bg-hex-bg border border-hex-border text-hex-text hover:bg-hex-border/50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <FlaskConical size={14} />
            Dry Run
          </button>
        </div>
      </div>

      {/* Results Section */}
      <div className="flex-1 overflow-y-auto">
        {/* Live execution progress */}
        {isRunning && liveNodes.size > 0 && (
          <div className="p-3 space-y-2">
            <div className="flex items-center gap-2 mb-2">
              <Loader2 size={14} className="animate-spin text-hex-accent" />
              <span className="text-xs text-hex-text-muted">
                Executing{currentWave !== null ? ` (Wave ${currentWave + 1})` : ''}...
              </span>
            </div>
            <div className="space-y-1">
              {Array.from(liveNodes.values()).map((node) => (
                <div
                  key={node.name}
                  className={`flex items-center gap-2 px-2 py-1.5 rounded border transition-all duration-300 ${
                    node.status === 'running'
                      ? 'bg-hex-accent/10 border-hex-accent animate-pulse'
                      : node.status === 'completed'
                      ? 'bg-hex-success/10 border-hex-success/50'
                      : node.status === 'failed'
                      ? 'bg-hex-error/10 border-hex-error/50'
                      : 'bg-hex-bg border-hex-border'
                  }`}
                >
                  {node.status === 'running' ? (
                    <Loader2 size={12} className="animate-spin text-hex-accent" />
                  ) : node.status === 'completed' ? (
                    <CheckCircle size={12} className="text-hex-success" />
                  ) : node.status === 'failed' ? (
                    <XCircle size={12} className="text-hex-error" />
                  ) : (
                    <Clock size={12} className="text-hex-text-muted" />
                  )}
                  <span className={`text-[11px] font-medium ${
                    node.status === 'running' ? 'text-hex-accent' : 'text-hex-text'
                  }`}>
                    {node.name}
                  </span>
                  {node.status === 'running' && (
                    <span className="ml-auto text-[9px] text-hex-accent">executing...</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Simple spinner when not using streaming */}
        {isRunning && liveNodes.size === 0 && (
          <div className="p-6 flex flex-col items-center justify-center gap-2">
            <Loader2 size={24} className="animate-spin text-hex-accent" />
            <p className="text-xs text-hex-text-muted">Executing pipeline...</p>
          </div>
        )}

        {!isRunning && result && (
          <div className="p-3 space-y-3">
            {/* Summary */}
            <div
              className={`p-3 rounded-lg border-l-4 ${
                result.success
                  ? 'bg-hex-success/10 border-hex-success'
                  : 'bg-hex-error/10 border-hex-error'
              }`}
            >
              <div className="flex items-center gap-2">
                {result.success ? (
                  <CheckCircle size={16} className="text-hex-success" />
                ) : (
                  <XCircle size={16} className="text-hex-error" />
                )}
                <span className="text-xs font-medium text-hex-text">
                  {result.success ? 'Execution Successful' : 'Execution Failed'}
                </span>
                {result.duration_ms != null && result.duration_ms > 0 && (
                  <span className="ml-auto flex items-center gap-1 text-[10px] text-hex-text-muted">
                    <Clock size={10} />
                    {result.duration_ms.toFixed(0)}ms
                  </span>
                )}
              </div>
              {result.environment && (
                <div className="mt-1 flex items-center gap-1 text-[9px] text-hex-text-muted">
                  <Server size={9} />
                  <span>Environment: <span className="text-hex-accent">{result.environment}</span></span>
                  {result.environment_source && (
                    <span className="ml-1 px-1 py-0.5 bg-hex-bg rounded text-[8px]">
                      {result.environment_source}
                    </span>
                  )}
                </div>
              )}
              {result.error && (
                <p className="mt-2 text-[11px] text-hex-error font-mono">{result.error}</p>
              )}
            </div>

            {/* Node Results */}
            {result.nodes.length > 0 && (
              <div className="space-y-1">
                <label className="block text-[10px] font-medium text-hex-text-muted uppercase tracking-wider">
                  Node Results ({result.nodes.length})
                </label>
                {result.nodes.map((node) => (
                  <div
                    key={node.name}
                    className="bg-hex-bg rounded border border-hex-border overflow-hidden"
                  >
                    <button
                      onClick={() => toggleNodeExpanded(node.name)}
                      className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-hex-border/30 transition-colors"
                    >
                      {expandedNodes.has(node.name) ? (
                        <ChevronDown size={12} className="text-hex-text-muted" />
                      ) : (
                        <ChevronRight size={12} className="text-hex-text-muted" />
                      )}
                      {node.status === 'completed' ? (
                        <CheckCircle size={12} className="text-hex-success" />
                      ) : node.status === 'failed' ? (
                        <XCircle size={12} className="text-hex-error" />
                      ) : node.status === 'running' ? (
                        <Loader2 size={12} className="animate-spin text-hex-accent" />
                      ) : node.status === 'pending' ? (
                        <Clock size={12} className="text-hex-text-muted" />
                      ) : (
                        <AlertCircle size={12} className="text-hex-warning" />
                      )}
                      <span className="text-[11px] font-medium text-hex-text flex-1 text-left">
                        {node.name}
                      </span>
                      {node.duration_ms != null && (
                        <span className="text-[9px] text-hex-text-muted">
                          {node.duration_ms.toFixed(0)}ms
                        </span>
                      )}
                    </button>
                    {expandedNodes.has(node.name) && (
                      <div className="px-2 pb-2 border-t border-hex-border/50">
                        {node.error ? (
                          <div className="mt-2 p-2 bg-hex-error/10 rounded">
                            <p className="text-[10px] text-hex-error font-mono">{node.error}</p>
                          </div>
                        ) : node.output !== undefined ? (
                          <div className="mt-2">
                            <label className="text-[9px] text-hex-text-muted uppercase">
                              Output:
                            </label>
                            <pre className="mt-1 p-2 bg-hex-surface rounded text-[10px] text-hex-text font-mono overflow-x-auto max-h-40">
                              {formatOutput(node.output)}
                            </pre>
                          </div>
                        ) : (
                          <p className="mt-2 text-[10px] text-hex-text-muted italic">
                            No output
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Final Output */}
            {result.final_output !== undefined && (
              <div>
                <label className="block text-[10px] font-medium text-hex-text-muted mb-1 uppercase tracking-wider">
                  Final Output
                </label>
                <pre className="p-2 bg-hex-bg rounded border border-hex-border text-[10px] text-hex-text font-mono overflow-x-auto max-h-60">
                  {formatOutput(result.final_output)}
                </pre>
              </div>
            )}
          </div>
        )}

        {!isRunning && !result && (
          <div className="p-6 flex flex-col items-center justify-center gap-2 text-center">
            <Play size={32} className="text-hex-text-muted opacity-30" />
            <p className="text-xs text-hex-text-muted">
              Click "Run Pipeline" to execute
            </p>
            <p className="text-[10px] text-hex-text-muted">
              Use "Dry Run" to validate without executing
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
