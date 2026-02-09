import { memo } from 'react'
import { Handle, Position } from '@xyflow/react'
import {
  AlertCircle,
  Play,
  Loader2,
  CheckCircle,
  XCircle,
  Clock,
} from 'lucide-react'
import type { HexdagNodeData, NodeExecutionStatus } from '../types'
import { getNodeColor, getNodeTemplate, getNodeIconComponent } from '../lib/nodeTemplates'
import { formatValuePreview } from '../lib/formatValue'
import { useStudioStore } from '../lib/store'

interface HexdagNodeProps {
  data: HexdagNodeData
  id: string
  selected?: boolean
}

// Helper to get status color classes
function getStatusStyles(status: NodeExecutionStatus): {
  border: string
  glow: string
  animate: boolean
} {
  switch (status) {
    case 'running':
      return {
        border: 'border-hex-accent',
        glow: 'shadow-lg shadow-hex-accent/40',
        animate: true,
      }
    case 'completed':
      return {
        border: 'border-hex-success',
        glow: 'shadow-md shadow-hex-success/30',
        animate: false,
      }
    case 'failed':
      return {
        border: 'border-hex-error',
        glow: 'shadow-md shadow-hex-error/30',
        animate: false,
      }
    case 'pending':
      return {
        border: 'border-hex-warning/50',
        glow: '',
        animate: false,
      }
    default:
      return {
        border: '',
        glow: '',
        animate: false,
      }
  }
}

function HexdagNode({ data, id, selected }: HexdagNodeProps) {
  // Use shared icon lookup from nodeTemplates
  const Icon = getNodeIconComponent(data.kind)
  const color = getNodeColor(data.kind)
  const template = getNodeTemplate(data.kind)

  // Get execution status from store
  const executionState = useStudioStore((state) => state.nodeExecutionStatus.get(id))
  const isExecuting = useStudioStore((state) => state.isExecuting)
  const executionStatus: NodeExecutionStatus = executionState?.status || 'idle'

  // Get key spec info to display
  const specPreview = getSpecPreview(data.kind, data.spec)

  // Get status-based styling
  const statusStyles = isExecuting ? getStatusStyles(executionStatus) : { border: '', glow: '', animate: false }

  return (
    <div
      className={`
        relative rounded-lg min-w-[200px] max-w-[280px]
        bg-hex-surface border-2 transition-all duration-300
        ${selected ? 'border-hex-accent shadow-lg shadow-hex-accent/30 scale-[1.02]' : 'border-hex-border hover:border-hex-border/80'}
        ${!data.isValid ? 'border-hex-error' : ''}
        ${isExecuting && statusStyles.border ? statusStyles.border : ''}
        ${statusStyles.glow}
        ${statusStyles.animate ? 'animate-pulse' : ''}
      `}
      style={{
        borderTopColor: isExecuting && executionStatus === 'running' ? '#3b82f6' : color,
        borderTopWidth: 3,
      }}
    >
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Left}
        className="!w-3 !h-3 !bg-hex-accent !border-2 !border-hex-surface !-left-1.5"
      />

      {/* Header */}
      <div className="px-3 py-2 flex items-center gap-2 border-b border-hex-border/50">
        <div
          className="w-7 h-7 rounded-md flex items-center justify-center flex-shrink-0"
          style={{ backgroundColor: `${color}15` }}
        >
          <Icon size={14} style={{ color }} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-xs font-semibold text-hex-text truncate">
            {data.label}
          </div>
          <div className="text-[10px] text-hex-text-muted truncate">
            {template?.label || data.kind.replace('_node', '')}
          </div>
        </div>
        {!data.isValid && (
          <AlertCircle size={14} className="text-hex-error flex-shrink-0" />
        )}
      </div>

      {/* Spec Preview */}
      {specPreview && (
        <div className="px-3 py-2 space-y-1">
          {specPreview.map((item, i) => (
            <div key={i} className="flex items-start gap-2 text-[10px]">
              <span className="text-hex-text-muted flex-shrink-0 w-16 truncate">
                {item.key}:
              </span>
              <span className="text-hex-text truncate font-mono">
                {item.value}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Footer with status */}
      <div className={`px-3 py-1.5 rounded-b-md flex items-center justify-between transition-colors duration-300 ${
        executionStatus === 'running' ? 'bg-hex-accent/10' :
        executionStatus === 'completed' ? 'bg-hex-success/10' :
        executionStatus === 'failed' ? 'bg-hex-error/10' :
        'bg-hex-bg/50'
      }`}>
        <div className="flex items-center gap-1">
          {executionStatus === 'running' ? (
            <>
              <Loader2 size={10} className="text-hex-accent animate-spin" />
              <span className="text-[9px] text-hex-accent uppercase tracking-wider font-medium">
                Running
              </span>
            </>
          ) : executionStatus === 'completed' ? (
            <>
              <CheckCircle size={10} className="text-hex-success" />
              <span className="text-[9px] text-hex-success uppercase tracking-wider">
                Done
              </span>
              {executionState?.duration_ms != null && (
                <span className="text-[8px] text-hex-success/70 ml-1">
                  {executionState.duration_ms.toFixed(0)}ms
                </span>
              )}
            </>
          ) : executionStatus === 'failed' ? (
            <>
              <XCircle size={10} className="text-hex-error" />
              <span className="text-[9px] text-hex-error uppercase tracking-wider">
                Failed
              </span>
            </>
          ) : executionStatus === 'pending' ? (
            <>
              <Clock size={10} className="text-hex-warning/70" />
              <span className="text-[9px] text-hex-warning/70 uppercase tracking-wider">
                Pending
              </span>
            </>
          ) : (
            <>
              <Play size={10} className="text-hex-text-muted" />
              <span className="text-[9px] text-hex-text-muted uppercase tracking-wider">
                Ready
              </span>
            </>
          )}
        </div>
        {data.errors.length > 0 && (
          <span className="text-[9px] text-hex-error">
            {data.errors.length} error{data.errors.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        className="!w-3 !h-3 !bg-hex-accent !border-2 !border-hex-surface !-right-1.5"
      />

      {/* Selection indicator */}
      {selected && (
        <div className="absolute -inset-1 border-2 border-hex-accent/30 rounded-xl pointer-events-none" />
      )}
    </div>
  )
}

// Extract key info to show in node preview
function getSpecPreview(kind: string, spec: Record<string, unknown>): Array<{ key: string; value: string }> | null {
  const preview: Array<{ key: string; value: string }> = []

  switch (kind) {
    case 'function_node':
      if (spec.fn) preview.push({ key: 'fn', value: formatValuePreview(spec.fn, 30) })
      break
    case 'llm_node':
      if (spec.prompt_template) preview.push({ key: 'prompt', value: formatValuePreview(spec.prompt_template, 25) })
      if (spec.system_prompt) preview.push({ key: 'system', value: formatValuePreview(spec.system_prompt, 20) })
      break
    case 're_act_agent_node':
    case 'agent_node':
      if (spec.main_prompt) preview.push({ key: 'prompt', value: formatValuePreview(spec.main_prompt, 25) })
      // Check for config.max_steps
      if (spec.config && typeof spec.config === 'object') {
        const config = spec.config as Record<string, unknown>
        if (config.max_steps) preview.push({ key: 'max_steps', value: formatValuePreview(config.max_steps) })
      }
      break
    case 'composite_node':
      if (spec.mode) preview.push({ key: 'mode', value: formatValuePreview(spec.mode) })
      if (spec.condition) preview.push({ key: 'condition', value: formatValuePreview(spec.condition, 20) })
      if (spec.items) preview.push({ key: 'items', value: formatValuePreview(spec.items, 20) })
      break
    case 'conditional_node':
      if (spec.condition) preview.push({ key: 'if', value: formatValuePreview(spec.condition, 25) })
      break
    case 'loop_node':
      if (spec.while_condition) preview.push({ key: 'while', value: formatValuePreview(spec.while_condition, 20) })
      if (spec.max_iterations) preview.push({ key: 'max', value: formatValuePreview(spec.max_iterations) })
      break
    case 'expression_node':
      if (spec.expressions && typeof spec.expressions === 'object') {
        const keys = Object.keys(spec.expressions)
        if (keys.length > 0) preview.push({ key: 'expr', value: formatValuePreview(keys.join(', '), 25) })
      }
      if (spec.merge_strategy) preview.push({ key: 'merge', value: formatValuePreview(spec.merge_strategy) })
      break
    case 'data_node':
      if (spec.output && typeof spec.output === 'object') {
        const keys = Object.keys(spec.output)
        if (keys.length > 0) preview.push({ key: 'output', value: formatValuePreview(keys.join(', '), 25) })
      }
      break
    case 'tool_call_node':
      if (spec.tool_name) preview.push({ key: 'tool', value: formatValuePreview(spec.tool_name, 25) })
      break
    case 'port_call_node':
      if (spec.port) preview.push({ key: 'port', value: formatValuePreview(spec.port) })
      if (spec.method) preview.push({ key: 'method', value: formatValuePreview(spec.method) })
      break
    default:
      // Show first 2 spec items - use formatValuePreview to handle objects properly
      const entries = Object.entries(spec).slice(0, 2)
      entries.forEach(([key, value]) => {
        preview.push({ key, value: formatValuePreview(value, 20) })
      })
  }

  return preview.length > 0 ? preview : null
}

export default memo(HexdagNode)
