import { memo } from 'react'
import { Handle, Position } from '@xyflow/react'
import {
  Code,
  Brain,
  Cpu,
  FileText,
  Scissors,
  Bot,
  GitBranch,
  Repeat,
  Box,
  AlertCircle,
  Play,
  FileInput,
  FileOutput,
  Table,
  Package,
} from 'lucide-react'
import type { HexdagNodeData } from '../types'
import { getNodeColor, getNodeTemplate } from '../lib/nodeTemplates'

const iconMap: Record<string, typeof Code> = {
  // Core/builtin nodes
  function_node: Code,
  llm_node: Brain,
  raw_llm_node: Cpu,
  prompt_node: FileText,
  parser_node: Scissors,
  agent_node: Bot,
  conditional_node: GitBranch,
  loop_node: Repeat,
}

// Dynamic icon selection for plugin nodes (used when not in iconMap)
function getPluginNodeIcon(kind: string): typeof Code {
  const kindLower = kind.toLowerCase()
  if (kindLower.includes('file_reader') || kindLower.includes('input')) return FileInput
  if (kindLower.includes('file_writer') || kindLower.includes('output')) return FileOutput
  if (kindLower.includes('transform') || kindLower.includes('pandas')) return Table
  if (kindLower.includes('llm') || kindLower.includes('openai')) return Brain
  if (kindLower.includes('database') || kindLower.includes('sql')) return Cpu
  return Package
}

interface HexdagNodeProps {
  data: HexdagNodeData
  selected?: boolean
}

function HexdagNode({ data, selected }: HexdagNodeProps) {
  // Use iconMap for core nodes, dynamic icon for plugin nodes
  const Icon = iconMap[data.kind] || getPluginNodeIcon(data.kind) || Box
  const color = getNodeColor(data.kind)
  const template = getNodeTemplate(data.kind)

  // Get key spec info to display
  const specPreview = getSpecPreview(data.kind, data.spec)

  return (
    <div
      className={`
        relative rounded-lg min-w-[200px] max-w-[280px]
        bg-hex-surface border-2 transition-all
        ${selected ? 'border-hex-accent shadow-lg shadow-hex-accent/30 scale-[1.02]' : 'border-hex-border hover:border-hex-border/80'}
        ${!data.isValid ? 'border-hex-error' : ''}
      `}
      style={{
        borderTopColor: color,
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
      <div className="px-3 py-1.5 bg-hex-bg/50 rounded-b-md flex items-center justify-between">
        <div className="flex items-center gap-1">
          <Play size={10} className="text-hex-text-muted" />
          <span className="text-[9px] text-hex-text-muted uppercase tracking-wider">
            Ready
          </span>
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
      if (spec.fn) preview.push({ key: 'fn', value: truncate(String(spec.fn), 30) })
      break
    case 'llm_node':
      if (spec.prompt_template) preview.push({ key: 'prompt', value: truncate(String(spec.prompt_template), 25) })
      break
    case 'raw_llm_node':
      if (spec.messages_key) preview.push({ key: 'messages', value: String(spec.messages_key) })
      break
    case 'prompt_node':
      if (spec.template) preview.push({ key: 'template', value: truncate(String(spec.template), 25) })
      break
    case 'parser_node':
      if (spec.parser_type) preview.push({ key: 'parser', value: String(spec.parser_type) })
      break
    case 'agent_node':
      if (spec.max_steps) preview.push({ key: 'max_steps', value: String(spec.max_steps) })
      break
    case 'conditional_node':
      if (spec.condition) preview.push({ key: 'if', value: truncate(String(spec.condition), 25) })
      break
    case 'loop_node':
      if (spec.items_key) preview.push({ key: 'items', value: String(spec.items_key) })
      if (spec.max_iterations) preview.push({ key: 'max', value: String(spec.max_iterations) })
      break
    default:
      // Show first 2 spec items
      const entries = Object.entries(spec).slice(0, 2)
      entries.forEach(([key, value]) => {
        preview.push({ key, value: truncate(String(value), 20) })
      })
  }

  return preview.length > 0 ? preview : null
}

function truncate(str: string, max: number): string {
  if (str.length <= max) return str
  return str.slice(0, max - 1) + '…'
}

export default memo(HexdagNode)
