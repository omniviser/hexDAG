import { useState } from 'react'
import { X, Trash2, Copy, Code, ChevronDown, ChevronRight, Plus, FileCode, ExternalLink, Plug, Info } from 'lucide-react'
import { useStudioStore } from '../lib/store'
import { getNodeTemplate, nodeTemplates, getNodeConfigSchema } from '../lib/nodeTemplates'
import { formatValueForInput, parseInputValue, generateUniqueName } from '../lib/formatValue'
import PythonEditor from './PythonEditor'
import NodePortsSection from './NodePortsSection'
import type { HexdagNode, HexdagEdge, NodeTemplate } from '../types'

interface NodeInspectorProps {
  nodeId: string | null
  onClose: () => void
}

export default function NodeInspector({ nodeId, onClose }: NodeInspectorProps) {
  const { nodes, setNodes, edges, setEdges, syncCanvasToYaml } = useStudioStore()
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['general', 'spec']))

  const node = nodes.find((n: HexdagNode) => n.id === nodeId)
  const template = node ? getNodeTemplate(node.data.kind) : null

  // Show helpful prompt when no node is selected
  if (!node || !nodeId) {
    return (
      <div className="h-full flex flex-col bg-hex-surface">
        <div className="p-3 border-b border-hex-border">
          <h3 className="text-sm font-medium text-hex-text">Node Inspector</h3>
          <p className="text-[10px] text-hex-text-muted">Select a node to edit its configuration</p>
        </div>
        <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
          <div className="w-12 h-12 rounded-full bg-hex-border/30 flex items-center justify-center mb-3">
            <Code size={20} className="text-hex-text-muted" />
          </div>
          <p className="text-xs text-hex-text-muted mb-1">No node selected</p>
          <p className="text-[10px] text-hex-text-muted">
            Click on a node in the canvas to inspect and edit it
          </p>
          <div className="mt-4 text-[10px] text-hex-text-muted space-y-1">
            <p className="flex items-center gap-1 justify-center">
              <span className="text-hex-accent">Tip:</span> Drag nodes from the palette on the left
            </p>
            <p className="flex items-center gap-1 justify-center">
              <span className="text-hex-accent">Tip:</span> Connect nodes by dragging between handles
            </p>
          </div>
        </div>
      </div>
    )
  }

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections)
    if (newExpanded.has(section)) {
      newExpanded.delete(section)
    } else {
      newExpanded.add(section)
    }
    setExpandedSections(newExpanded)
  }

  const updateNode = (updates: Partial<HexdagNode['data']>) => {
    setNodes(
      nodes.map((n: HexdagNode) =>
        n.id === nodeId
          ? { ...n, data: { ...n.data, ...updates } }
          : n
      )
    )
    syncCanvasToYaml()
  }

  const updateSpec = (key: string, value: unknown) => {
    const newSpec = { ...node.data.spec, [key]: value }
    updateNode({ spec: newSpec })
  }

  const deleteSpec = (key: string) => {
    const newSpec = { ...node.data.spec }
    delete newSpec[key]
    updateNode({ spec: newSpec })
  }

  const renameNode = (newName: string) => {
    if (!newName || newName === nodeId) return

    // Check for duplicates
    if (nodes.some((n: HexdagNode) => n.id === newName)) {
      alert('A node with this name already exists')
      return
    }

    // Update node id
    setNodes(
      nodes.map((n: HexdagNode) =>
        n.id === nodeId
          ? { ...n, id: newName, data: { ...n.data, label: newName } }
          : n
      )
    )

    // Update edges
    setEdges(
      edges.map((e: HexdagEdge) => ({
        ...e,
        id: e.id.replace(nodeId, newName),
        source: e.source === nodeId ? newName : e.source,
        target: e.target === nodeId ? newName : e.target,
      }))
    )

    syncCanvasToYaml()
  }

  const changeNodeKind = (newKind: string) => {
    const newTemplate = getNodeTemplate(newKind)
    if (!newTemplate) return

    updateNode({
      kind: newKind,
      spec: { ...newTemplate.defaultSpec },
    })
  }

  const deleteNode = () => {
    if (!confirm(`Delete node "${nodeId}"?`)) return

    setNodes(nodes.filter((n: HexdagNode) => n.id !== nodeId))
    setEdges(edges.filter((e: HexdagEdge) => e.source !== nodeId && e.target !== nodeId))
    syncCanvasToYaml()
    onClose()
  }

  const duplicateNode = () => {
    const existingNames = nodes.map((n: HexdagNode) => n.id)
    const newName = generateUniqueName(nodeId, existingNames, '_copy')

    const newNode: HexdagNode = {
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
    }

    setNodes([...nodes, newNode])
    syncCanvasToYaml()
  }

  // Get dependencies (incoming edges)
  const dependencies = edges.filter((e: HexdagEdge) => e.target === nodeId).map((e: HexdagEdge) => e.source)

  return (
    <div className="h-full flex flex-col bg-hex-surface">
      {/* Header */}
      <div className="p-3 border-b border-hex-border flex items-center gap-2">
        <div
          className="w-8 h-8 rounded flex items-center justify-center flex-shrink-0"
          style={{ backgroundColor: `${template?.color || '#6b7280'}20` }}
        >
          <Code size={14} style={{ color: template?.color || '#6b7280' }} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-hex-text truncate">{nodeId}</div>
          <div className="text-[10px] text-hex-text-muted">{node.data.kind}</div>
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-hex-border/50 rounded transition-colors"
        >
          <X size={14} className="text-hex-text-muted" />
        </button>
      </div>

      {/* Actions */}
      <div className="p-2 border-b border-hex-border flex gap-1">
        <button
          onClick={duplicateNode}
          className="flex-1 flex items-center justify-center gap-1 py-1.5 text-xs rounded bg-hex-bg hover:bg-hex-border/50 transition-colors"
        >
          <Copy size={12} />
          Duplicate
        </button>
        <button
          onClick={deleteNode}
          className="flex-1 flex items-center justify-center gap-1 py-1.5 text-xs rounded bg-hex-error/10 text-hex-error hover:bg-hex-error/20 transition-colors"
        >
          <Trash2 size={12} />
          Delete
        </button>
      </div>

      {/* Sections */}
      <div className="flex-1 overflow-y-auto">
        {/* General Section */}
        <Section
          title="General"
          expanded={expandedSections.has('general')}
          onToggle={() => toggleSection('general')}
        >
          <Field label="Name">
            <input
              type="text"
              defaultValue={nodeId}
              onBlur={(e) => renameNode(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
              className="w-full bg-hex-bg border border-hex-border rounded px-2 py-1.5 text-xs text-hex-text focus:border-hex-accent focus:outline-none"
            />
          </Field>

          <Field label="Type">
            <select
              value={node.data.kind}
              onChange={(e) => changeNodeKind(e.target.value)}
              className="w-full bg-hex-bg border border-hex-border rounded px-2 py-1.5 text-xs text-hex-text focus:border-hex-accent focus:outline-none"
            >
              {nodeTemplates.map((t: NodeTemplate) => (
                <option key={t.kind} value={t.kind}>
                  {t.label}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Dependencies">
            <div className="space-y-1">
              {dependencies.length === 0 ? (
                <div className="text-[10px] text-hex-text-muted italic">No dependencies</div>
              ) : (
                dependencies.map((dep: string) => (
                  <div
                    key={dep}
                    className="flex items-center gap-2 text-xs bg-hex-bg rounded px-2 py-1"
                  >
                    <div className="w-2 h-2 bg-hex-accent rounded-full" />
                    {dep}
                  </div>
                ))
              )}
            </div>
          </Field>
        </Section>

        {/* Spec Section */}
        <Section
          title="Configuration"
          expanded={expandedSections.has('spec')}
          onToggle={() => toggleSection('spec')}
        >
          <SchemaBasedConfig
            nodeKind={node.data.kind}
            spec={node.data.spec}
            onUpdateSpec={updateSpec}
            onDeleteSpec={deleteSpec}
          />
        </Section>

        {/* Ports Section - only show for nodes that require ports */}
        {template?.requiredPorts && template.requiredPorts.length > 0 && (
          <Section
            title="Ports"
            expanded={expandedSections.has('ports')}
            onToggle={() => toggleSection('ports')}
            icon={<Plug size={12} />}
          >
            <NodePortsSection
              nodeId={nodeId}
              requiredPorts={template.requiredPorts}
              nodePorts={(node.data.spec.ports as Record<string, { adapter: string; config: Record<string, unknown> } | undefined>) || {}}
              onPortsChange={(newPorts) => {
                // Clean up empty ports
                const cleanedPorts: Record<string, unknown> = {}
                for (const [key, value] of Object.entries(newPorts)) {
                  if (value && value.adapter) {
                    cleanedPorts[key] = value
                  }
                }
                if (Object.keys(cleanedPorts).length > 0) {
                  updateSpec('ports', cleanedPorts)
                } else {
                  deleteSpec('ports')
                }
              }}
            />
          </Section>
        )}

        {/* Python Code Editor Section - only for function_node */}
        {node.data.kind === 'function_node' && (
          <Section
            title="Python Code"
            expanded={expandedSections.has('code')}
            onToggle={() => toggleSection('code')}
            icon={<FileCode size={12} />}
          >
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-[10px] font-medium text-hex-text-muted uppercase tracking-wider">
                  Inline Code
                </label>
                <span className="text-[10px] text-hex-text-muted">
                  Python 3.12+
                </span>
              </div>
              <PythonEditor
                value={String(node.data.spec.code || node.data.spec.inline_code || getDefaultPythonCode())}
                onChange={(code) => updateSpec('code', code)}
                height="200px"
              />
              <div className="text-[10px] text-hex-text-muted space-y-1">
                <p>Define a function that processes the input data.</p>
                <p className="text-hex-accent">
                  Access inputs via <code className="bg-hex-bg px-1 rounded">ctx</code> parameter.
                </p>
              </div>
              {typeof node.data.spec.fn === 'string' && node.data.spec.fn && (
                <div className="flex items-center gap-2 p-2 bg-hex-bg rounded">
                  <ExternalLink size={12} className="text-hex-text-muted" />
                  <span className="text-[10px] text-hex-text-muted">
                    Module path: <code className="text-hex-accent">{node.data.spec.fn}</code>
                  </span>
                </div>
              )}
            </div>
          </Section>
        )}

        {/* Template Info */}
        {template && (
          <Section
            title="Info"
            expanded={expandedSections.has('info')}
            onToggle={() => toggleSection('info')}
          >
            <div className="text-xs text-hex-text-muted">
              {template.description}
            </div>
          </Section>
        )}
      </div>
    </div>
  )
}

// Helper to get default Python code for new function nodes
function getDefaultPythonCode(): string {
  return `def process(ctx):
    """Process the input data.

    Args:
        ctx: Execution context with input data

    Returns:
        Processed result
    """
    # Access inputs from previous nodes
    # input_data = ctx.get("previous_node")

    return {"result": "processed"}
`
}

// Helper components
function Section({
  title,
  expanded,
  onToggle,
  children,
  icon,
}: {
  title: string
  expanded: boolean
  onToggle: () => void
  children: React.ReactNode
  icon?: React.ReactNode
}) {
  return (
    <div className="border-b border-hex-border">
      <button
        onClick={onToggle}
        className="w-full px-3 py-2 flex items-center gap-2 text-xs font-medium text-hex-text-muted hover:bg-hex-border/30 transition-colors"
      >
        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {icon}
        {title}
      </button>
      {expanded && <div className="px-3 pb-3 space-y-3">{children}</div>}
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[10px] font-medium text-hex-text-muted mb-1 uppercase tracking-wider">
        {label}
      </label>
      {children}
    </div>
  )
}

function AddSpecButton({
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
          placeholder="Key name"
          autoFocus
          className="flex-1 bg-hex-bg border border-hex-border rounded px-2 py-1 text-xs text-hex-text focus:border-hex-accent focus:outline-none"
        />
        <button
          onClick={handleAdd}
          className="px-2 py-1 text-xs bg-hex-accent text-white rounded hover:bg-hex-accent-hover"
        >
          Add
        </button>
        <button
          onClick={() => setIsAdding(false)}
          className="px-2 py-1 text-xs bg-hex-border text-hex-text rounded hover:bg-hex-border/70"
        >
          Cancel
        </button>
      </div>
    )
  }

  return (
    <button
      onClick={() => setIsAdding(true)}
      className="w-full flex items-center justify-center gap-1 py-1.5 text-xs text-hex-text-muted hover:text-hex-text border border-dashed border-hex-border rounded hover:border-hex-accent transition-colors"
    >
      <Plus size={12} />
      Add property
    </button>
  )
}

/**
 * Schema-based configuration component that shows fields based on node schema
 * when available, with descriptions and proper input types.
 */
function SchemaBasedConfig({
  nodeKind,
  spec,
  onUpdateSpec,
  onDeleteSpec,
}: {
  nodeKind: string
  spec: Record<string, unknown>
  onUpdateSpec: (key: string, value: unknown) => void
  onDeleteSpec: (key: string) => void
}) {
  const configSchema = getNodeConfigSchema(nodeKind)
  const schemaProperties = configSchema?.properties as Record<string, {
    type?: string
    description?: string
    default?: unknown
    enum?: unknown[]
  }> | undefined

  // Get all keys: both from schema and from current spec
  const schemaKeys = schemaProperties ? Object.keys(schemaProperties) : []
  const specKeys = Object.keys(spec).filter(
    (key) => key !== 'code' && key !== 'inline_code' && key !== 'ports' && key !== 'name' && key !== 'deps'
  )
  const allKeys = [...new Set([...schemaKeys, ...specKeys])]

  // Filter out internal keys that shouldn't be shown
  const displayKeys = allKeys.filter((key) => !['self', 'kwargs'].includes(key))

  if (displayKeys.length === 0 && !schemaProperties) {
    return (
      <>
        <div className="text-[10px] text-hex-text-muted italic py-2">
          No configuration options available
        </div>
        <AddSpecButton
          onAdd={(key) => onUpdateSpec(key, '')}
          existingKeys={Object.keys(spec)}
        />
      </>
    )
  }

  return (
    <>
      {displayKeys.map((key) => {
        const schemaProp = schemaProperties?.[key]
        const value = spec[key]
        const hasValue = key in spec

        return (
          <SpecFieldWithSchema
            key={key}
            name={key}
            value={value}
            schema={schemaProp}
            hasValue={hasValue}
            onChange={(v) => onUpdateSpec(key, v)}
            onDelete={() => onDeleteSpec(key)}
          />
        )
      })}
      <AddSpecButton
        onAdd={(key) => onUpdateSpec(key, '')}
        existingKeys={Object.keys(spec)}
      />
    </>
  )
}

/**
 * A spec field that uses schema information for better UX
 */
function SpecFieldWithSchema({
  name,
  value,
  schema,
  hasValue,
  onChange,
  onDelete,
}: {
  name: string
  value: unknown
  schema?: {
    type?: string
    description?: string
    default?: unknown
    enum?: unknown[]
  }
  hasValue: boolean
  onChange: (value: unknown) => void
  onDelete: () => void
}) {
  // If field is from schema but not in spec, show it as optional/placeholder
  const isOptional = schema && !hasValue
  const displayValue = hasValue ? value : schema?.default

  // For enum types, render a dropdown
  if (schema?.enum && schema.enum.length > 0) {
    return (
      <div className="group">
        <div className="flex items-center justify-between mb-1">
          <label className="text-[10px] font-medium text-hex-text-muted uppercase tracking-wider flex items-center gap-1">
            {name}
            {schema.description && (
              <span title={schema.description} className="cursor-help">
                <Info size={10} className="text-hex-text-muted opacity-50" />
              </span>
            )}
          </label>
          {hasValue && (
            <button
              onClick={onDelete}
              className="opacity-0 group-hover:opacity-100 p-0.5 hover:bg-hex-error/20 rounded transition-all"
            >
              <X size={10} className="text-hex-error" />
            </button>
          )}
        </div>
        <select
          value={String(displayValue ?? '')}
          onChange={(e) => onChange(e.target.value)}
          className={`w-full bg-hex-bg border border-hex-border rounded px-2 py-1.5 text-xs text-hex-text focus:border-hex-accent focus:outline-none ${isOptional ? 'opacity-60' : ''}`}
        >
          {isOptional && <option value="">-- Select --</option>}
          {schema.enum.map((opt) => (
            <option key={String(opt)} value={String(opt)}>
              {String(opt)}
            </option>
          ))}
        </select>
        {schema.description && (
          <div className="text-[9px] text-hex-text-muted mt-0.5">{schema.description}</div>
        )}
      </div>
    )
  }

  // For values already set, use the standard SpecField
  if (hasValue) {
    return (
      <div className="group">
        <div className="flex items-center justify-between mb-1">
          <label className="text-[10px] font-medium text-hex-text-muted uppercase tracking-wider flex items-center gap-1">
            {name}
            {schema?.description && (
              <span title={schema.description} className="cursor-help">
                <Info size={10} className="text-hex-text-muted opacity-50" />
              </span>
            )}
          </label>
          <button
            onClick={onDelete}
            className="opacity-0 group-hover:opacity-100 p-0.5 hover:bg-hex-error/20 rounded transition-all"
          >
            <X size={10} className="text-hex-error" />
          </button>
        </div>
        <SpecFieldInput value={value} onChange={onChange} />
        {schema?.description && (
          <div className="text-[9px] text-hex-text-muted mt-0.5">{schema.description}</div>
        )}
      </div>
    )
  }

  // Schema-only field (not yet set) - show as add button
  return (
    <button
      onClick={() => onChange(schema?.default ?? '')}
      className="w-full flex items-center gap-2 py-1.5 px-2 text-xs text-hex-text-muted hover:text-hex-text border border-dashed border-hex-border rounded hover:border-hex-accent transition-colors group"
    >
      <Plus size={10} />
      <span className="flex-1 text-left">{name}</span>
      {schema?.description && (
        <span title={schema.description} className="cursor-help">
          <Info size={10} className="text-hex-text-muted opacity-50" />
        </span>
      )}
    </button>
  )
}

/**
 * Input component for spec values - handles different types
 */
function SpecFieldInput({
  value,
  onChange,
}: {
  value: unknown
  onChange: (value: unknown) => void
}) {
  const displayValue = formatValueForInput(value)
  const isMultiline = typeof value === 'string' && (value.includes('\n') || value.length > 50)
  const isObject = typeof value === 'object' && value !== null

  if (isObject) {
    return (
      <textarea
        value={displayValue}
        onChange={(e) => {
          const parsed = parseInputValue(e.target.value)
          onChange(parsed)
        }}
        rows={4}
        className="w-full bg-hex-bg border border-hex-border rounded px-2 py-1.5 text-xs text-hex-text focus:border-hex-accent focus:outline-none font-mono resize-y"
      />
    )
  }

  if (typeof value === 'boolean') {
    return (
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={value}
          onChange={(e) => onChange(e.target.checked)}
          className="w-4 h-4 rounded border-hex-border bg-hex-bg text-hex-accent focus:ring-hex-accent"
        />
        <span className="text-xs text-hex-text">{value ? 'true' : 'false'}</span>
      </label>
    )
  }

  if (typeof value === 'number') {
    return (
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full bg-hex-bg border border-hex-border rounded px-2 py-1.5 text-xs text-hex-text focus:border-hex-accent focus:outline-none"
      />
    )
  }

  if (isMultiline) {
    return (
      <textarea
        value={displayValue}
        onChange={(e) => onChange(e.target.value)}
        rows={4}
        className="w-full bg-hex-bg border border-hex-border rounded px-2 py-1.5 text-xs text-hex-text focus:border-hex-accent focus:outline-none font-mono resize-y"
      />
    )
  }

  return (
    <input
      type="text"
      value={displayValue}
      onChange={(e) => onChange(e.target.value)}
      className="w-full bg-hex-bg border border-hex-border rounded px-2 py-1.5 text-xs text-hex-text focus:border-hex-accent focus:outline-none font-mono"
    />
  )
}
