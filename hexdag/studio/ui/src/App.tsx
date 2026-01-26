import { useEffect, useState, useCallback } from 'react'
import { ReactFlowProvider } from '@xyflow/react'
import Header from './components/Header'
import NodePalette from './components/NodePalette'
import Canvas from './components/Canvas'
import YamlEditor from './components/YamlEditor'
import ValidationPanel from './components/ValidationPanel'
import NodeInspector from './components/NodeInspector'
import { useStudioStore } from './lib/store'
import { saveFile } from './lib/api'

type ViewMode = 'split' | 'canvas' | 'yaml'
type RightPanel = 'validation' | 'inspector'

export default function App() {
  const [viewMode, setViewMode] = useState<ViewMode>('split')
  const [rightPanel, setRightPanel] = useState<RightPanel>('validation')
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)

  const { currentFile, yamlContent, setIsSaving, setIsDirty, nodes } = useStudioStore()

  // Handle node selection from canvas
  const handleNodeSelect = useCallback((nodeId: string | null) => {
    setSelectedNodeId(nodeId)
    if (nodeId) {
      setRightPanel('inspector')
    }
  }, [])

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd/Ctrl + S to save
      if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault()
        handleSave()
      }
      // Cmd/Ctrl + 1/2/3 to switch views
      if ((e.metaKey || e.ctrlKey) && e.key === '1') {
        e.preventDefault()
        setViewMode('split')
      }
      if ((e.metaKey || e.ctrlKey) && e.key === '2') {
        e.preventDefault()
        setViewMode('canvas')
      }
      if ((e.metaKey || e.ctrlKey) && e.key === '3') {
        e.preventDefault()
        setViewMode('yaml')
      }
      // Escape to deselect
      if (e.key === 'Escape') {
        setSelectedNodeId(null)
        setRightPanel('validation')
      }
      // Delete selected node
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedNodeId) {
        // Only if not in an input
        const target = e.target as HTMLElement
        if (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA') {
          e.preventDefault()
          // Delete will be handled by Canvas component
        }
      }
    }

    const handleSaveEvent = () => {
      handleSave()
    }

    // Listen for node selection events from Canvas
    const handleNodeSelectEvent = (e: CustomEvent<{ nodeId: string | null }>) => {
      handleNodeSelect(e.detail.nodeId)
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('hexdag-save', handleSaveEvent)
    window.addEventListener('hexdag-node-select', handleNodeSelectEvent as EventListener)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('hexdag-save', handleSaveEvent)
      window.removeEventListener('hexdag-node-select', handleNodeSelectEvent as EventListener)
    }
  }, [currentFile, yamlContent, selectedNodeId, handleNodeSelect])

  const handleSave = async () => {
    if (!currentFile || !yamlContent) return

    setIsSaving(true)
    try {
      await saveFile(currentFile, yamlContent)
      setIsDirty(false)
    } catch (error) {
      console.error('Failed to save:', error)
    } finally {
      setIsSaving(false)
    }
  }

  return (
    <ReactFlowProvider>
      <div className="h-screen flex flex-col bg-hex-bg">
        {/* Header */}
        <Header />

        {/* Main content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Node palette */}
          <div className="w-48 flex-shrink-0">
            <NodePalette />
          </div>

          {/* Main editor area */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* View mode tabs */}
            <div className="h-8 bg-hex-surface border-b border-hex-border flex items-center px-2 gap-1">
              <button
                onClick={() => setViewMode('split')}
                className={`px-3 py-1 text-xs rounded transition-colors ${
                  viewMode === 'split'
                    ? 'bg-hex-accent text-white'
                    : 'text-hex-text-muted hover:text-hex-text hover:bg-hex-border/50'
                }`}
              >
                Split
              </button>
              <button
                onClick={() => setViewMode('canvas')}
                className={`px-3 py-1 text-xs rounded transition-colors ${
                  viewMode === 'canvas'
                    ? 'bg-hex-accent text-white'
                    : 'text-hex-text-muted hover:text-hex-text hover:bg-hex-border/50'
                }`}
              >
                Canvas
              </button>
              <button
                onClick={() => setViewMode('yaml')}
                className={`px-3 py-1 text-xs rounded transition-colors ${
                  viewMode === 'yaml'
                    ? 'bg-hex-accent text-white'
                    : 'text-hex-text-muted hover:text-hex-text hover:bg-hex-border/50'
                }`}
              >
                YAML
              </button>

              <div className="ml-auto flex items-center gap-2">
                {selectedNodeId && (
                  <span className="text-[10px] text-hex-accent">
                    Selected: {selectedNodeId}
                  </span>
                )}
                <span className="text-[10px] text-hex-text-muted">
                  {nodes.length} node{nodes.length !== 1 ? 's' : ''}
                </span>
              </div>
            </div>

            {/* Editor panels */}
            <div className="flex-1 flex overflow-hidden">
              {/* Canvas panel */}
              {(viewMode === 'split' || viewMode === 'canvas') && (
                <div className={`${viewMode === 'split' ? 'w-1/2' : 'flex-1'} border-r border-hex-border`}>
                  <Canvas onNodeSelect={handleNodeSelect} selectedNodeId={selectedNodeId} />
                </div>
              )}

              {/* YAML editor panel */}
              {(viewMode === 'split' || viewMode === 'yaml') && (
                <div className={`${viewMode === 'split' ? 'w-1/2' : 'flex-1'} flex flex-col`}>
                  <div className="flex-1">
                    <YamlEditor />
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right sidebar: Validation / Inspector */}
          <div className="w-72 flex-shrink-0 bg-hex-surface border-l border-hex-border flex flex-col">
            {/* Tab switcher */}
            <div className="h-8 border-b border-hex-border flex items-center px-1 gap-1">
              <button
                onClick={() => setRightPanel('validation')}
                className={`flex-1 py-1 text-[10px] font-semibold uppercase tracking-wider rounded transition-colors ${
                  rightPanel === 'validation'
                    ? 'bg-hex-accent/20 text-hex-accent'
                    : 'text-hex-text-muted hover:text-hex-text'
                }`}
              >
                Validation
              </button>
              <button
                onClick={() => setRightPanel('inspector')}
                className={`flex-1 py-1 text-[10px] font-semibold uppercase tracking-wider rounded transition-colors ${
                  rightPanel === 'inspector'
                    ? 'bg-hex-accent/20 text-hex-accent'
                    : 'text-hex-text-muted hover:text-hex-text'
                }`}
              >
                Inspector
                {selectedNodeId && (
                  <span className="ml-1 w-2 h-2 bg-hex-accent rounded-full inline-block" />
                )}
              </button>
            </div>

            {/* Panel content */}
            <div className="flex-1 overflow-hidden">
              {rightPanel === 'validation' ? (
                <ValidationPanel />
              ) : (
                <NodeInspector
                  nodeId={selectedNodeId}
                  onClose={() => {
                    setSelectedNodeId(null)
                    setRightPanel('validation')
                  }}
                />
              )}
            </div>
          </div>
        </div>

        {/* Status bar */}
        <div className="h-6 bg-hex-bg border-t border-hex-border flex items-center px-3 text-[10px] text-hex-text-muted">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 bg-hex-success rounded-full" />
            <span>Connected</span>
          </div>
          <div className="mx-3">|</div>
          <span>hexdag studio v0.1.0</span>
          <div className="ml-auto flex items-center gap-3">
            <span>
              <kbd className="px-1 py-0.5 bg-hex-surface rounded border border-hex-border">Cmd+S</kbd> Save
            </span>
            <span>
              <kbd className="px-1 py-0.5 bg-hex-surface rounded border border-hex-border">Esc</kbd> Deselect
            </span>
          </div>
        </div>
      </div>
    </ReactFlowProvider>
  )
}
