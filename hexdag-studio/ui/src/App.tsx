import { useEffect, useState, useCallback } from 'react'
import { ReactFlowProvider } from '@xyflow/react'
import Header from './components/Header'
import FileBrowser from './components/FileBrowser'
import NodePalette from './components/NodePalette'
import Canvas from './components/Canvas'
import YamlEditor from './components/YamlEditor'
import ValidationPanel from './components/ValidationPanel'
import NodeInspector from './components/NodeInspector'
import ExecutionPanel from './components/ExecutionPanel'
import EnvironmentEditor from './components/EnvironmentEditor'
import ResizablePanel, { ResizableHorizontal } from './components/ResizablePanel'
import { useStudioStore } from './lib/store'
import { saveFile, initializeNodeRegistry } from './lib/api'

type ViewMode = 'split' | 'canvas' | 'yaml'
type RightPanel = 'validation' | 'inspector' | 'environments' | 'execute'

export default function App() {
  console.log('[App] Rendering App component')
  const [viewMode, setViewMode] = useState<ViewMode>('split')
  const [rightPanel, setRightPanel] = useState<RightPanel>('validation')
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)

  const { currentFile, yamlContent, setIsSaving, setIsDirty, nodes } = useStudioStore()

  // Initialize node registry (fetch schemas for core + plugin nodes) on mount
  useEffect(() => {
    console.log('[App] useEffect running - calling initializeNodeRegistry')
    initializeNodeRegistry().then(() => {
      console.log('[App] initializeNodeRegistry completed')
    }).catch((err) => {
      console.error('[App] initializeNodeRegistry failed:', err)
    })
  }, [])

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
      <div className="h-screen flex flex-col bg-hex-bg select-none">
        {/* Header */}
        <Header />

        {/* Main content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Left sidebar: Files + Node palette (resizable) */}
          <ResizablePanel
            defaultWidth={220}
            minWidth={160}
            maxWidth={400}
            side="left"
            className="flex flex-col bg-hex-surface border-r border-hex-border"
          >
            <FileBrowser />
            {/* Node Palette - no more tabs, just nodes */}
            <div className="flex-1 overflow-hidden">
              <NodePalette />
            </div>
          </ResizablePanel>

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
            <div className="flex-1 overflow-hidden">
              {viewMode === 'split' ? (
                <ResizableHorizontal defaultSplit={50} minLeft={300} minRight={300}>
                  <div className="h-full">
                    <Canvas onNodeSelect={handleNodeSelect} selectedNodeId={selectedNodeId} />
                  </div>
                  <div className="h-full flex flex-col">
                    <YamlEditor />
                  </div>
                </ResizableHorizontal>
              ) : viewMode === 'canvas' ? (
                <Canvas onNodeSelect={handleNodeSelect} selectedNodeId={selectedNodeId} />
              ) : (
                <div className="h-full flex flex-col">
                  <YamlEditor />
                </div>
              )}
            </div>
          </div>

          {/* Right sidebar: Validation / Inspector / Execute (resizable) */}
          <ResizablePanel
            defaultWidth={288}
            minWidth={200}
            maxWidth={500}
            side="right"
            className="bg-hex-surface border-l border-hex-border flex flex-col"
          >
            {/* Tab switcher */}
            <div className="h-8 border-b border-hex-border flex items-center px-1 gap-0.5">
              <button
                onClick={() => setRightPanel('validation')}
                className={`flex-1 py-1 text-[9px] font-semibold uppercase tracking-wider rounded transition-colors ${
                  rightPanel === 'validation'
                    ? 'bg-hex-accent/20 text-hex-accent'
                    : 'text-hex-text-muted hover:text-hex-text'
                }`}
              >
                Validate
              </button>
              <button
                onClick={() => setRightPanel('inspector')}
                className={`flex-1 py-1 text-[9px] font-semibold uppercase tracking-wider rounded transition-colors ${
                  rightPanel === 'inspector'
                    ? 'bg-hex-accent/20 text-hex-accent'
                    : 'text-hex-text-muted hover:text-hex-text'
                }`}
              >
                Inspect
                {selectedNodeId && (
                  <span className="ml-1 w-1.5 h-1.5 bg-hex-accent rounded-full inline-block" />
                )}
              </button>
              <button
                onClick={() => setRightPanel('environments')}
                className={`flex-1 py-1 text-[9px] font-semibold uppercase tracking-wider rounded transition-colors ${
                  rightPanel === 'environments'
                    ? 'bg-hex-warning/20 text-hex-warning'
                    : 'text-hex-text-muted hover:text-hex-text'
                }`}
              >
                Envs
              </button>
              <button
                onClick={() => setRightPanel('execute')}
                className={`flex-1 py-1 text-[9px] font-semibold uppercase tracking-wider rounded transition-colors ${
                  rightPanel === 'execute'
                    ? 'bg-hex-success/20 text-hex-success'
                    : 'text-hex-text-muted hover:text-hex-text'
                }`}
              >
                Run
              </button>
            </div>

            {/* Panel content */}
            <div className="flex-1 overflow-hidden">
              {rightPanel === 'validation' ? (
                <ValidationPanel />
              ) : rightPanel === 'execute' ? (
                <ExecutionPanel />
              ) : rightPanel === 'environments' ? (
                <EnvironmentEditor />
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
          </ResizablePanel>
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
