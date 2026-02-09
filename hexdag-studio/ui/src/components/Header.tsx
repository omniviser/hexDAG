import { Save, Play, RefreshCw, Settings, Package, Plug } from 'lucide-react'
import { useStudioStore } from '../lib/store'
import { saveFile, executePipeline, downloadProject } from '../lib/api'
import { useState } from 'react'
import PluginManager from './PluginManager'

export default function Header() {
  const [isRunning, setIsRunning] = useState(false)
  const [isExporting, setIsExporting] = useState(false)
  const [isPluginManagerOpen, setIsPluginManagerOpen] = useState(false)

  const {
    currentFile,
    yamlContent,
    isDirty,
    isSaving,
    setIsSaving,
    setIsDirty,
  } = useStudioStore()

  const handleSave = async () => {
    if (!currentFile || !yamlContent) return

    setIsSaving(true)
    try {
      await saveFile(currentFile, yamlContent)
      setIsDirty(false)
    } catch (error) {
      console.error('Failed to save:', error)
      alert(`Failed to save: ${error}`)
    } finally {
      setIsSaving(false)
    }
  }

  const handleRun = async () => {
    if (!yamlContent) {
      alert('No content to run')
      return
    }

    setIsRunning(true)
    try {
      const result = await executePipeline(yamlContent, {}, true)
      if (result.success) {
        alert(`Pipeline executed successfully in ${result.duration_ms.toFixed(0)}ms`)
      } else {
        alert(`Pipeline failed: ${result.error}`)
      }
    } catch (error) {
      alert(`Execution error: ${error}`)
    } finally {
      setIsRunning(false)
    }
  }

  const handleExport = async () => {
    if (!yamlContent) {
      alert('No pipeline to export')
      return
    }

    setIsExporting(true)
    try {
      await downloadProject(yamlContent, undefined, true)
    } catch (error) {
      alert(`Export failed: ${error}`)
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <header className="h-12 bg-hex-bg border-b border-hex-border flex items-center px-4 gap-4">
      {/* Logo */}
      <div className="flex items-center gap-2">
        <div className="w-6 h-6 bg-hex-accent rounded flex items-center justify-center">
          <span className="text-white text-xs font-bold">H</span>
        </div>
        <span className="text-sm font-semibold">
          <span className="text-hex-accent">hexdag</span>
          <span className="text-hex-text">studio</span>
        </span>
      </div>

      {/* Current file */}
      <div className="flex-1 flex items-center gap-2 min-w-0">
        {currentFile && (
          <>
            <span className="text-xs text-hex-text-muted truncate">
              {currentFile}
            </span>
            {isDirty && (
              <span className="w-2 h-2 bg-hex-warning rounded-full" title="Unsaved changes" />
            )}
          </>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => useStudioStore.getState().syncYamlToCanvas()}
          className="p-2 rounded hover:bg-hex-border/50 transition-colors text-hex-text-muted hover:text-hex-text"
          title="Refresh canvas from YAML"
        >
          <RefreshCw size={16} />
        </button>

        <button
          onClick={handleRun}
          disabled={isRunning || !yamlContent}
          className={`
            flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium
            transition-colors
            ${isRunning || !yamlContent
              ? 'bg-hex-border/50 text-hex-text-muted cursor-not-allowed'
              : 'bg-hex-success/20 text-hex-success hover:bg-hex-success/30'}
          `}
        >
          <Play size={14} className={isRunning ? 'animate-pulse' : ''} />
          {isRunning ? 'Running...' : 'Test Run'}
        </button>

        <button
          onClick={handleSave}
          disabled={isSaving || !currentFile || !isDirty}
          className={`
            flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium
            transition-colors
            ${isSaving || !currentFile || !isDirty
              ? 'bg-hex-border/50 text-hex-text-muted cursor-not-allowed'
              : 'bg-hex-accent text-white hover:bg-hex-accent-hover'}
          `}
        >
          <Save size={14} className={isSaving ? 'animate-pulse' : ''} />
          {isSaving ? 'Saving...' : 'Save'}
        </button>

        <div className="w-px h-6 bg-hex-border" />

        <button
          onClick={handleExport}
          disabled={isExporting || !yamlContent}
          className={`
            flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium
            transition-colors
            ${isExporting || !yamlContent
              ? 'bg-hex-border/50 text-hex-text-muted cursor-not-allowed'
              : 'bg-purple-500/20 text-purple-400 hover:bg-purple-500/30'}
          `}
          title="Export as standalone Python project"
        >
          <Package size={14} className={isExporting ? 'animate-pulse' : ''} />
          {isExporting ? 'Exporting...' : 'Export Project'}
        </button>

        <button
          onClick={() => setIsPluginManagerOpen(true)}
          className="p-2 rounded hover:bg-hex-border/50 transition-colors text-hex-text-muted hover:text-hex-text"
          title="Plugin Manager"
        >
          <Plug size={16} />
        </button>

        <button
          className="p-2 rounded hover:bg-hex-border/50 transition-colors text-hex-text-muted hover:text-hex-text"
          title="Settings"
        >
          <Settings size={16} />
        </button>
      </div>

      {/* Plugin Manager Modal */}
      <PluginManager
        isOpen={isPluginManagerOpen}
        onClose={() => setIsPluginManagerOpen(false)}
      />
    </header>
  )
}
