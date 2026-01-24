import { useEffect, useState } from 'react'
import { Folder, FileText, ChevronRight, RefreshCw } from 'lucide-react'
import { useStudioStore } from '../lib/store'
import { listFiles, readFile } from '../lib/api'
import type { FileInfo } from '../types'

export default function FileBrowser() {
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set())
  const [isLoading, setIsLoading] = useState(false)

  const {
    files,
    setFiles,
    currentFile,
    setCurrentFile,
    setYamlContent,
    setValidation,
    syncYamlToCanvas,
    setIsDirty,
  } = useStudioStore()

  const loadFiles = async (path: string = '') => {
    setIsLoading(true)
    try {
      const fileList = await listFiles(path)
      if (path === '') {
        setFiles(fileList)
      }
      return fileList
    } catch (error) {
      console.error('Failed to load files:', error)
      return []
    } finally {
      setIsLoading(false)
    }
  }

  const openFile = async (path: string) => {
    try {
      const data = await readFile(path)
      setCurrentFile(path)
      setYamlContent(data.content)
      setIsDirty(false)
      setValidation(null)
      // Sync to canvas after a short delay
      setTimeout(() => {
        syncYamlToCanvas()
      }, 100)
    } catch (error) {
      console.error('Failed to open file:', error)
      alert(`Failed to open file: ${error}`)
    }
  }

  const toggleDir = async (path: string) => {
    const newExpanded = new Set(expandedDirs)
    if (newExpanded.has(path)) {
      newExpanded.delete(path)
    } else {
      newExpanded.add(path)
      // Load directory contents if needed
      await loadFiles(path)
    }
    setExpandedDirs(newExpanded)
  }

  useEffect(() => {
    loadFiles()
  }, [])

  const renderFile = (file: FileInfo, depth: number = 0) => {
    const isExpanded = expandedDirs.has(file.path)
    const isActive = currentFile === file.path

    return (
      <div key={file.path}>
        <div
          className={`
            flex items-center gap-2 py-1.5 px-2 cursor-pointer
            hover:bg-hex-border/30 transition-colors rounded-sm
            ${isActive ? 'bg-hex-accent text-white' : 'text-hex-text'}
          `}
          style={{ paddingLeft: `${depth * 12 + 8}px` }}
          onClick={() => file.is_directory ? toggleDir(file.path) : openFile(file.path)}
        >
          {file.is_directory ? (
            <>
              <ChevronRight
                size={14}
                className={`text-hex-text-muted transition-transform ${isExpanded ? 'rotate-90' : ''}`}
              />
              <Folder size={14} className="text-hex-warning" />
            </>
          ) : (
            <>
              <span className="w-3.5" />
              <FileText size={14} className="text-hex-accent" />
            </>
          )}
          <span className="text-xs truncate flex-1">{file.name}</span>
        </div>

        {/* Render children if directory is expanded */}
        {file.is_directory && isExpanded && (
          <div>
            {files
              .filter((f) => {
                const parentPath = f.path.substring(0, f.path.lastIndexOf('/'))
                return parentPath === file.path
              })
              .map((child) => renderFile(child, depth + 1))}
          </div>
        )}
      </div>
    )
  }

  // Get root-level files (no parent directory in path)
  const rootFiles = files.filter((f) => !f.path.includes('/'))

  return (
    <div className="h-full bg-hex-surface border-r border-hex-border flex flex-col">
      <div className="p-3 border-b border-hex-border flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase text-hex-text-muted tracking-wider">
          Files
        </h2>
        <button
          onClick={() => loadFiles()}
          className="p-1 hover:bg-hex-border/50 rounded transition-colors"
          disabled={isLoading}
        >
          <RefreshCw
            size={12}
            className={`text-hex-text-muted ${isLoading ? 'animate-spin' : ''}`}
          />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto py-1">
        {rootFiles.length === 0 ? (
          <div className="p-4 text-center text-hex-text-muted text-xs">
            {isLoading ? 'Loading...' : 'No YAML files found'}
          </div>
        ) : (
          rootFiles.map((file) => renderFile(file))
        )}
      </div>

      <div className="p-2 border-t border-hex-border">
        <div className="text-[10px] text-hex-text-muted text-center">
          {currentFile ? (
            <span className="truncate block">{currentFile}</span>
          ) : (
            'No file selected'
          )}
        </div>
      </div>
    </div>
  )
}
