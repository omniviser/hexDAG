import { useState, useEffect, useCallback } from 'react'
import { File, Folder, RefreshCw, ChevronRight, ChevronDown } from 'lucide-react'
import { listFiles, readFile } from '../lib/api'
import { useStudioStore } from '../lib/store'
import type { FileInfo } from '../types'

export default function FileBrowser() {
  const [files, setFiles] = useState<FileInfo[]>([])
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set())
  const [dirContents, setDirContents] = useState<Map<string, FileInfo[]>>(new Map())
  const [loadingDirs, setLoadingDirs] = useState<Set<string>>(new Set())
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const { currentFile, setCurrentFile, setYamlContent, syncYamlToCanvas, setIsDirty } = useStudioStore()

  useEffect(() => {
    loadFiles()
  }, [])

  const loadFiles = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const fileList = await listFiles('')
      setFiles(fileList)
      // Clear cached directory contents on refresh
      setDirContents(new Map())
      setExpandedDirs(new Set())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load files')
    } finally {
      setIsLoading(false)
    }
  }

  const loadDirContents = useCallback(async (dirPath: string) => {
    // Use functional state updates to check current state without stale closures
    let alreadyLoadingOrLoaded = false

    // Check and set loading atomically
    setLoadingDirs((prev) => {
      if (prev.has(dirPath)) {
        alreadyLoadingOrLoaded = true
        return prev
      }
      return new Set(prev).add(dirPath)
    })

    if (alreadyLoadingOrLoaded) return

    // Check if already loaded
    setDirContents((prev) => {
      if (prev.has(dirPath)) {
        alreadyLoadingOrLoaded = true
      }
      return prev
    })

    if (alreadyLoadingOrLoaded) {
      setLoadingDirs((prev) => {
        const next = new Set(prev)
        next.delete(dirPath)
        return next
      })
      return
    }

    try {
      const contents = await listFiles(dirPath)
      setDirContents((prev) => new Map(prev).set(dirPath, contents))
    } catch (err) {
      console.error(`Failed to load directory ${dirPath}:`, err)
      setDirContents((prev) => new Map(prev).set(dirPath, []))
    } finally {
      setLoadingDirs((prev) => {
        const next = new Set(prev)
        next.delete(dirPath)
        return next
      })
    }
  }, []) // No dependencies - uses functional updates only

  const handleFileClick = async (file: FileInfo) => {
    if (file.is_directory) {
      const isExpanded = expandedDirs.has(file.path)

      if (!isExpanded) {
        // Load contents when expanding
        await loadDirContents(file.path)
      }

      setExpandedDirs((prev) => {
        const next = new Set(prev)
        if (next.has(file.path)) {
          next.delete(file.path)
        } else {
          next.add(file.path)
        }
        return next
      })
      return
    }

    try {
      const content = await readFile(file.path)
      setCurrentFile(file.path)
      setYamlContent(content.content)
      setIsDirty(false)
      setTimeout(() => {
        syncYamlToCanvas()
      }, 0)
    } catch (err) {
      console.error('Failed to load file:', err)
    }
  }

  const renderFile = (file: FileInfo, depth: number = 0) => {
    const isSelected = currentFile === file.path
    const isExpanded = expandedDirs.has(file.path)
    const isLoadingDir = loadingDirs.has(file.path)
    const paddingLeft = 8 + depth * 12

    return (
      <div key={file.path}>
        <button
          onClick={() => handleFileClick(file)}
          style={{ paddingLeft: `${paddingLeft}px` }}
          className={
            'w-full flex items-center gap-2 py-1.5 pr-2 text-left transition-colors rounded-sm ' +
            (isSelected
              ? 'bg-hex-accent/20 text-hex-accent'
              : 'text-hex-text hover:bg-hex-border/30')
          }
        >
          {file.is_directory ? (
            <>
              {isLoadingDir ? (
                <RefreshCw size={12} className="text-hex-text-muted flex-shrink-0 animate-spin" />
              ) : isExpanded ? (
                <ChevronDown size={12} className="text-hex-text-muted flex-shrink-0" />
              ) : (
                <ChevronRight size={12} className="text-hex-text-muted flex-shrink-0" />
              )}
              <Folder size={14} className="text-hex-warning flex-shrink-0" />
            </>
          ) : (
            <>
              <span className="w-3" />
              <File size={14} className="text-hex-text-muted flex-shrink-0" />
            </>
          )}
          <span className="text-xs truncate">{file.name}</span>
        </button>

        {/* Render directory contents if expanded */}
        {file.is_directory && isExpanded && dirContents.has(file.path) && (
          <div>
            {dirContents.get(file.path)?.map((child) => renderFile(child, depth + 1))}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="flex flex-col border-b border-hex-border">
      <div className="flex items-center justify-between p-2 border-b border-hex-border">
        <h2 className="text-xs font-semibold uppercase text-hex-text-muted tracking-wider">
          Files
        </h2>
        <button
          onClick={loadFiles}
          disabled={isLoading}
          className="p-1 rounded hover:bg-hex-border/50 transition-colors text-hex-text-muted hover:text-hex-text"
          title="Refresh files"
        >
          <RefreshCw size={12} className={isLoading ? 'animate-spin' : ''} />
        </button>
      </div>

      <div className="max-h-48 overflow-y-auto p-1">
        {error ? (
          <div className="p-2 text-xs text-hex-error">{error}</div>
        ) : isLoading ? (
          <div className="p-2 text-xs text-hex-text-muted">Loading...</div>
        ) : files.length === 0 ? (
          <div className="p-2 text-xs text-hex-text-muted">No YAML files found</div>
        ) : (
          files.map((file) => renderFile(file, 0))
        )}
      </div>
    </div>
  )
}
