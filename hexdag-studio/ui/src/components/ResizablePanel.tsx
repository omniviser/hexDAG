import { useState, useRef, useCallback, useEffect } from 'react'

interface ResizablePanelProps {
  children: React.ReactNode
  defaultWidth: number
  minWidth: number
  maxWidth: number
  side: 'left' | 'right'
  className?: string
}

export default function ResizablePanel({
  children,
  defaultWidth,
  minWidth,
  maxWidth,
  side,
  className = '',
}: ResizablePanelProps) {
  const [width, setWidth] = useState(defaultWidth)
  const [isResizing, setIsResizing] = useState(false)
  const panelRef = useRef<HTMLDivElement>(null)

  const startResizing = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setIsResizing(true)
  }, [])

  const stopResizing = useCallback(() => {
    setIsResizing(false)
  }, [])

  const resize = useCallback(
    (e: MouseEvent) => {
      if (!isResizing || !panelRef.current) return

      const rect = panelRef.current.getBoundingClientRect()
      let newWidth: number

      if (side === 'left') {
        newWidth = e.clientX - rect.left
      } else {
        newWidth = rect.right - e.clientX
      }

      if (newWidth >= minWidth && newWidth <= maxWidth) {
        setWidth(newWidth)
      }
    },
    [isResizing, minWidth, maxWidth, side]
  )

  useEffect(() => {
    if (isResizing) {
      window.addEventListener('mousemove', resize)
      window.addEventListener('mouseup', stopResizing)
    }

    return () => {
      window.removeEventListener('mousemove', resize)
      window.removeEventListener('mouseup', stopResizing)
    }
  }, [isResizing, resize, stopResizing])

  return (
    <div
      ref={panelRef}
      className={`relative flex-shrink-0 ${className}`}
      style={{ width }}
    >
      {children}
      {/* Resize handle */}
      <div
        className={`absolute top-0 bottom-0 w-1 cursor-col-resize z-10 hover:bg-hex-accent/50 transition-colors ${
          isResizing ? 'bg-hex-accent' : 'bg-transparent'
        } ${side === 'left' ? 'right-0' : 'left-0'}`}
        onMouseDown={startResizing}
      />
    </div>
  )
}

interface ResizableHorizontalProps {
  children: [React.ReactNode, React.ReactNode]
  defaultSplit?: number // 0-100 percentage for left panel
  minLeft?: number
  minRight?: number
  className?: string
}

export function ResizableHorizontal({
  children,
  defaultSplit = 50,
  minLeft = 200,
  minRight = 200,
  className = '',
}: ResizableHorizontalProps) {
  const [splitPercent, setSplitPercent] = useState(defaultSplit)
  const [isResizing, setIsResizing] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const startResizing = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setIsResizing(true)
  }, [])

  const stopResizing = useCallback(() => {
    setIsResizing(false)
  }, [])

  const resize = useCallback(
    (e: MouseEvent) => {
      if (!isResizing || !containerRef.current) return

      const rect = containerRef.current.getBoundingClientRect()
      const containerWidth = rect.width
      const relativeX = e.clientX - rect.left
      const newPercent = (relativeX / containerWidth) * 100

      // Check min widths
      const leftWidth = (newPercent / 100) * containerWidth
      const rightWidth = containerWidth - leftWidth

      if (leftWidth >= minLeft && rightWidth >= minRight) {
        setSplitPercent(newPercent)
      }
    },
    [isResizing, minLeft, minRight]
  )

  useEffect(() => {
    if (isResizing) {
      window.addEventListener('mousemove', resize)
      window.addEventListener('mouseup', stopResizing)
    }

    return () => {
      window.removeEventListener('mousemove', resize)
      window.removeEventListener('mouseup', stopResizing)
    }
  }, [isResizing, resize, stopResizing])

  return (
    <div ref={containerRef} className={`flex h-full ${className}`}>
      <div style={{ width: `${splitPercent}%` }} className="overflow-hidden">
        {children[0]}
      </div>
      {/* Resize handle */}
      <div
        className={`w-1 cursor-col-resize flex-shrink-0 hover:bg-hex-accent/50 transition-colors ${
          isResizing ? 'bg-hex-accent' : 'bg-hex-border'
        }`}
        onMouseDown={startResizing}
      />
      <div style={{ width: `${100 - splitPercent}%` }} className="overflow-hidden">
        {children[1]}
      </div>
    </div>
  )
}
