import { useEffect, useRef } from 'react'
import { Copy, Trash2, Edit3, Unlink, ExternalLink, ChevronRight } from 'lucide-react'

export interface ContextMenuItem {
  id: string
  label: string
  icon?: React.ReactNode
  shortcut?: string
  disabled?: boolean
  danger?: boolean
  divider?: boolean
  submenu?: ContextMenuItem[]
  onClick?: () => void
}

interface ContextMenuProps {
  x: number
  y: number
  items: ContextMenuItem[]
  onClose: () => void
}

export default function ContextMenu({ x, y, items, onClose }: ContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null)

  // Close on outside click or escape
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose()
      }
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }

    document.addEventListener('mousedown', handleClick)
    document.addEventListener('keydown', handleKeyDown)

    return () => {
      document.removeEventListener('mousedown', handleClick)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [onClose])

  // Adjust position to stay within viewport
  useEffect(() => {
    if (menuRef.current) {
      const rect = menuRef.current.getBoundingClientRect()
      const viewportWidth = window.innerWidth
      const viewportHeight = window.innerHeight

      if (rect.right > viewportWidth) {
        menuRef.current.style.left = `${x - rect.width}px`
      }
      if (rect.bottom > viewportHeight) {
        menuRef.current.style.top = `${y - rect.height}px`
      }
    }
  }, [x, y])

  const handleItemClick = (item: ContextMenuItem) => {
    if (item.disabled || item.submenu) return
    item.onClick?.()
    onClose()
  }

  return (
    <div
      ref={menuRef}
      className="fixed z-50 min-w-[180px] bg-hex-surface border border-hex-border rounded-lg shadow-xl py-1"
      style={{ left: x, top: y }}
    >
      {items.map((item) =>
        item.divider ? (
          <div key={item.id} className="h-px bg-hex-border my-1" />
        ) : (
          <button
            key={item.id}
            onClick={() => handleItemClick(item)}
            disabled={item.disabled}
            className={`
              w-full flex items-center gap-2 px-3 py-1.5 text-xs text-left
              transition-colors
              ${item.disabled
                ? 'text-hex-text-muted cursor-not-allowed'
                : item.danger
                  ? 'text-hex-error hover:bg-hex-error/10'
                  : 'text-hex-text hover:bg-hex-border/50'
              }
            `}
          >
            {item.icon && (
              <span className="w-4 h-4 flex items-center justify-center">
                {item.icon}
              </span>
            )}
            <span className="flex-1">{item.label}</span>
            {item.shortcut && (
              <span className="text-hex-text-muted text-[10px]">{item.shortcut}</span>
            )}
            {item.submenu && (
              <ChevronRight size={12} className="text-hex-text-muted" />
            )}
          </button>
        )
      )}
    </div>
  )
}

// Helper to create standard node context menu items
export function createNodeContextMenuItems(
  _nodeId: string,
  options: {
    onEdit?: () => void
    onDuplicate?: () => void
    onDelete?: () => void
    onDisconnect?: () => void
    onViewCode?: () => void
  }
): ContextMenuItem[] {
  const items: ContextMenuItem[] = []

  if (options.onEdit) {
    items.push({
      id: 'edit',
      label: 'Edit Node',
      icon: <Edit3 size={14} />,
      onClick: options.onEdit,
    })
  }

  if (options.onViewCode) {
    items.push({
      id: 'view-code',
      label: 'View Code',
      icon: <ExternalLink size={14} />,
      onClick: options.onViewCode,
    })
  }

  if (options.onDuplicate) {
    items.push({
      id: 'duplicate',
      label: 'Duplicate',
      icon: <Copy size={14} />,
      shortcut: 'Cmd+D',
      onClick: options.onDuplicate,
    })
  }

  if (options.onDisconnect) {
    items.push({
      id: 'divider-1',
      label: '',
      divider: true,
    })
    items.push({
      id: 'disconnect',
      label: 'Disconnect All',
      icon: <Unlink size={14} />,
      onClick: options.onDisconnect,
    })
  }

  if (options.onDelete) {
    items.push({
      id: 'divider-2',
      label: '',
      divider: true,
    })
    items.push({
      id: 'delete',
      label: 'Delete',
      icon: <Trash2 size={14} />,
      shortcut: 'Del',
      danger: true,
      onClick: options.onDelete,
    })
  }

  return items
}
