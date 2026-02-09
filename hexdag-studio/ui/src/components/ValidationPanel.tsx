import { CheckCircle, XCircle, AlertTriangle, Info } from 'lucide-react'
import { useStudioStore } from '../lib/store'
import type { ValidationError } from '../types'

export default function ValidationPanel() {
  const { validation } = useStudioStore()

  if (!validation) {
    return (
      <div className="h-full flex items-center justify-center text-hex-text-muted">
        <div className="text-center">
          <Info size={32} className="mx-auto mb-2 opacity-50" />
          <p className="text-xs">Edit a file to see validation</p>
        </div>
      </div>
    )
  }

  const errorCount = validation.errors.filter((e: ValidationError) => e.severity === 'error').length
  const warningCount = validation.errors.filter((e: ValidationError) => e.severity === 'warning').length

  return (
    <div className="h-full flex flex-col">
      {/* Summary */}
      <div className="p-3 border-b border-hex-border flex items-center gap-3">
        {validation.valid ? (
          <div className="flex items-center gap-2 text-hex-success">
            <CheckCircle size={16} />
            <span className="text-xs font-medium">Valid</span>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-hex-error">
            <XCircle size={16} />
            <span className="text-xs font-medium">Invalid</span>
          </div>
        )}

        {validation.node_count !== undefined && (
          <div className="text-xs text-hex-text-muted">
            {validation.node_count} node{validation.node_count !== 1 ? 's' : ''}
          </div>
        )}

        <div className="ml-auto flex items-center gap-2 text-xs">
          {errorCount > 0 && (
            <span className="text-hex-error">{errorCount} error{errorCount !== 1 ? 's' : ''}</span>
          )}
          {warningCount > 0 && (
            <span className="text-hex-warning">{warningCount} warning{warningCount !== 1 ? 's' : ''}</span>
          )}
        </div>
      </div>

      {/* Errors list */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {validation.errors.length === 0 && validation.valid && (
          <div className="p-3 rounded-md bg-hex-success/10 border border-hex-success/20">
            <div className="flex items-start gap-2">
              <CheckCircle size={14} className="text-hex-success mt-0.5" />
              <div>
                <p className="text-xs text-hex-success font-medium">Pipeline is valid</p>
                {validation.nodes && validation.nodes.length > 0 && (
                  <p className="text-[10px] text-hex-text-muted mt-1">
                    Execution order: {validation.nodes.join(' → ')}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {validation.errors.map((error: ValidationError, index: number) => (
          <div
            key={index}
            className={`
              p-2 rounded-md border-l-2
              ${error.severity === 'error'
                ? 'bg-hex-error/10 border-hex-error'
                : error.severity === 'warning'
                  ? 'bg-hex-warning/10 border-hex-warning'
                  : 'bg-hex-accent/10 border-hex-accent'}
            `}
          >
            <div className="flex items-start gap-2">
              {error.severity === 'error' ? (
                <XCircle size={14} className="text-hex-error mt-0.5 flex-shrink-0" />
              ) : error.severity === 'warning' ? (
                <AlertTriangle size={14} className="text-hex-warning mt-0.5 flex-shrink-0" />
              ) : (
                <Info size={14} className="text-hex-accent mt-0.5 flex-shrink-0" />
              )}
              <div className="flex-1 min-w-0">
                <p className="text-xs text-hex-text break-words">{error.message}</p>
                {error.line && (
                  <p className="text-[10px] text-hex-text-muted mt-0.5">
                    Line {error.line}{error.column ? `:${error.column}` : ''}
                  </p>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
