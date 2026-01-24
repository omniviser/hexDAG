import { useEffect, useRef, useCallback } from 'react'
import Editor, { type OnMount, type OnChange } from '@monaco-editor/react'
import type * as Monaco from 'monaco-editor'
import { useStudioStore } from '../lib/store'
import { validateYaml } from '../lib/api'

export default function YamlEditor() {
  const editorRef = useRef<Monaco.editor.IStandaloneCodeEditor | null>(null)
  const monacoRef = useRef<typeof Monaco | null>(null)
  const validationTimeoutRef = useRef<number | null>(null)

  const {
    yamlContent,
    setYamlContent,
    setValidation,
    syncYamlToCanvas,
    currentFile,
  } = useStudioStore()

  const handleEditorMount: OnMount = (editor, monaco) => {
    editorRef.current = editor
    monacoRef.current = monaco

    // Configure YAML language
    monaco.languages.registerCompletionItemProvider('yaml', {
      provideCompletionItems: (model: Monaco.editor.ITextModel, position: Monaco.Position) => {
        const word = model.getWordUntilPosition(position)
        const range = {
          startLineNumber: position.lineNumber,
          endLineNumber: position.lineNumber,
          startColumn: word.startColumn,
          endColumn: word.endColumn,
        }

        const suggestions: Monaco.languages.CompletionItem[] = [
          {
            label: 'apiVersion',
            kind: monaco.languages.CompletionItemKind.Property,
            insertText: 'apiVersion: hexdag/v1',
            range,
          },
          {
            label: 'kind',
            kind: monaco.languages.CompletionItemKind.Property,
            insertText: 'kind: Pipeline',
            range,
          },
          {
            label: 'metadata',
            kind: monaco.languages.CompletionItemKind.Property,
            insertText: 'metadata:\n  name: ${1:pipeline-name}',
            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
            range,
          },
          {
            label: 'spec',
            kind: monaco.languages.CompletionItemKind.Property,
            insertText: 'spec:\n  nodes:\n    - kind: ${1:function_node}\n      metadata:\n        name: ${2:node-name}\n      spec:\n        ${3:}\n      dependencies: []',
            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
            range,
          },
          {
            label: 'llm_node',
            kind: monaco.languages.CompletionItemKind.Value,
            insertText: 'kind: llm_node\n      metadata:\n        name: ${1:llm}\n      spec:\n        prompt_template: "${2:Process: {{input}}}"\n      dependencies: [${3}]',
            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
            range,
          },
          {
            label: 'function_node',
            kind: monaco.languages.CompletionItemKind.Value,
            insertText: 'kind: function_node\n      metadata:\n        name: ${1:function}\n      spec:\n        fn: "${2:json.loads}"\n      dependencies: [${3}]',
            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
            range,
          },
        ]

        return { suggestions }
      },
    })

    // Define custom theme
    monaco.editor.defineTheme('hexdag-dark', {
      base: 'vs-dark',
      inherit: true,
      rules: [
        { token: 'key', foreground: '6366f1' },
        { token: 'string.yaml', foreground: '22c55e' },
        { token: 'number.yaml', foreground: 'f59e0b' },
        { token: 'keyword.yaml', foreground: 'ec4899' },
      ],
      colors: {
        'editor.background': '#0f0f1a',
        'editor.foreground': '#e4e4e7',
        'editor.lineHighlightBackground': '#1a1a2e',
        'editorCursor.foreground': '#6366f1',
        'editor.selectionBackground': '#6366f140',
        'editorLineNumber.foreground': '#4a4a5e',
        'editorLineNumber.activeForeground': '#6366f1',
      },
    })

    monaco.editor.setTheme('hexdag-dark')

    // Keyboard shortcut for save
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
      // Trigger save via parent component
      const event = new CustomEvent('hexdag-save')
      window.dispatchEvent(event)
    })
  }

  const runValidation = useCallback(async (content: string) => {
    if (!content.trim()) {
      setValidation(null)
      return
    }

    try {
      const result = await validateYaml(content, currentFile || undefined)
      setValidation(result)

      // Update editor markers
      if (monacoRef.current && editorRef.current) {
        const model = editorRef.current.getModel()
        if (model) {
          const markers: Monaco.editor.IMarkerData[] = result.errors.map((err) => ({
            severity: err.severity === 'error'
              ? monacoRef.current!.MarkerSeverity.Error
              : err.severity === 'warning'
                ? monacoRef.current!.MarkerSeverity.Warning
                : monacoRef.current!.MarkerSeverity.Info,
            message: err.message,
            startLineNumber: err.line || 1,
            startColumn: err.column || 1,
            endLineNumber: err.line || 1,
            endColumn: (err.column || 1) + 10,
          }))
          monacoRef.current.editor.setModelMarkers(model, 'hexdag', markers)
        }
      }
    } catch (error) {
      console.error('Validation error:', error)
    }
  }, [currentFile, setValidation])

  const handleChange: OnChange = useCallback((value) => {
    const content = value || ''
    setYamlContent(content)

    // Debounced validation
    if (validationTimeoutRef.current) {
      clearTimeout(validationTimeoutRef.current)
    }
    validationTimeoutRef.current = window.setTimeout(() => {
      runValidation(content)
      syncYamlToCanvas()
    }, 500)
  }, [setYamlContent, runValidation, syncYamlToCanvas])

  // Initial validation
  useEffect(() => {
    if (yamlContent) {
      runValidation(yamlContent)
    }
  }, []) // Only run once on mount

  return (
    <div className="h-full w-full">
      <Editor
        height="100%"
        language="yaml"
        value={yamlContent}
        onChange={handleChange}
        onMount={handleEditorMount}
        options={{
          minimap: { enabled: false },
          fontSize: 13,
          fontFamily: "'SF Mono', 'Fira Code', Consolas, monospace",
          lineNumbers: 'on',
          renderWhitespace: 'selection',
          scrollBeyondLastLine: false,
          automaticLayout: true,
          tabSize: 2,
          wordWrap: 'on',
          folding: true,
          bracketPairColorization: { enabled: true },
          suggest: {
            showWords: false,
          },
        }}
        theme="hexdag-dark"
      />
    </div>
  )
}
