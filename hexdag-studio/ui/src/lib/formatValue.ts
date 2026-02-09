/**
 * Format a value for display in the UI.
 * Handles objects, arrays, and other types properly instead of showing [object Object].
 */

/**
 * Format a value for display, with optional truncation.
 * @param value - The value to format
 * @param maxLength - Maximum length for string output (0 = no limit)
 * @returns Formatted string representation
 */
export function formatValue(value: unknown, maxLength: number = 0): string {
  if (value === null) {
    return 'null'
  }

  if (value === undefined) {
    return 'undefined'
  }

  if (typeof value === 'string') {
    return maxLength > 0 && value.length > maxLength
      ? value.slice(0, maxLength - 1) + '…'
      : value
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return '[]'
    }
    try {
      const formatted = JSON.stringify(value)
      return maxLength > 0 && formatted.length > maxLength
        ? formatted.slice(0, maxLength - 1) + '…'
        : formatted
    } catch {
      return '[Array with circular reference]'
    }
  }

  if (typeof value === 'object') {
    const keys = Object.keys(value)
    if (keys.length === 0) {
      return '{}'
    }
    try {
      const formatted = JSON.stringify(value)
      return maxLength > 0 && formatted.length > maxLength
        ? formatted.slice(0, maxLength - 1) + '…'
        : formatted
    } catch {
      return '{Object with circular reference}'
    }
  }

  // Fallback for functions, symbols, etc.
  return String(value)
}

/**
 * Format a value for display in a compact preview.
 * @param value - The value to format
 * @param maxLength - Maximum length (default: 30)
 */
export function formatValuePreview(value: unknown, maxLength: number = 30): string {
  return formatValue(value, maxLength)
}

/**
 * Format a value for display in a text input field.
 * Objects and arrays are serialized to JSON, other values to strings.
 */
export function formatValueForInput(value: unknown): string {
  if (value === null || value === undefined) {
    return ''
  }

  if (typeof value === 'string') {
    return value
  }

  if (typeof value === 'object') {
    try {
      return JSON.stringify(value, null, 2)
    } catch {
      return '{Object with circular reference}'
    }
  }

  return String(value)
}

/**
 * Parse a string value from an input field.
 * Attempts to parse as JSON, falls back to string.
 */
export function parseInputValue(input: string): unknown {
  if (!input.trim()) {
    return ''
  }

  // Try to parse as JSON if it looks like JSON
  const trimmed = input.trim()
  if (
    (trimmed.startsWith('{') && trimmed.endsWith('}')) ||
    (trimmed.startsWith('[') && trimmed.endsWith(']'))
  ) {
    try {
      return JSON.parse(trimmed)
    } catch {
      // Not valid JSON, return as string
    }
  }

  // Try to parse as number
  if (/^-?\d+(\.\d+)?$/.test(trimmed)) {
    return Number(trimmed)
  }

  // Try to parse as boolean
  if (trimmed === 'true') return true
  if (trimmed === 'false') return false

  // Return as string
  return input
}

/**
 * Generate a unique name that doesn't exist in the given list.
 * @param baseName - Base name to start with
 * @param existingNames - List of names that already exist
 * @param suffix - Optional suffix to append (default: '_copy' for duplicates, '' for new nodes)
 * @returns A unique name
 */
export function generateUniqueName(
  baseName: string,
  existingNames: string[],
  suffix: string = ''
): string {
  const nameWithSuffix = suffix ? `${baseName}${suffix}` : baseName
  let name = nameWithSuffix
  let counter = 1

  while (existingNames.includes(name)) {
    name = suffix ? `${baseName}${suffix}_${counter}` : `${baseName}_${counter}`
    counter++
  }

  return name
}
