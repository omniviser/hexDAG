/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'hex': {
          'bg': '#0f0f1a',
          'surface': '#1a1a2e',
          'border': '#2a2a3e',
          'accent': '#6366f1',
          'accent-hover': '#818cf8',
          'text': '#e4e4e7',
          'text-muted': '#a1a1aa',
          'success': '#22c55e',
          'error': '#ef4444',
          'warning': '#f59e0b',
        }
      },
      fontFamily: {
        'mono': ['SF Mono', 'Fira Code', 'Consolas', 'monospace'],
      }
    },
  },
  plugins: [],
}
