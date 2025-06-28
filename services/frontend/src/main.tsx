import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './App.css'

console.log('main.tsx loaded')

try {
  const rootElement = document.getElementById('root')
  console.log('Root element:', rootElement)
  
  if (!rootElement) {
    document.body.innerHTML = '<h1 style="color: red;">ERROR: Root element not found!</h1>'
    throw new Error('Root element not found')
  }

  console.log('Creating React root...')
  const root = createRoot(rootElement)
  
  console.log('Rendering React app...')
  root.render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
  
  console.log('React app rendered successfully')
} catch (error) {
  console.error('Error rendering React app:', error)
  document.body.innerHTML = `<h1 style="color: red;">ERROR: ${error}</h1>`
}