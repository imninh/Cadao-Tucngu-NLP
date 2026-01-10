import React, { useState } from 'react'
import './App.css'
import PredictionForm from './components/PredictionForm'
import ResultCard from './components/ResultCard'
import LoadingSpinner from './components/LoadingSpinner'
import { getPredictions } from './services/api'

// Định nghĩa type cho 1 kết quả dự đoán
export interface PredictionResult {
  text: string
  confidence: number
  [key: string]: any
}

// Định nghĩa type cho response API
interface PredictionResponse {
  results: PredictionResult[]
}

const App: React.FC = () => {
  const [results, setResults] = useState<PredictionResult[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')

  const handleSubmit = async (inputText: string): Promise<void> => {
    if (!inputText.trim()) {
      setError('Vui lòng nhập văn bản tiếng Việt')
      return
    }

    setLoading(true)
    setError('')

    try {
      const data: PredictionResponse = await getPredictions(inputText)
      setResults(data.results || [])
    } catch (err) {
      setError('Không thể kết nối đến máy chủ. Vui lòng thử lại sau.')
      console.error('API Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = (): void => {
    setResults([])
    setError('')
  }

  return (
    <div className="app">
      <header className="header">
        <div className="container">
          <h1>
            <i className="fas fa-language"></i> N-gram
          </h1>
          <p className="subtitle">N-gram (n=4)</p>
          <p className="description">
            Demo NLP
          </p>
        </div>
      </header>

      <main className="main-content">
        <div className="container">
          <div className="demo-section">
            <div className="model-info">
              <h2>
                <i className="fas fa-info-circle"></i> Thông tin mô hình
              </h2>
              <p>
                <strong>Mô hình N-gram (n=4)</strong> 
              </p>
              <div className="tech-badge">
                <span>NLP</span>
                <span>Thống kê</span>
                <span>N-gram</span>
                <span>Tiếng Việt</span>
              </div>
            </div>

            <PredictionForm
              onSubmit={handleSubmit}
              onReset={handleReset}
              disabled={loading}
            />

            {error && (
              <div className="error-message">
                <i className="fas fa-exclamation-triangle"></i> {error}
              </div>
            )}

            {loading && <LoadingSpinner />}

            {!loading && results.length > 0 && (
              <div className="results-section">
                <div className="section-header">
                  <h2>
                    <i className="fas fa-list-alt"></i> Result
                  </h2>
                  <p className="result-count">
                    {results.length} Result found
                  </p>
                </div>
                <div className="results-grid">
                  {results.map((result, index) => (
                    <ResultCard
                      key={index}
                      result={result}
                      index={index + 1}
                    />
                  ))}
                </div>
              </div>
            )}

            {!loading && results.length === 0 && !error && (
              <div className="empty-state">
                <i className="fas fa-comment-dots"></i>
                <h3>Chưa có kết quả</h3>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="footer">
        <div className="container">
          <p> NLP - Mô hình N-gram &copy; {new Date().getFullYear()}</p>
          <p className="github-link">
            <i className="fab fa-github"></i>
            <a
              href="https://github.com/imninh/Cadao-Tucngu-NLP.git"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
