import React from 'react'
import './ResultCard.css'

interface ResultData {
  text: string
  confidence: number
  method?: string
  [key: string]: any
}

interface ResultCardProps {
  result: ResultData
  index: number
}

const ResultCard: React.FC<ResultCardProps> = ({ result, index }) => {
  // Format confidence as percentage
  const confidencePercent: string = (result.confidence * 100).toFixed(1)

  // Determine confidence level for styling
  const getConfidenceLevel = (confidence: number): 'high' | 'medium' | 'low' => {
    if (confidence >= 0.8) return 'high'
    if (confidence >= 0.5) return 'medium'
    return 'low'
  }

  const confidenceLevel = getConfidenceLevel(result.confidence)

  // Translate method to Vietnamese
  const getMethodText = (method?: string): string => {
    const methodMap: Record<string, string> = {
      generated: 'Tạo mới',
      completed: 'Hoàn thiện',
      predicted: 'Dự đoán',
      default: 'Mặc định'
    }
    return method ? methodMap[method] || method : 'Không xác định'
  }

  return (
    <div className="result-card">
      <div className="card-header">
        <div className="card-index">
          <span className="index-number">{index}</span>
        </div>
        <div className="card-title">
          <h3>OK #{index}</h3>
          <span className={`confidence-badge ${confidenceLevel}`}>
            <i className="fas fa-chart-line"></i> {confidencePercent}%
          </span>
        </div>
      </div>

      <div className="card-body">
        <div className="result-text">
          <p>
            <i className="fas fa-quote-left"></i> {result.text}{' '}
            <i className="fas fa-quote-right"></i>
          </p>
        </div>

        <div className="result-meta">
          <div className="meta-item">
            <div className="meta-label">
              <i className="fas fa-percentage"></i> Độ tin cậy
            </div>
            <div className="meta-value">
              <div className="confidence-bar">
                <div
                  className={`confidence-fill ${confidenceLevel}`}
                  style={{ width: `${confidencePercent}%` }}
                ></div>
              </div>
              <span className="confidence-text">{confidencePercent}%</span>
            </div>
          </div>

          <div className="meta-item">
            <div className="meta-label">
              <i className="fas fa-cogs"></i> Method
            </div>
            <div className="meta-value method-value">
              {getMethodText(result.method)}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ResultCard
