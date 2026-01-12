import React from 'react'
import './LoadingSpinner.css'

interface LoadingSpinnerProps {
  message?: string
  subMessage?: string
  size?: 'small' | 'medium' | 'large'
  fullPage?: boolean
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  message = 'Processing with Retrieval Model',
  subMessage = 'Analyzing text patterns...',
  size = 'medium',
  fullPage = false
}) => {
  const spinnerSize = {
    small: { diameter: 40, strokeWidth: 4 },
    medium: { diameter: 60, strokeWidth: 5 },
    large: { diameter: 80, strokeWidth: 6 }
  }[size]

  const strokeDasharray = 2 * Math.PI * (spinnerSize.diameter / 2 - spinnerSize.strokeWidth)
  const strokeDashoffset = strokeDasharray * 0.75

  return (
    <div 
      className={`loading-container ${fullPage ? 'loading-container--fullpage' : ''}`}
      role="alert"
      aria-live="assertive"
      aria-label="Loading content"
    >
      <div className="loading-spinner">
        <div className="spinner-wrapper" style={{ width: spinnerSize.diameter, height: spinnerSize.diameter }}>
          <svg 
            className="spinner-svg" 
            viewBox={`0 0 ${spinnerSize.diameter} ${spinnerSize.diameter}`}
            aria-hidden="true"
          >
            <circle
              className="spinner-background"
              cx={spinnerSize.diameter / 2}
              cy={spinnerSize.diameter / 2}
              r={spinnerSize.diameter / 2 - spinnerSize.strokeWidth}
              strokeWidth={spinnerSize.strokeWidth}
              fill="none"
            />
            <circle
              className="spinner-foreground"
              cx={spinnerSize.diameter / 2}
              cy={spinnerSize.diameter / 2}
              r={spinnerSize.diameter / 2 - spinnerSize.strokeWidth}
              strokeWidth={spinnerSize.strokeWidth}
              fill="none"
              strokeDasharray={strokeDasharray}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
            />
          </svg>
          <div className="spinner-inner">
            <i className="fas fa-brain spinner-icon" aria-hidden="true"></i>
          </div>
        </div>

        <div className="loading-content">
          <h3 className="loading-title">{message}</h3>
          <p className="loading-description">{subMessage}</p>
          
          <div className="loading-progress" role="progressbar" aria-valuetext="Loading">
            <div className="loading-progress-bar"></div>
          </div>

          <div className="loading-details">
            <div className="loading-detail">
              <i className="fas fa-microchip" aria-hidden="true"></i>
              <span>Retrieval Analysis</span>
            </div>
            <div className="loading-detail">
              <i className="fas fa-chart-line" aria-hidden="true"></i>
              <span>Statistical Processing</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default LoadingSpinner
