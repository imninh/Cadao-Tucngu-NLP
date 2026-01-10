import React from 'react'
import './LoadingSpinner.css'

const LoadingSpinner: React.FC = () => {
  return (
    <div className="loading-container">
      <div className="loading-spinner">
        <div className="spinner"></div>
        <p className="loading-text">Processing...</p>
        <p className="loading-subtext">Waiting...</p>
      </div>
    </div>
  )
}

export default LoadingSpinner
