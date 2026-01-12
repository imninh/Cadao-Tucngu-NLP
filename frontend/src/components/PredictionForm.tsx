import React, { useState, useCallback, memo } from 'react'
import type { FormEvent, ChangeEvent } from 'react'
import './PredictionForm.css'

interface PredictionFormProps {
  onSubmit: (inputText: string) => void
  onReset: () => void
  disabled: boolean
  maxLength?: number
}

const PredictionForm: React.FC<PredictionFormProps> = memo(({
  onSubmit,
  onReset,
  disabled,
  maxLength = 500
}) => {
  const [inputText, setInputText] = useState<string>('')

  const handleSubmit = useCallback((e: FormEvent<HTMLFormElement>): void => {
    e.preventDefault()
    if (inputText.trim() && !disabled) {
      onSubmit(inputText)
    }
  }, [inputText, disabled, onSubmit])

  const handleReset = useCallback((): void => {
    setInputText('')
    onReset()
  }, [onReset])

  const handleChange = useCallback((e: ChangeEvent<HTMLTextAreaElement>): void => {
    const value = e.target.value
    if (value.length <= maxLength) {
      setInputText(value)
    }
  }, [maxLength])

  const handleClear = useCallback((): void => {
    setInputText('')
  }, [])

  const isNearLimit = inputText.length > maxLength * 0.9
  const isAtLimit = inputText.length >= maxLength

  return (
    <div className="prediction-form">
      <header className="form-header">
        <h2 className="form-title">
          <i className="fas fa-keyboard form-icon"></i>
          Text Input
        </h2>
        <p className="form-description">
          Enter Vietnamese text to generate predictions 
        </p>
      </header>

      <form onSubmit={handleSubmit} className="form-content" noValidate>
        <div className="form-group">
          <label htmlFor="text-input" className="form-label">
            Vietnamese Input
          </label>
          <textarea
            id="text-input"
            className={`form-textarea ${isAtLimit ? 'form-textarea--limit' : ''}`}
            value={inputText}
            onChange={handleChange}
            placeholder="Type Vietnamese text here..."
            rows={4}
            disabled={disabled}
            maxLength={maxLength}
            aria-label="Vietnamese text input for N-gram prediction"
            aria-describedby="char-count"
          />
          
          <div className="form-meta">
            <div className="form-meta-left">
              <button
                type="button"
                className="btn-clear"
                onClick={handleClear}
                disabled={!inputText || disabled}
                aria-label="Clear input text"
              >
                <i className="fas fa-times"></i> Clear
              </button>
            </div>
            
            <div className="form-meta-right">
              <span 
                id="char-count"
                className={`char-count ${isNearLimit ? 'char-count--warning' : ''} ${isAtLimit ? 'char-count--limit' : ''}`}
                aria-live="polite"
              >
                {inputText.length}/{maxLength}
              </span>
            </div>
          </div>
        </div>

        <div className="form-actions">
          <button
            type="button"
            className="btn btn--secondary"
            onClick={handleReset}
            disabled={disabled}
            aria-label="Reset form and clear results"
          >
            <i className="fas fa-redo-alt"></i>
            Reset All
          </button>

          <button
            type="submit"
            className="btn btn--primary"
            disabled={disabled || !inputText.trim()}
            aria-label="Submit text for prediction"
            aria-busy={disabled}
          >
            {disabled ? (
              <>
                <i className="fas fa-spinner fa-spin" aria-hidden="true"></i>
                Processing...
              </>
            ) : (
              <>
                <i className="fas fa-bolt" aria-hidden="true"></i>
                Generate Predictions
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  )
})

PredictionForm.displayName = 'PredictionForm'

export default PredictionForm