import React, { useState, type ChangeEvent, type FormEvent } from 'react'
import './PredictionForm.css'

interface PredictionFormProps {
  onSubmit: (inputText: string) => void
  onReset: () => void
  disabled: boolean
}

const PredictionForm: React.FC<PredictionFormProps> = ({
  onSubmit,
  onReset,
  disabled
}) => {
  const [inputText, setInputText] = useState<string>('')

  const handleSubmit = (e: FormEvent<HTMLFormElement>): void => {
    e.preventDefault()
    if (inputText.trim() && !disabled) {
      onSubmit(inputText)
    }
  }

  const handleReset = (): void => {
    setInputText('')
    onReset()
  }

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>): void => {
    setInputText(e.target.value)
  }

  return (
    <div className="prediction-form">
      <h2>
        <i className="fas fa-keyboard"></i> Nhập văn bản
      </h2>

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <textarea
            id="text-input"
            value={inputText}
            onChange={handleChange}
            placeholder="Nhập văn bản..."
            rows={4}
            disabled={disabled}
          />
          <div className="character-count">
            {inputText.length} ký tự
          </div>
        </div>

        <div className="form-actions">
          <button
            type="button"
            className="btn-reset"
            onClick={handleReset}
            disabled={disabled}
          >
            <i className="fas fa-redo"></i> Reset
          </button>

          <button
            type="submit"
            className="btn-submit"
            disabled={disabled || !inputText.trim()}
          >
            {disabled ? (
              <>
                <i className="fas fa-spinner fa-spin"></i> Processing...
              </>
            ) : (
              <>
                <i className="fas fa-lightbulb"></i> OK
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  )
}

export default PredictionForm
