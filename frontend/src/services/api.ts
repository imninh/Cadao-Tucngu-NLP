const API_BASE_URL = import.meta.env.VITE_API_URL;

// ==== TYPES ====

export interface PredictionItem {
  text: string
  confidence: number
  probability?: number
  method?: string
}

export interface PredictionResponse {
  model?: string
  results: PredictionItem[]
}

// ==== REAL API CALL ====

/**
 * Send text input to the N-gram model for prediction
 * @param inputText Partial Vietnamese text
 * @returns Promise with prediction results
 */
export const getPredictions = async (
  inputText: string
): Promise<PredictionResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ input: inputText })
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data: PredictionResponse = await response.json()
    return data
  } catch (error) {
    console.error('Failed to fetch predictions:', error)
    throw error
  }
}

// ==== MOCK API (FOR TESTING) ====

/**
 * Mock API response for testing (when backend is not available)
 */
export const getMockPredictions = async (
  inputText: string
): Promise<PredictionResponse> => {
  // Simulate API delay
  await new Promise<void>((resolve) => setTimeout(resolve, 800))

  const mockResults: PredictionItem[] = [
    {
      text: `${inputText} đẹp và trong xanh.`,
      confidence: 0.85,
      method: 'generated'
    },
    {
      text: `${inputText} nắng và ấm áp.`,
      confidence: 0.72,
      method: 'completed'
    },
    {
      text: `${inputText} mát mẻ dễ chịu.`,
      confidence: 0.63,
      method: 'predicted'
    }
  ]

  return {
    model: 'ngram',
    results: mockResults
  }
}
