import React, { useState } from 'react';
import './App.css';
import PredictionForm from './components/PredictionForm';
import ResultCard from './components/ResultCard';
import LoadingSpinner from './components/LoadingSpinner';
import { getPredictions } from './services/api';

export interface PredictionResult {
  text: string;
  confidence: number;
  [key: string]: any;
}

interface PredictionResponse {
  results: PredictionResult[];
}

const App: React.FC = () => {
  const [results, setResults] = useState<PredictionResult[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const handleSubmit = async (inputText: string): Promise<void> => {
    if (!inputText.trim()) {
      setError('Please input Vietnamese text');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const data: PredictionResponse = await getPredictions(inputText);
      setResults(data.results || []);
    } catch (err) {
      setError('Unable to connect to server. Please try again later.');
      console.error('API Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = (): void => {
    setResults([]);
    setError('');
  };

  return (
    <div className="app dark-theme">
      <header className="header">
        <div className="container">
          <div className="header-content">
            <div className="logo-title">
              <div className="logo-icon">
                <i className="fas fa-brain-circuit"></i>
              </div>
              <div>
                <h1>Vietnamese Proverbs & Folk Poetry Prediction System</h1>
                <p className="subtitle">Language Prediction Interface</p>
              </div>
            </div>
            <p className="project-tagline">
              An experimental interface for exploring pattern completion in textual data.
            </p>
          </div>
        </div>
      </header>

      <main className="main-content">
        <div className="container">
          <div className="app-layout">
            {/* Left Panel: Input & Info */}
            <div className="left-panel">
              <div className="info-card">
                <div className="card-header">
                  <i className="fas fa-flask"></i>
                  <h2>Project Overview</h2>
                </div>
                <p>
                  This interface demonstrates a statistical approach to text generation. Input partial text to receive potential continuations based on learned patterns.
                </p>
                <div className="feature-tags">
                  <span className="tag">Pattern Analysis</span>
                  <span className="tag">Statistical Model</span>
                  <span className="tag">Text Generation</span>
                </div>
              </div>

              <PredictionForm
                onSubmit={handleSubmit}
                onReset={handleReset}
                disabled={loading}
              />

              {error && (
                <div className="error-message">
                  <i className="fas fa-exclamation-circle"></i> {error}
                </div>
              )}

              {!loading && results.length === 0 && !error && (
                <div className="empty-state">
                  <div className="empty-illustration">
                    <i className="fas fa-stream"></i>
                  </div>
                  <h3>No predictions yet</h3>
                  <p>Enter text above to generate prediction results.</p>
                </div>
              )}
            </div>

            {/* Right Panel: Results */}
            <div className="right-panel">
              {loading ? (
                <div className="loading-container">
                  <LoadingSpinner />
                </div>
              ) : results.length > 0 ? (
                <div className="results-section">
                  <div className="results-header">
                    <div>
                      <h2>
                        <i className="fas fa-cube"></i>
                        Predictions
                      </h2>
                      <p className="results-count">{results.length} generated</p>
                    </div>
                    <div className="results-controls">
                      <button className="icon-btn" title="Export results">
                        <i className="fas fa-download"></i>
                      </button>
                    </div>
                  </div>

                  <div className="results-container">
                    {results.map((result, index) => (
                      <ResultCard
                        key={index}
                        result={result}
                        index={index + 1}
                      />
                    ))}
                  </div>
                </div>
              ) : (
                <div className="placeholder-section">
                  <div className="placeholder-content">
                    <i className="fas fa-project-diagram"></i>
                    <h3>Output Panel</h3>
                    <p>Generated predictions will appear here in a structured card format.</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      <footer className="footer">
        <div className="container">
          <div className="footer-content">
            <div className="footer-left">
              <p>Language Processing Project • Academic Demo</p>
            </div>
            <div className="footer-right">
              <a
                href="https://github.com/imninh/Cadao-Tucngu-NLP.git"
                target="_blank"
                rel="noopener noreferrer"
                className="github-link"
              >
                <i className="fab fa-github"></i>
                <span>View Source</span>
              </a>
              <p className="copyright">© {new Date().getFullYear()} Research Group</p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;