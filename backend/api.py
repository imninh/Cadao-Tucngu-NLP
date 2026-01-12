"""
FIXED API.PY - Compatible with Improved Retrieval Model
Works with both old and new model formats
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import sys
import torch

app = Flask(__name__)
CORS(app)

print("="*60)
print("🚀 CADAO AUTOCOMPLETE API - LOADING")
print("="*60)

# Get directories
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")
trained_models_dir = os.path.join(current_dir, "trained_models")

print(f"📁 Current dir: {current_dir}")
print(f"📁 Models dir: {models_dir}")
print(f"📁 Trained models dir: {trained_models_dir}")

# Add models to path
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

# ============================================================
# LOAD RETRIEVAL MODEL
# ============================================================

retrieval_model = None

try:
    print(f"\n📚 Loading Retrieval Model...")
    
    # Import RetrievalModel class
    from retrieval import RetrievalModel
    
    # Try different model files
    model_files = [
        "improved_retrieval_model.pkl",  # New BM25 model
        "retrieval_model.pkl"             # Old TF-IDF model
    ]
    
    for model_file in model_files:
        model_path = os.path.join(trained_models_dir, model_file)
        
        if os.path.exists(model_path):
            print(f"   Found: {model_file}")
            
            try:
                # Create model instance
                retrieval_model = RetrievalModel()
                
                # Try loading
                retrieval_model.load(model_path)
                
                print(f"✅ Retrieval model loaded!")
                print(f"   Type: {model_file}")
                print(f"   Database: {len(retrieval_model.database)} sentences")
                
                # Check if model is properly trained
                if hasattr(retrieval_model, 'is_trained') and retrieval_model.is_trained:
                    print(f"   Status: ✅ Trained")
                elif hasattr(retrieval_model, 'vectors') and retrieval_model.vectors is not None:
                    # Old model format
                    print(f"   Status: ✅ Legacy format")
                    retrieval_model.is_trained = True
                else:
                    print(f"   Status: ⚠️  May not be trained properly")
                
                break
                
            except KeyError as e:
                print(f"   ⚠️  Missing key: {e}")
                print(f"   This is an old/incompatible model format")
                
                # Try manual reconstruction for old format
                try:
                    print(f"   Attempting manual load...")
                    
                    with open(model_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    retrieval_model = RetrievalModel()
                    retrieval_model.vectorizer = data['vectorizer']
                    retrieval_model.database = data['database']
                    
                    # Old format might have 'vectors' instead of 'term_freqs'
                    if 'vectors' in data:
                        retrieval_model.vectors = data['vectors']
                        retrieval_model.is_trained = True
                        
                        print(f"✅ Loaded legacy model!")
                        print(f"   Database: {len(retrieval_model.database)} sentences")
                        break
                    else:
                        print(f"   ❌ Cannot reconstruct - missing data")
                        
                except Exception as e2:
                    print(f"   ❌ Manual load failed: {e2}")
                    continue
                    
            except Exception as e:
                print(f"   ❌ Load failed: {e}")
                continue
    
    if retrieval_model is None:
        print(f"\n⚠️  No valid retrieval model found!")
        print(f"\n💡 Please train the model:")
        print(f"   cd backend/models")
        print(f"   python retrieval.py")
        
except Exception as e:
    print(f"❌ Error loading Retrieval: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# LOAD LSTM MODEL (Optional)
# ============================================================

lstm_predictor = None

try:
    print(f"\n🧠 Loading LSTM Model...")
    
    lstm_path = os.path.join(trained_models_dir, "lstm_cadao.pt")
    
    if os.path.exists(lstm_path):
        print(f"   Found: lstm_cadao.pt")
        
        # Try importing LSTM classes
        try:
            from lstm_cadao import LSTMCaDao, LSTMPredictor # type: ignore
            
            # Load checkpoint
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            checkpoint = torch.load(lstm_path, map_location=device)
            vocab = checkpoint['vocab']
            
            # Create model
            model = LSTMCaDao(
                vocab_size=vocab.n_words,
                embed_dim=128,
                hidden_dim=256,
                num_layers=2
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Create predictor
            lstm_predictor = LSTMPredictor(model, vocab, device)
            
            print(f"✅ LSTM model loaded!")
            print(f"   Vocab: {vocab.n_words} words")
            print(f"   Device: {device}")
            
        except ImportError:
            print(f"   ⚠️  lstm_cadao.py not found")
            print(f"   LSTM model available but cannot load")
    else:
        print(f"   ℹ️  LSTM model not found (optional)")
        
except Exception as e:
    print(f"⚠️  LSTM not available: {e}")

# ============================================================
# CREATE ENSEMBLE
# ============================================================

ensemble_model = None

if retrieval_model and lstm_predictor:
    print(f"\n🎯 Creating Ensemble...")
    
    try:
        class SimpleEnsemble:
            """Simple ensemble: Retrieval + LSTM"""
            
            def __init__(self, retrieval, lstm):
                self.retrieval = retrieval
                self.lstm = lstm
            
            def predict_multiple(self, input_text, top_k=5):
                """Ensemble prediction"""
                from collections import defaultdict
                
                # Get predictions
                ret_cands = []
                lstm_cands = []
                
                try:
                    ret_cands = self.retrieval.predict_multiple(input_text, top_k=top_k)
                except Exception as e:
                    print(f"   ⚠️  Retrieval error: {e}")
                
                try:
                    lstm_cands = self.lstm.predict_multiple(input_text, top_k=top_k)
                except Exception as e:
                    print(f"   ⚠️  LSTM error: {e}")
                
                # Combine scores (50/50 weight)
                scores = defaultdict(lambda: {'score': 0, 'sources': []})
                
                for i, cand in enumerate(ret_cands):
                    text = cand['text']
                    conf = cand.get('confidence', 0.5)
                    rank_bonus = 0.1 * (1 - i / max(len(ret_cands), 1))
                    
                    scores[text]['score'] += (conf + rank_bonus) * 0.5
                    scores[text]['sources'].append('retrieval')
                
                for i, cand in enumerate(lstm_cands):
                    text = cand['text']
                    conf = cand.get('confidence', 0.5)
                    rank_bonus = 0.1 * (1 - i / max(len(lstm_cands), 1))
                    
                    scores[text]['score'] += (conf + rank_bonus) * 0.5
                    scores[text]['sources'].append('lstm')
                
                # Diversity bonus
                for text, info in scores.items():
                    if len(set(info['sources'])) > 1:
                        scores[text]['score'] += 0.15
                
                # Sort
                sorted_items = sorted(
                    scores.items(),
                    key=lambda x: x[1]['score'],
                    reverse=True
                )
                
                # Format
                results = []
                for text, info in sorted_items[:top_k]:
                    results.append({
                        'text': text,
                        'confidence': min(0.95, info['score']),
                        'model': 'ensemble',
                        'sources': list(set(info['sources']))
                    })
                
                return results
        
        ensemble_model = SimpleEnsemble(retrieval_model, lstm_predictor)
        print(f"✅ Ensemble created!")
        
    except Exception as e:
        print(f"⚠️  Ensemble creation failed: {e}")

# ============================================================
# DETERMINE ACTIVE MODEL
# ============================================================

active_model = None
model_type = None

if ensemble_model:
    active_model = ensemble_model
    model_type = "ensemble"
    print(f"\n🎯 Active model: ENSEMBLE (Retrieval + LSTM)")
elif retrieval_model:
    active_model = retrieval_model
    model_type = "retrieval"
    print(f"\n📚 Active model: RETRIEVAL only")
else:
    print(f"\n❌ NO MODELS LOADED!")
    print(f"\n💡 Please train at least the Retrieval model:")
    print(f"   cd backend/models")
    print(f"   python retrieval.py")

print("="*60)

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    """Main prediction endpoint"""
    
    if request.method == "OPTIONS":
        return "", 200
    
    try:
        # Parse request
        data = request.get_json()
        input_text = data.get("input", "").strip()
        
        if not input_text:
            return jsonify({
                "model": model_type,
                "results": [],
                "error": "Vui lòng nhập văn bản"
            }), 400
        
        print(f"\n📥 Request: '{input_text}'")
        
        # Check model loaded
        if active_model is None:
            return jsonify({
                "model": "none",
                "results": [],
                "error": "Model chưa được load"
            }), 500
        
        # Get predictions
        results = []
        
        try:
            candidates = active_model.predict_multiple(input_text, top_k=5)
            
            for cand in candidates:
                results.append({
                    "text": cand.get('text', ''),
                    "confidence": float(cand.get('confidence', 0.5)),
                    "method": cand.get('model', model_type),
                    "sources": cand.get('sources', [model_type])
                })
            
            print(f"✅ Generated {len(results)} predictions")
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return error response
            results = [{
                "text": f"[Error: {str(e)}]",
                "confidence": 0.0,
                "method": "error",
                "sources": []
            }]
        
        # Return top 3
        response = {
            "model": model_type,
            "results": results[:3]
        }
        
        print(f"📤 Sending {len(response['results'])} results:")
        for i, r in enumerate(response['results'], 1):
            sources = '+'.join(r.get('sources', []))
            print(f"   {i}. [{r['confidence']:.0%}] {r['text'][:50]}... ({sources})")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "model": model_type or "error",
            "results": [],
            "error": str(e)
        }), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    
    status = "healthy" if active_model else "no_model"
    
    return jsonify({
        "status": status,
        "model_type": model_type,
        "models_available": {
            "retrieval": retrieval_model is not None,
            "lstm": lstm_predictor is not None,
            "ensemble": ensemble_model is not None
        },
        "active_model": model_type,
        "ready": active_model is not None
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    """Model statistics endpoint"""
    
    info = {
        "model_type": model_type,
        "retrieval": None,
        "lstm": None
    }
    
    if retrieval_model:
        vocab_size = 0
        if hasattr(retrieval_model, 'vectorizer') and hasattr(retrieval_model.vectorizer, 'vocabulary_'):
            vocab_size = len(retrieval_model.vectorizer.vocabulary_)
        
        info["retrieval"] = {
            "database_size": len(retrieval_model.database),
            "vocab_size": vocab_size,
            "trained": getattr(retrieval_model, 'is_trained', False)
        }
    
    if lstm_predictor:
        info["lstm"] = {
            "vocab_size": lstm_predictor.vocab.n_words,
            "device": str(lstm_predictor.device)
        }
    
    return jsonify(info)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌐 CADAO AUTOCOMPLETE API - READY")
    print("="*60)
    
    if active_model:
        print(f"✅ Active model: {model_type}")
    else:
        print(f"❌ No models loaded - API will return errors")
    
    print(f"\n📌 Endpoints:")
    print(f"   POST /api/predict - Get predictions")
    print(f"   GET  /api/health  - Health check")
    print(f"   GET  /api/stats   - Model statistics")
    print("="*60)
    print(f"🚀 Starting server on http://localhost:5000")
    print("="*60)
    
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
