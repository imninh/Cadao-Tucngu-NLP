from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict, Counter
import pickle
import os
import sys
import importlib.util

app = Flask(__name__)
CORS(app)

# Import NgramModel - Đường dẫn ĐÚNG
NgramModel = None
try:
    # Lấy thư mục hiện tại (backend/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ngram_path = os.path.join(current_dir, "models", "ngram.py")
    
    print(f"🔍 Looking for ngram.py at: {ngram_path}")
    print(f"🔍 File exists: {os.path.exists(ngram_path)}")
    
    if os.path.exists(ngram_path):
        # Load module từ file path
        spec = importlib.util.spec_from_file_location("ngram", ngram_path)
        ngram_module = importlib.util.module_from_spec(spec)
        sys.modules["ngram"] = ngram_module
        spec.loader.exec_module(ngram_module)
        
        # Lấy class NgramModel
        NgramModel = ngram_module.NgramModel
        print("✅ Imported NgramModel class successfully")
    else:
        print("❌ ngram.py file not found")
        
except Exception as e:
    print(f"❌ Cannot import NgramModel: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
print("🚀 LOADING N-GRAM MODEL API")
print("=" * 60)

# ✅ FIX: Đường dẫn model ĐÚNG (trong backend/)
current_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = [
    os.path.join(current_dir, "trained_models", "improved_ngram_model.pkl"),
    os.path.join(current_dir, "trained_models", "ngram_model.pkl")
]

ngram_model = None
model_path_used = None

# Thử load từ các đường dẫn
for model_path in MODEL_PATHS:
    print(f"🔍 Checking: {model_path}")
    print(f"   Exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        try:
            print(f"📦 Loading model from: {model_path}")
            
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            
            print(f"📊 Loaded data type: {type(model_data)}")
            
            # Nếu là dict (đã save từ NgramModel.save())
            if isinstance(model_data, dict):
                print("🔄 Reconstructing NgramModel from saved data...")
                
                # KIỂM TRA NgramModel có tồn tại không
                if NgramModel is None:
                    print("❌ Cannot reconstruct - NgramModel class not imported")
                    continue
                
                # Tạo instance mới
                ngram_model = NgramModel(
                    n=model_data.get('n', 4),
                    alpha=model_data.get('alpha', 0.1)
                )
                
                # Khôi phục dữ liệu
                ngram_model.counts = {}
                for k, v in model_data['counts'].items():
                    ngram_model.counts[k] = defaultdict(Counter)
                    for context, counter in v.items():
                        ngram_model.counts[k][context] = Counter(counter)
                
                ngram_model.vocab = set(model_data['vocab'])
                ngram_model.full_sentences = model_data.get('full_sentences', [])
                ngram_model.total_ngrams = model_data.get('total_ngrams', 0)
                
                print("✅ Successfully reconstructed NgramModel")
                
            # Nếu đã là instance của NgramModel
            elif NgramModel and isinstance(model_data, NgramModel):
                print("✅ Loaded NgramModel instance directly")
                ngram_model = model_data
                
            else:
                print(f"⚠️ Unknown model type: {type(model_data)}")
                ngram_model = model_data
            
            model_path_used = model_path
            break
            
        except Exception as e:
            print(f"❌ Error loading {model_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

if ngram_model is None:
    print("❌ No model could be loaded!")
else:
    print(f"✅ Model loaded successfully from: {model_path_used}")
    print(f"📊 Model type: {type(ngram_model)}")
    
    # Kiểm tra methods
    print("🔍 Checking available methods:")
    methods = ['predict_multiple', 'predict', 'predict_next_word', 'get_matching_candidates']
    for method in methods:
        if hasattr(ngram_model, method):
            print(f"   ✅ {method}() is available")
        else:
            print(f"   ❌ {method}() is NOT available")

@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    """API endpoint cho N-gram model prediction"""
    
    if request.method == "OPTIONS":
        return "", 200
    
    try:
        data = request.get_json()
        input_text = data.get("input", "").strip()
        
        if not input_text:
            return jsonify({
                "model": "ngram",
                "results": [],
                "error": "Vui lòng nhập văn bản"
            }), 400
        
        print(f"📥 Received request: '{input_text}'")
        
        if ngram_model is None:
            return jsonify({
                "model": "ngram", 
                "results": [],
                "error": "Model chưa được load"
            }), 500
        
        results = []
        
        # Thử các phương thức predict
        if hasattr(ngram_model, 'predict_multiple'):
            print("🔄 Using predict_multiple()...")
            try:
                candidates = ngram_model.predict_multiple(input_text, top_k=5)
                
                for cand in candidates:
                    if isinstance(cand, dict):
                        results.append({
                            "text": cand.get('text', ''),
                            "confidence": float(cand.get('confidence', 0.5)),
                            "method": cand.get('method', 'ngram')
                        })
                    else:
                        results.append({
                            "text": str(cand),
                            "confidence": 0.7,
                            "method": "ngram"
                        })
                        
            except Exception as e:
                print(f"❌ predict_multiple failed: {e}")
        
        if not results and hasattr(ngram_model, 'get_matching_candidates'):
            print("🔄 Using get_matching_candidates()...")
            try:
                candidates = ngram_model.get_matching_candidates(input_text, top_k=5)
                for cand in candidates:
                    results.append({
                        "text": cand.get('text', ''),
                        "confidence": float(cand.get('confidence', 0.5)),
                        "method": cand.get('method', 'ngram')
                    })
            except Exception as e:
                print(f"❌ get_matching_candidates failed: {e}")
        
        if not results and hasattr(ngram_model, 'predict'):
            print("🔄 Using predict()...")
            try:
                prediction = ngram_model.predict(input_text)
                results.append({
                    "text": prediction,
                    "confidence": 0.8,
                    "method": "ngram"
                })
            except Exception as e:
                print(f"❌ predict failed: {e}")
        
        # Manual prediction từ counts
        if not results and hasattr(ngram_model, 'counts') and hasattr(ngram_model, 'n'):
            print("🔄 Manual prediction from counts...")
            try:
                words = input_text.split()
                
                for k in range(min(ngram_model.n, len(words) + 1), 0, -1):
                    if len(words) >= k - 1:
                        context = tuple(words[-(k - 1):]) if k > 1 else ()
                        
                        if k in ngram_model.counts and context in ngram_model.counts[k]:
                            counter = ngram_model.counts[k][context]
                            top_words = counter.most_common(3)
                            
                            for word, count in top_words:
                                full_text = input_text + " " + word
                                confidence = count / sum(counter.values()) if sum(counter.values()) > 0 else 0.5
                                results.append({
                                    "text": full_text,
                                    "confidence": float(confidence),
                                    "method": f"ngram_order_{k}"
                                })
                            break
            except Exception as e:
                print(f"❌ Manual prediction failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Fallback
        if not results:
            print("⚠️ No prediction methods worked, using fallback")
            results = [{
                "text": f"{input_text} [Dự đoán từ mô hình N-gram]",
                "confidence": 0.5,
                "method": "fallback"
            }]
        
        response_data = {
            "model": "ngram",
            "results": results[:3]
        }
        
        print(f"📤 Sending {len(response_data['results'])} results")
        for i, result in enumerate(response_data['results']):
            print(f"   {i+1}. {result['text']} (conf: {result['confidence']:.2f})")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "model": "ngram",
            "results": [],
            "error": f"Server error: {str(e)}"
        }), 500

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if ngram_model else "no_model",
        "model_loaded": ngram_model is not None,
        "model_type": str(type(ngram_model)) if ngram_model else None,
        "endpoints": {
            "predict": "/api/predict",
            "health": "/api/health"
        }
    })

if __name__ == "__main__":
    print("=" * 60)
    print("🌐 N-gram Language Model API - READY")
    print("=" * 60)
    print("📌 Available endpoints:")
    print("   POST /api/predict - Predict text completions")
    print("   GET  /api/health  - Health check")
    print("=" * 60)
    print(f"🔧 Model status: {'✅ LOADED' if ngram_model else '❌ NOT LOADED'}")
    if ngram_model:
        print(f"🔧 Source: {model_path_used}")
    print("=" * 60)
    print("🚀 Server starting on http://localhost:5000")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=5000, debug=True)