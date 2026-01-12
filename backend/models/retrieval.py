# Improved retrieval.py with BM25 instead of TF-IDF cosine, and prefix matching boost

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import pickle
from pathlib import Path


class RetrievalModel:
    """
    Improved Retrieval-based model using BM25 scoring with prefix matching boost
    
    Improvements:
    - Replaced TF-IDF cosine similarity with BM25 ranking (better for IR tasks)
    - Added prefix matching: if the sentence starts with the query (word-level), boost score significantly
    - Used char_wb analyzer for better handling of Vietnamese compound words
    - Adjusted ngram_range to (2,4) for balanced lexical matching
    - Better confidence scaling
    - Fallback to random only if no matches above threshold
    
    BM25 advantages over TF-IDF cosine:
    - Handles document length normalization better
    - Saturates TF (term frequency) to prevent long docs dominating
    - Proven better for text retrieval tasks
    
    Prefix boost: For partial completion tasks, exact prefix matches get +10 to score
    """
    
    def __init__(self, analyzer='char_wb', ngram_range=(2, 4), max_features=10000, bm25_k1=1.5, bm25_b=0.75):
        """
        Args:
            analyzer: 'char_wb' for character n-grams within words (good for Vietnamese)
            ngram_range: Character n-grams range
            max_features: Max vocabulary size
            bm25_k1: BM25 TF saturation parameter (1.2-2.0 typical)
            bm25_b: BM25 length normalization (0.75 typical)
        """
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=True,
            strip_accents=None,  # Keep Vietnamese accents
        )
        
        self.database = []  # List of full sentences
        self.term_freqs = None  # Document-term matrix (counts)
        self.idf = None  # IDF values
        self.doc_lengths = None  # Length of each doc in words
        self.avg_doc_len = 0
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.is_trained = False
    
    def train(self, train_data):
        """
        Train model: Build database and compute necessary statistics for BM25
        
        Args:
            train_data: List of dicts [{'full': '...', ...}]
        """
        print(f"\n{'─'*60}")
        print(f"🔄 TRAINING IMPROVED RETRIEVAL MODEL (BM25)")
        print(f"{'─'*60}")
        
        # Get unique sentences
        seen = set()
        for item in train_data:
            sentence = item['full']
            if sentence not in seen:
                self.database.append(sentence)
                seen.add(sentence)
        
        print(f"📊 Database: {len(self.database)} unique sentences")
        
        # Vectorize to get term frequencies (use CountVectorizer internally)
        from sklearn.feature_extraction.text import CountVectorizer
        count_vec = CountVectorizer(
            analyzer=self.vectorizer.analyzer,
            ngram_range=self.vectorizer.ngram_range,
            max_features=self.vectorizer.max_features,
            lowercase=True,
            strip_accents=None,
        )
        self.term_freqs = count_vec.fit_transform(self.database)
        
        # Compute IDF from TF-IDF vectorizer
        self.idf = self.vectorizer.fit(self.database).idf_
        
        # Compute document lengths (number of terms)
        self.doc_lengths = np.array([len(doc.split()) for doc in self.database])
        self.avg_doc_len = np.mean(self.doc_lengths) if len(self.doc_lengths) > 0 else 0
        
        print(f"✓ Term matrix shape: {self.term_freqs.shape}")
        print(f"✓ Vocabulary size: {len(count_vec.vocabulary_):,}")
        print(f"✓ Average doc length: {self.avg_doc_len:.1f} words")
        
        # Top features analysis
        feature_names = count_vec.get_feature_names_out()
        print(f"\n📝 Top 10 features:")
        top_indices = np.argsort(self.idf)[:10]  # Low IDF = common
        for i, idx in enumerate(top_indices, 1):
            print(f"   {i:2d}. '{feature_names[idx]}' (IDF: {self.idf[idx]:.2f})")
        
        self.is_trained = True
    
    def compute_bm25_scores(self, query):
        """
        Compute BM25 scores for all documents given a query
        
        BM25 formula:
        score(d, q) = sum_{t in q} IDF(t) * (TF(t,d) * (k1 + 1)) / (TF(t,d) + k1 * (1 - b + b * len(d)/avg_len))
        """
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        # Get query terms
        query_terms = self.vectorizer.transform([query])  # But we need counts
        from sklearn.feature_extraction.text import CountVectorizer
        count_vec = CountVectorizer(vocabulary=self.vectorizer.vocabulary_)
        query_tf = count_vec.fit_transform([query]).toarray()[0]
        
        scores = np.zeros(len(self.database))
        
        for term_idx in np.nonzero(query_tf)[0]:
            tf_query = query_tf[term_idx]
            if tf_query == 0:
                continue
            
            # TF in docs for this term
            tf_docs = self.term_freqs[:, term_idx].toarray().flatten()
            
            # IDF for term
            idf_term = self.idf[term_idx]
            
            # BM25 term score
            numerator = tf_docs * (self.bm25_k1 + 1)
            denominator = tf_docs + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * (self.doc_lengths / self.avg_doc_len))
            term_scores = idf_term * (numerator / denominator)
            
            scores += term_scores
        
        return scores
    
    def retrieve(self, query, top_k=10):
        """
        Retrieve top-k similar sentences using BM25 with prefix boost
        
        Args:
            query: Input string
            top_k: Number to return
        
        Returns:
            List of (sentence, score) tuples
        """
        # Compute BM25 scores
        scores = self.compute_bm25_scores(query.lower())
        
        # Add prefix matching boost
        query_words = query.lower().split()
        query_len = len(query_words)
        
        for i, sentence in enumerate(self.database):
            sent_words = sentence.lower().split()
            if sent_words[:query_len] == query_words:
                scores[i] += 10.0  # Strong boost for exact prefix match
        
        # Get top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.database[idx], float(scores[idx])))
        
        return results
    
    def predict_multiple(self, partial_input, top_k=3, min_score=0.1):
        """
        Return top-k candidates for API
        
        Args:
            partial_input: Input string
            top_k: Number of candidates
            min_score: Minimum score threshold
        
        Returns:
            List of dicts [{'text': '...', 'confidence': 0.9, 'model': 'retrieval'}]
        """
        retrieved = self.retrieve(partial_input, top_k=top_k*2)
        
        candidates = []
        max_score = max([s[1] for s in retrieved]) if retrieved else 1.0
        
        for sentence, score in retrieved:
            if score < min_score:
                continue
            
            # Normalize score to confidence (0-1)
            confidence = min(0.99, score / max_score) if max_score > 0 else 0.05
            
            candidates.append({
                'text': sentence,
                'confidence': round(confidence, 3),
                'model': 'retrieval',
                'score': round(score, 3)
            })
            
            if len(candidates) >= top_k:
                break
        
        # Fallback if no good matches
        if not candidates:
            import random
            random_sentence = random.choice(self.database) if self.database else partial_input
            candidates = [{
                'text': random_sentence,
                'confidence': 0.05,
                'model': 'retrieval',
                'score': 0.0,
                'method': 'fallback'
            }]
        
        return candidates
    
    def predict(self, partial_input):
        """Return single best result"""
        candidates = self.predict_multiple(partial_input, top_k=1)
        return candidates[0]['text'] if candidates else partial_input
    
    def evaluate(self, test_data):
        """
        Evaluate on test set
        
        Metrics:
        - Exact match accuracy
        - Top-3 accuracy
        - Average score
        """
        print(f"\n{'─'*60}")
        print(f"📊 EVALUATING IMPROVED RETRIEVAL MODEL")
        print(f"{'─'*60}")
        
        exact_correct = 0
        top3_correct = 0
        total = len(test_data)
        scores = []
        
        for item in test_data:
            candidates = self.predict_multiple(item['input'], top_k=3)
            
            if candidates[0]['text'] == item['full']:
                exact_correct += 1
            
            top3_texts = [c['text'] for c in candidates]
            if item['full'] in top3_texts:
                top3_correct += 1
            
            if candidates:
                scores.append(candidates[0]['score'])
        
        exact_acc = exact_correct / total if total > 0 else 0
        top3_acc = top3_correct / total if total > 0 else 0
        avg_score = sum(scores) / len(scores) if scores else 0
        
        print(f"Test samples: {total}")
        print(f"Exact matches: {exact_correct} ({exact_acc:.1%})")
        print(f"Top-3 matches: {top3_correct} ({top3_acc:.1%})")
        print(f"Avg score: {avg_score:.3f}")
        
        return {
            'exact_accuracy': exact_acc,
            'top3_accuracy': top3_acc,
            'avg_score': avg_score,
            'exact_correct': exact_correct,
            'top3_correct': top3_correct,
            'total': total
        }
    
    def save(self, file_path):
        """Save model"""
        data = {
            'vectorizer': self.vectorizer,
            'database': self.database,
            'term_freqs': self.term_freqs,
            'idf': self.idf,
            'doc_lengths': self.doc_lengths,
            'avg_doc_len': self.avg_doc_len,
            'bm25_k1': self.bm25_k1,
            'bm25_b': self.bm25_b
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Model saved to {file_path}")
    
    def load(self, file_path):
        """Load model"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.vectorizer = data['vectorizer']
        self.database = data['database']
        self.term_freqs = data['term_freqs']
        self.idf = data['idf']
        self.doc_lengths = data['doc_lengths']
        self.avg_doc_len = data['avg_doc_len']
        self.bm25_k1 = data['bm25_k1']
        self.bm25_b = data['bm25_b']
        self.is_trained = True
        
        print(f"✓ Model loaded from {file_path}")


# ========== SCRIPT TRAINING ==========
def train_retrieval_model():
    """Script to train and test model"""
    
    print("\n" + "="*70)
    print("🚀 IMPROVED RETRIEVAL MODEL TRAINING")
    print("="*70)
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "processed"
    MODEL_DIR = BASE_DIR / "trained_models"
    
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Load data
    print(f"\n📂 Loading data...")
    
    with open(DATA_DIR / "train.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(DATA_DIR / "test.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"✓ Train: {len(train_data)} samples")
    print(f"✓ Test:  {len(test_data)} samples")
    
    # Train model
    model = RetrievalModel(analyzer='char_wb', ngram_range=(2, 4), max_features=10000)
    model.train(train_data)
    
    # Test predictions
    print(f"\n{'─'*60}")
    print("🧪 TEST PREDICTIONS")
    print(f"{'─'*60}")
    
    test_inputs = [
        "ăn",
        "ăn quả",
        "ăn quả nhớ",
        "có công",
        "có công mài sắt",
        "gần mực",
        "học thầy không"
    ]
    
    for inp in test_inputs:
        print(f"\n📝 Input: '{inp}'")
        candidates = model.predict_multiple(inp, top_k=3)
        
        for i, cand in enumerate(candidates, 1):
            print(f"   {i}. {cand['text']}")
            print(f"      📊 Confidence: {cand['confidence']:.1%} | Score: {cand['score']:.3f}")
    
    # Evaluate
    metrics = model.evaluate(test_data[:100])
    
    # Save model
    model_path = MODEL_DIR / "improved_retrieval_model.pkl"
    model.save(model_path)
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   • Database size: {len(model.database):,} sentences")
    print(f"   • Vector dimension: {model.term_freqs.shape[1]:,}")
    print(f"   • Exact accuracy: {metrics['exact_accuracy']:.1%}")
    print(f"   • Top-3 accuracy: {metrics['top3_accuracy']:.1%}")
    print(f"   • Avg score: {metrics['avg_score']:.3f}")
    print(f"   • Model saved: {model_path}")
    print()


# ========== MAIN ==========
if __name__ == "__main__":
    train_retrieval_model()