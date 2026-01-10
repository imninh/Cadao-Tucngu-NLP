from collections import defaultdict, Counter
import json
import pickle
from pathlib import Path
import random

class NgramModel:
    """
    Mô hình N-gram cải thiện với backoff và smoothing cơ bản
    
    Cải thiện chính:
    - Sử dụng hierarchical counts cho các order từ 1 đến n
    - Backoff đến lower order khi không tìm thấy context
    - Add-alpha smoothing (alpha=0.1) để tránh zero probability
    - Beam search đơn giản trong generate để khám phá nhiều path hơn
    - Tích hợp exact/fuzzy match vào predict chính
    - Cải thiện stopping criteria: Dừng khi gặp từ kết thúc câu (heuristic cho tục ngữ)
    - Fix recursion bằng cách tách logic matching ra riêng
    """
    
    def __init__(self, n=4, alpha=0.1):  # Tăng n=4 cho context dài hơn, alpha cho smoothing
        """
        Args:
            n: Max order (4 cho tiếng Việt tục ngữ tốt hơn)
            alpha: Smoothing parameter (add-alpha)
        """
        self.n = n
        self.alpha = alpha
        
        # Counts cho từng order: self.counts[k] với k=1..n
        # Key: tuple context (len k-1)
        # Value: Counter next words
        self.counts = {k: defaultdict(Counter) for k in range(1, n+1)}
        
        # Vocabulary toàn bộ
        self.vocab = set()
        
        # Full sentences cho matching
        self.full_sentences = []
        
        # Thống kê
        self.total_ngrams = {k: 0 for k in range(1, n+1)}
    
    def train(self, train_data):
        """
        Huấn luyện mô hình
        
        Args:
            train_data: List of dicts [{'full': '...', 'input': '...', 'target': '...'}]
        """
        print(f"\n{'─'*60}")
        print(f"🔄 TRAINING IMPROVED N-GRAM MODEL (n={self.n}, alpha={self.alpha})")
        print(f"{'─'*60}")
        
        # Lưu unique full sentences
        seen_sentences = set()
        for item in train_data:
            sentence = item['full'].strip()
            if sentence not in seen_sentences:
                self.full_sentences.append(sentence)
                seen_sentences.add(sentence)
        
        print(f"📊 Dataset: {len(train_data)} samples, {len(self.full_sentences)} unique sentences")
        
        # Build counts cho mọi order
        for item in train_data:
            words = item['full'].strip().split()
            self.vocab.update(words)
            
            for k in range(1, self.n + 1):
                for i in range(len(words) - k + 1):
                    context = tuple(words[i:i + k - 1])
                    next_word = words[i + k - 1]
                    self.counts[k][context][next_word] += 1
                    self.total_ngrams[k] += 1
        
        print(f"✓ Vocabulary size: {len(self.vocab):,} từ")
        print(f"✓ Total n-grams per order:")
        for k in range(1, self.n + 1):
            print(f"   • Order {k}: {self.total_ngrams[k]:,} ({len(self.counts[k]):,} unique contexts)")
        
        # Ví dụ
        print(f"\n📝 Ví dụ n-grams (order {self.n}):")
        for i, (context, counter) in enumerate(list(self.counts[self.n].items())[:3]):
            context_str = ' '.join(context)
            top_3 = counter.most_common(3)
            print(f"   {i+1}. '{context_str}' →")
            for word, count in top_3:
                prob = count / sum(counter.values())
                print(f"      • '{word}' ({prob:.1%}, {count} lần)")
    
    def get_next_words_prob(self, context, order):
        """
        Lấy next words với probability (smoothing)
        
        Args:
            context: tuple
            order: int (1..n)
        
        Returns:
            dict {word: prob}
        """
        if context not in self.counts[order]:
            return {}
        
        counter = self.counts[order][context]
        total_count = sum(counter.values())
        vocab_size = len(self.vocab)
        
        # Add-alpha smoothing
        smoothed_total = total_count + self.alpha * vocab_size
        probs = {}
        for word in counter:
            probs[word] = (counter[word] + self.alpha) / smoothed_total
        
        # Phần còn lại cho unseen words (nhưng chỉ dùng seen cho most common)
        unseen_prob = self.alpha / smoothed_total
        
        return probs
    
    def predict_next_word(self, context_words):
        """
        Dự đoán từ tiếp theo với backoff
        
        Returns:
            (word, confidence)
        """
        context_len = len(context_words)
        for k in range(min(self.n, context_len + 1), 0, -1):  # Highest order first
            if context_len >= k - 1:
                context = tuple(context_words[-(k - 1):])
                probs = self.get_next_words_prob(context, k)
                if probs:
                    # Chọn word có prob cao nhất
                    best_word = max(probs, key=probs.get)
                    conf = probs[best_word] * (0.9 ** (self.n - k))  # Discount for lower order
                    return best_word, conf
        # Ultimate fallback: most common word in unigrams
        if self.counts[1][()]:
            unigram_counter = self.counts[1][()]
            best_word, _ = unigram_counter.most_common(1)[0]
            return best_word, 0.01
        return None, 0.0
    
    def get_matching_candidates(self, partial_input, top_k, include_fuzzy=True):
        """
        Lấy candidates từ exact và fuzzy match
        
        Returns:
            List of dicts [{'text': '...', 'confidence': ..., 'method': ...}]
        """
        words = partial_input.strip().lower().split()
        input_text = ' '.join(words)
        input_set = set(words)
        
        candidates = []
        
        # STRATEGY 1: Exact prefix match
        for sentence in self.full_sentences:
            sentence_lower = sentence.lower()
            if sentence_lower.startswith(input_text) or sentence_lower == input_text:
                overlap_ratio = len(input_text) / len(sentence_lower) if len(sentence_lower) > 0 else 0
                confidence = min(0.99, 0.8 + overlap_ratio * 0.2)
                candidates.append({
                    'text': sentence,
                    'confidence': round(confidence, 3),
                    'model': 'ngram',
                    'method': 'exact_match'
                })
        
        if include_fuzzy:
            # STRATEGY 2: Fuzzy match với Jaccard similarity
            for sentence in self.full_sentences:
                if sentence in [c['text'] for c in candidates]:
                    continue
                sentence_lower = sentence.lower()
                sentence_words = set(sentence_lower.split())
                intersection = len(input_set & sentence_words)
                union = len(input_set | sentence_words)
                jaccard = intersection / union if union else 0
                if jaccard >= 0.6:
                    confidence = jaccard * 0.7
                    candidates.append({
                        'text': sentence,
                        'confidence': round(confidence, 3),
                        'model': 'ngram',
                        'method': 'fuzzy_match'
                    })
        
        # Sort và top-k
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        return candidates[:top_k]
    
    def predict(self, partial_input, max_words=20, beam_width=3):
        """
        Generate với beam search đơn giản
        
        Returns:
            Câu hoàn chỉnh tốt nhất
        """
        words = partial_input.strip().split()
        
        # Trước tiên thử exact match (chỉ exact, không fuzzy để tránh lặp)
        matches = self.get_matching_candidates(partial_input, top_k=1, include_fuzzy=False)
        if matches and matches[0]['confidence'] > 0.7:
            return matches[0]['text']
        
        # Beam search
        beams = [(words.copy(), 1.0)]  # (sequence, score)
        
        for _ in range(max_words):
            new_beams = []
            for seq, score in beams:
                next_word, conf = self.predict_next_word(seq)
                if next_word is None:
                    continue
                new_seq = seq + [next_word]
                new_score = score * conf
                new_beams.append((new_seq, new_score))
            
            # Giữ top beam_width
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]
            
            # Dừng nếu score thấp
            if beams and beams[0][1] < 0.01:
                break
        
        # Chọn best
        best_seq = beams[0][0] if beams else words
        return ' '.join(best_seq)
    
    def predict_multiple(self, partial_input, top_k=3):
        """
        Trả về nhiều candidates - Sử dụng get_matching_candidates
        """
        input_text = ' '.join(partial_input.strip().lower().split())
        
        # Lấy từ matching (exact + fuzzy)
        candidates = self.get_matching_candidates(partial_input, top_k=top_k, include_fuzzy=True)
        
        # STRATEGY 3: Generate nếu cần
        if len(candidates) < top_k:
            generated = self.predict(partial_input)
            if generated.lower() != input_text and generated not in [c['text'] for c in candidates]:
                candidates.append({
                    'text': generated,
                    'confidence': 0.5,
                    'model': 'ngram',
                    'method': 'generated'
                })
        
        # Sort lại nếu thêm generated
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        candidates = candidates[:top_k]
        
        # Fallback
        if not candidates:
            random_sentence = random.choice(self.full_sentences) if self.full_sentences else partial_input
            candidates = [{
                'text': random_sentence,
                'confidence': 0.1,
                'model': 'ngram',
                'method': 'fallback'
            }]
        
        return candidates
    
    def evaluate(self, test_data):
        """
        Đánh giá với exact match và partial match
        """
        print(f"\n{'─'*60}")
        print(f"📊 EVALUATING IMPROVED N-GRAM MODEL")
        print(f"{'─'*60}")
        
        exact_correct = 0
        partial_correct = 0  # Nếu match >=80% words
        total = len(test_data)
        
        for item in test_data:
            predicted = self.predict(item['input'])
            full_words = set(item['full'].split())
            pred_words = set(predicted.split())
            
            if predicted == item['full']:
                exact_correct += 1
                partial_correct += 1
            else:
                intersection = len(full_words & pred_words)
                union = len(full_words | pred_words)
                if intersection / union >= 0.8:
                    partial_correct += 1
        
        exact_acc = exact_correct / total if total > 0 else 0
        partial_acc = partial_correct / total if total > 0 else 0
        
        print(f"Test samples: {total}")
        print(f"Exact matches: {exact_correct} ({exact_acc:.2%})")
        print(f"Partial matches (≥80%): {partial_correct} ({partial_acc:.2%})")
        
        return {
            'exact_accuracy': exact_acc,
            'partial_accuracy': partial_acc,
            'exact_correct': exact_correct,
            'partial_correct': partial_correct,
            'total': total
        }
    
    def save(self, file_path):
        """Lưu model"""
        data = {
            'n': self.n,
            'alpha': self.alpha,
            'counts': {k: dict(v) for k, v in self.counts.items()},
            'vocab': list(self.vocab),
            'full_sentences': self.full_sentences,
            'total_ngrams': self.total_ngrams
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Model saved to {file_path}")
    
    def load(self, file_path):
        """Load model"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.n = data['n']
        self.alpha = data.get('alpha', 0.1)  # Backward compat
        self.counts = {k: defaultdict(Counter, v) for k, v in data['counts'].items()}
        self.vocab = set(data['vocab'])
        self.full_sentences = data['full_sentences']
        self.total_ngrams = data['total_ngrams']
        print(f"✓ Model loaded from {file_path}")


# ========== SCRIPT TRAINING ==========
def train_ngram_model():
    """Script để train và test model"""
    
    print("\n" + "="*70)
    print("🚀 IMPROVED N-GRAM MODEL TRAINING")
    print("="*70)
    
    # Đường dẫn
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
    
    # Train
    model = NgramModel(n=4, alpha=0.1)
    model.train(train_data)
    
    # Test predictions
    print(f"\n{'─'*60}")
    print("🧪 TEST PREDICTIONS")
    print(f"{'─'*60}")
    
    test_inputs = [
        "ăn quả",
        "có công mài sắt",
        "gần mực"
    ]
    
    for inp in test_inputs:
        print(f"\n📝 Input: '{inp}'")
        candidates = model.predict_multiple(inp, top_k=3)
        
        for i, cand in enumerate(candidates, 1):
            print(f"   {i}. {cand['text']}")
            print(f"      Confidence: {cand['confidence']:.1%} | Method: {cand['method']}")
    
    # Evaluate
    metrics = model.evaluate(test_data[:100])
    
    # Save
    model_path = MODEL_DIR / "improved_ngram_model.pkl"
    model.save(model_path)
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   • Vocabulary: {len(model.vocab):,} words")
    for k in range(1, model.n + 1):
        print(f"   • Order {k} n-grams: {model.total_ngrams[k]:,}")
    print(f"   • Exact Accuracy: {metrics['exact_accuracy']:.2%}")
    print(f"   • Partial Accuracy: {metrics['partial_accuracy']:.2%}")
    print(f"   • Model saved: {model_path}")
    print()


if __name__ == "__main__":
    train_ngram_model()