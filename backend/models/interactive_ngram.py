"""
INTERACTIVE N-GRAM AUTOCOMPLETE
Nhập input trên terminal → Nhận suggestions từ improved N-gram model

Usage:
    python interactive_ngram.py
    
    >>> Nhập: có công
    
    Suggestions:
    1. [95% 🟢] có công mài sắt có ngày nên kim (exact_match)
    2. [70% 🟡] công cha như núi thái sơn (fuzzy_match)
    3. [50% 🟡] thất bại là mẹ thành công (generated)
"""

import sys
from pathlib import Path
from collections import Counter
import json


class InteractiveNgramAutocomplete:
    """
    Interactive autocomplete sử dụng N-gram model
    """
    
    def __init__(self, ngram_model):
        """
        Args:
            ngram_model: NgramModel instance đã train
        """
        self.model = ngram_model
        self.session_stats = {
            'queries': 0,
            'successful': 0,
            'methods': Counter(),
            'avg_confidence': []
        }
        
        print(f"✓ N-gram model loaded")
        print(f"   • Order: {self.model.n}")
        print(f"   • Vocabulary: {len(self.model.vocab):,} words")
        print(f"   • Database: {len(self.model.full_sentences)} sentences")
    
    def get_suggestions(self, input_text, top_k=5):
        """
        Get autocomplete suggestions từ N-gram model
        
        Returns:
            List of {'text': ..., 'confidence': ..., 'method': ...}
        """
        
        if not input_text.strip():
            # Empty input → random popular sentences
            import random
            popular = random.sample(
                self.model.full_sentences, 
                min(top_k, len(self.model.full_sentences))
            )
            return [{
                'text': sent,
                'confidence': 0.3,
                'method': 'popular'
            } for sent in popular]
        
        # Use model's predict_multiple
        candidates = self.model.predict_multiple(input_text, top_k=top_k)
        
        return candidates
    
    def display_suggestions(self, suggestions):
        """Display suggestions với formatting đẹp"""
        
        if not suggestions:
            print("   ⚠️  Không tìm thấy câu phù hợp")
            return
        
        print(f"\n   📋 Suggestions:")
        
        for i, sugg in enumerate(suggestions, 1):
            conf = sugg['confidence']
            method = sugg.get('method', 'unknown')
            
            # Confidence color
            if conf >= 0.8:
                icon = "🟢"
                conf_label = "HIGH"
            elif conf >= 0.6:
                icon = "🟡"
                conf_label = "MED "
            else:
                icon = "🔴"
                conf_label = "LOW "
            
            # Method icon
            method_icons = {
                'exact_match': '🎯',
                'fuzzy_match': '🔍',
                'generated': '⚙️',
                'popular': '⭐',
                'fallback': '💫'
            }
            method_icon = method_icons.get(method, '❓')
            
            print(f"   {i}. {icon} [{conf*100:.0f}% {conf_label}] {method_icon} {sugg['text']}")
            print(f"      └─ Method: {method}")
    
    def show_detailed_info(self, input_text):
        """Show thông tin chi tiết về prediction process"""
        
        print(f"\n{'─'*70}")
        print(f"🔍 DETAILED ANALYSIS")
        print(f"{'─'*70}")
        
        words = input_text.strip().split()
        print(f"\n📝 Input: '{input_text}'")
        print(f"   • Words: {words}")
        print(f"   • Length: {len(words)} words")
        
        # Show n-gram context matching
        print(f"\n🔢 N-gram Context Matching:")
        for k in range(min(self.model.n, len(words) + 1), 0, -1):
            if len(words) >= k - 1:
                context = tuple(words[-(k - 1):])
                context_str = ' '.join(context) if context else '<START>'
                
                probs = self.model.get_next_words_prob(context, k)
                
                if probs:
                    print(f"   Order {k} ('{context_str}'):")
                    top_words = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    for word, prob in top_words:
                        print(f"      • '{word}' → {prob:.1%}")
                else:
                    print(f"   Order {k} ('{context_str}'): No matches")
        
        # Show matching sentences
        print(f"\n🎯 Matching Sentences:")
        matches = self.model.get_matching_candidates(input_text, top_k=5, include_fuzzy=True)
        
        exact_matches = [m for m in matches if m['method'] == 'exact_match']
        fuzzy_matches = [m for m in matches if m['method'] == 'fuzzy_match']
        
        if exact_matches:
            print(f"   Exact matches: {len(exact_matches)}")
            for m in exact_matches[:3]:
                print(f"      • [{m['confidence']:.0%}] {m['text']}")
        else:
            print(f"   Exact matches: None")
        
        if fuzzy_matches:
            print(f"   Fuzzy matches: {len(fuzzy_matches)}")
            for m in fuzzy_matches[:3]:
                print(f"      • [{m['confidence']:.0%}] {m['text']}")
        else:
            print(f"   Fuzzy matches: None")
        
        print(f"{'─'*70}")
    
    def run_interactive(self):
        """Run interactive terminal session"""
        
        print("\n" + "="*70)
        print("🎯 INTERACTIVE N-GRAM AUTOCOMPLETE - Ca Dao & Tục Ngữ")
        print("="*70)
        print("\n📖 Instructions:")
        print("   • Nhập một phần câu ca dao/tục ngữ")
        print("   • N-gram model sẽ gợi ý các câu hoàn chỉnh")
        print("   • Gõ 'q' hoặc 'quit' để thoát")
        print("   • Gõ 'stats' để xem thống kê")
        print("   • Gõ 'detail:<text>' để xem phân tích chi tiết")
        print("   • Gõ 'help' để xem hướng dẫn")
        print("\n" + "-"*70)
        
        while True:
            try:
                # Get input
                print("\n" + "─"*70)
                user_input = input(">>> Nhập: ").strip()
                
                # Commands
                if user_input.lower() in ['q', 'quit', 'exit']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'stats':
                    self._show_stats()
                    continue
                
                if user_input.lower().startswith('detail:'):
                    text = user_input[7:].strip()
                    if text:
                        self.show_detailed_info(text)
                    else:
                        print("   ⚠️  Usage: detail:<text>")
                    continue
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if user_input.lower() == 'model':
                    self._show_model_info()
                    continue
                
                if not user_input:
                    print("   ⚠️  Vui lòng nhập text")
                    continue
                
                # Get suggestions
                self.session_stats['queries'] += 1
                suggestions = self.get_suggestions(user_input, top_k=5)
                
                if suggestions:
                    self.session_stats['successful'] += 1
                    for sugg in suggestions:
                        self.session_stats['methods'][sugg.get('method', 'unknown')] += 1
                        self.session_stats['avg_confidence'].append(sugg['confidence'])
                
                # Display
                self.display_suggestions(suggestions)
                
                # Show tip
                if self.session_stats['queries'] == 1:
                    print(f"\n   💡 Tip: Gõ 'detail:{user_input}' để xem phân tích chi tiết")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final stats
        print("\n" + "="*70)
        self._show_stats()
        print("="*70)
    
    def _show_stats(self):
        """Show session statistics"""
        
        stats = self.session_stats
        
        print("\n📊 Session Statistics:")
        print(f"   Total queries: {stats['queries']}")
        
        if stats['queries'] > 0:
            success_rate = stats['successful'] / stats['queries'] * 100
            print(f"   Successful: {stats['successful']} ({success_rate:.1f}%)")
        
        if stats['avg_confidence']:
            avg_conf = sum(stats['avg_confidence']) / len(stats['avg_confidence'])
            print(f"   Average confidence: {avg_conf:.1%}")
        
        if stats['methods']:
            print(f"\n   Methods used:")
            for method, count in stats['methods'].most_common():
                pct = count / sum(stats['methods'].values()) * 100
                print(f"      • {method}: {count} ({pct:.1f}%)")
    
    def _show_help(self):
        """Show help message"""
        
        print("\n📖 Available Commands:")
        print("   • <text>           : Get suggestions for text")
        print("   • detail:<text>    : Show detailed analysis")
        print("   • stats            : Show session statistics")
        print("   • model            : Show model information")
        print("   • help             : Show this help message")
        print("   • q / quit / exit  : Exit program")
        
        print("\n💡 Examples:")
        print("   >>> ăn quả")
        print("   >>> detail:ăn quả")
        print("   >>> có công mài sắt")
    
    def _show_model_info(self):
        """Show model information"""
        
        print("\n📊 Model Information:")
        print(f"   • Model type: Improved N-gram with backoff & smoothing")
        print(f"   • Max order: {self.model.n}")
        print(f"   • Smoothing: Add-alpha (α={self.model.alpha})")
        print(f"   • Vocabulary size: {len(self.model.vocab):,} words")
        print(f"   • Database size: {len(self.model.full_sentences)} sentences")
        
        print(f"\n   N-grams counts:")
        for k in range(1, self.model.n + 1):
            total = self.model.total_ngrams[k]
            unique = len(self.model.counts[k])
            print(f"      • Order {k}: {total:,} total, {unique:,} unique contexts")
        
        print(f"\n   Strategies:")
        print(f"      1. Exact prefix match (highest confidence)")
        print(f"      2. Fuzzy match with Jaccard similarity")
        print(f"      3. Beam search generation")
        print(f"      4. Backoff to lower order n-grams")


# ========== MAIN ==========
def main():
    """Main entry point"""
    
    # Setup paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "processed"
    MODEL_DIR = BASE_DIR / "trained_models"
    
    # Check for model file
    model_path = MODEL_DIR / "improved_ngram_model.pkl"
    
    if not model_path.exists():
        print("❌ Model file not found!")
        print(f"   Looking for: {model_path}")
        print("\n💡 Please train the model first:")
        print("   python backend/models/ngram.py")
        return
    
    # Load model
    print("🔄 Loading N-gram model...")
    
    try:
        # Import NgramModel from the document
        import sys
        sys.path.append(str(BASE_DIR / "models"))
        
        from ngram import NgramModel
        
        model = NgramModel()
        model.load(model_path)
        
        print("✓ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Make sure you have trained the improved N-gram model:")
        print("   python backend/models/ngram.py")
        return
    
    # Create interactive autocomplete
    autocomplete = InteractiveNgramAutocomplete(model)
    
    # Run interactive mode
    autocomplete.run_interactive()


# ========== QUICK TEST MODE ==========
def quick_test():
    """Quick test without trained model"""
    
    print("\n" + "="*70)
    print("🧪 QUICK TEST MODE (Demo Data)")
    print("="*70)
    print("\n⚠️  Note: Using demo data. For full functionality, train the model first.")
    
    # Create minimal model for demo
    from collections import defaultdict, Counter
    
    class DemoNgramModel:
        def __init__(self):
            self.n = 3
            self.alpha = 0.1
            self.vocab = set()
            self.full_sentences = [
                "ăn quả nhớ kẻ trồng cây",
                "có công mài sắt có ngày nên kim",
                "gần mực thì đen gần đèn thì sáng",
                "học thầy không tày học bạn",
                "ăn cháo đá bát",
                "công cha như núi thái sơn",
                "học ăn học nói học gói học mở",
                "tiên học lễ hậu học văn",
                "xa thơm gần thối",
                "uống nước nhớ nguồn",
            ]
            self.counts = {k: defaultdict(Counter) for k in range(1, 4)}
            self.total_ngrams = {1: 0, 2: 0, 3: 0}
            
            # Build simple counts
            for sent in self.full_sentences:
                words = sent.split()
                self.vocab.update(words)
        
        def get_next_words_prob(self, context, order):
            return {}
        
        def get_matching_candidates(self, input_text, top_k, include_fuzzy):
            words = input_text.lower().strip().split()
            input_text_lower = ' '.join(words)
            candidates = []
            
            for sent in self.full_sentences:
                sent_lower = sent.lower()
                if sent_lower.startswith(input_text_lower):
                    conf = 0.9
                    candidates.append({
                        'text': sent,
                        'confidence': conf,
                        'method': 'exact_match'
                    })
            
            return candidates[:top_k]
        
        def predict_multiple(self, input_text, top_k):
            return self.get_matching_candidates(input_text, top_k, True)
    
    model = DemoNgramModel()
    
    # Test
    test_cases = ["ăn", "có công", "gần mực", "học"]
    
    for test_input in test_cases:
        print(f"\n{'─'*70}")
        print(f">>> Nhập: {test_input}")
        
        candidates = model.predict_multiple(test_input, top_k=3)
        
        if candidates:
            print(f"\n   📋 Suggestions:")
            for i, cand in enumerate(candidates, 1):
                print(f"   {i}. 🟢 [{cand['confidence']*100:.0f}%] {cand['text']}")
        else:
            print("   ⚠️  Không tìm thấy")
    
    print("\n" + "="*70)
    print("✅ Quick test complete!")
    print("\nTo run full interactive mode:")
    print("   1. Train model: python backend/models/ngram.py")
    print("   2. Run: python interactive_ngram.py")
    print("="*70)


if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        main()