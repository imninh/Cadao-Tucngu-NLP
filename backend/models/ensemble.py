"""
INTERACTIVE TERMINAL AUTOCOMPLETE
Nhập input trên terminal → Nhận suggestions realtime

Usage:
    python interactive_autocomplete.py
    
    >>> Nhập: có công
    
    Suggestions:
    1. [95% 🟢] có công mài sắt có ngày nên kim
    2. [75% 🟡] công cha như núi thái sơn
    3. [60% 🟡] thất bại là mẹ thành công
    
    >>> Nhập: q (để quit)
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import json


class InteractiveAutocomplete:
    """
    Interactive autocomplete system for terminal
    """
    
    def __init__(self, database):
        """
        Args:
            database: List of full sentences
        """
        self.database = database
        
        # Build indexes
        self.prefix_index = defaultdict(list)
        self.word_index = defaultdict(list)
        self.sentence_freq = Counter()
        
        self._build_indexes()
        
        print(f"✓ Loaded {len(self.database)} sentences")
    
    def _build_indexes(self):
        """Build fast lookup indexes"""
        
        for sentence in self.database:
            words = sentence.lower().split()
            
            # Prefix index (1-6 words)
            for i in range(1, min(len(words) + 1, 7)):
                prefix = ' '.join(words[:i])
                self.prefix_index[prefix].append(sentence)
            
            # Word index
            for word in words:
                self.word_index[word].append(sentence)
            
            # Frequency (giả sử tất cả bằng nhau)
            self.sentence_freq[sentence] = 1
    
    def fuzzy_similarity(self, input_text, sentence):
        """Calculate fuzzy match score"""
        input_lower = input_text.lower()
        sentence_lower = sentence.lower()
        
        # Exact prefix
        if sentence_lower.startswith(input_lower):
            return 1.0
        
        # Sequence matching
        matcher = SequenceMatcher(None, input_lower, 
                                 sentence_lower[:len(input_lower)*2])
        return matcher.ratio()
    
    def get_suggestions(self, input_text, top_k=5, min_confidence=0.3):
        """
        Get autocomplete suggestions
        
        Returns:
            List of {'text': ..., 'confidence': ..., 'strategy': ...}
        """
        
        input_text = input_text.strip()
        
        if not input_text:
            # Empty input → popular sentences
            popular = self.sentence_freq.most_common(top_k)
            return [{
                'text': sent,
                'confidence': 0.3,
                'strategy': 'popular'
            } for sent, _ in popular]
        
        candidates = {}
        input_lower = input_text.lower()
        
        # ========== STRATEGY 1: EXACT PREFIX ==========
        if input_lower in self.prefix_index:
            matches = self.prefix_index[input_lower]
            
            for sentence in matches:
                input_words = len(input_lower.split())
                total_words = len(sentence.split())
                coverage = input_words / total_words if total_words > 0 else 0
                
                confidence = min(0.95, 0.70 + coverage * 0.25)
                
                candidates[sentence] = {
                    'confidence': confidence,
                    'strategy': 'exact_prefix',
                    'score': 100
                }
        
        # ========== STRATEGY 2: FUZZY PREFIX ==========
        if len(candidates) < top_k:
            for sentence in self.database:
                if sentence in candidates:
                    continue
                
                fuzzy_score = self.fuzzy_similarity(input_lower, sentence)
                
                if fuzzy_score >= 0.6:
                    confidence = min(0.85, fuzzy_score * 0.85)
                    
                    candidates[sentence] = {
                        'confidence': confidence,
                        'strategy': 'fuzzy_match',
                        'score': 80
                    }
        
        # ========== STRATEGY 3: WORD MATCHING ==========
        if len(candidates) < top_k * 2:
            input_words = set(input_lower.split())
            
            for sentence in self.database:
                if sentence in candidates:
                    continue
                
                sentence_words = set(sentence.lower().split())
                common_words = input_words & sentence_words
                
                if common_words:
                    match_ratio = len(common_words) / len(input_words)
                    
                    if match_ratio >= 0.5:
                        confidence = min(0.70, match_ratio * 0.8)
                        
                        candidates[sentence] = {
                            'confidence': confidence,
                            'strategy': 'word_match',
                            'score': 60
                        }
        
        # ========== STRATEGY 4: POPULAR FALLBACK ==========
        if len(candidates) < 3:
            for sentence, _ in self.sentence_freq.most_common(5):
                if sentence not in candidates:
                    candidates[sentence] = {
                        'confidence': 0.35,
                        'strategy': 'popular',
                        'score': 30
                    }
        
        # Sort and filter
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda x: (x[1]['score'], x[1]['confidence']),
            reverse=True
        )
        
        results = []
        for text, info in sorted_candidates[:top_k]:
            if info['confidence'] >= min_confidence:
                results.append({
                    'text': text,
                    'confidence': round(info['confidence'], 2),
                    'strategy': info['strategy']
                })
        
        return results
    
    def display_suggestions(self, suggestions):
        """Display suggestions with nice formatting"""
        
        if not suggestions:
            print("   ⚠️  Không tìm thấy câu phù hợp")
            return
        
        print(f"\n   📋 Suggestions:")
        
        for i, sugg in enumerate(suggestions, 1):
            conf = sugg['confidence']
            
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
            
            # Strategy icon
            strategy_icons = {
                'exact_prefix': '🎯',
                'fuzzy_match': '🔍',
                'word_match': '🔤',
                'popular': '⭐'
            }
            strategy_icon = strategy_icons.get(sugg['strategy'], '❓')
            
            print(f"   {i}. {icon} [{conf*100:.0f}% {conf_label}] {strategy_icon} {sugg['text']}")
            print(f"      └─ Strategy: {sugg['strategy']}")
    
    def run_interactive(self):
        """Run interactive terminal session"""
        
        print("\n" + "="*70)
        print("🎯 INTERACTIVE AUTOCOMPLETE - Ca Dao & Tục Ngữ")
        print("="*70)
        print("\n📖 Instructions:")
        print("   • Nhập một phần câu ca dao/tục ngữ")
        print("   • System sẽ gợi ý các câu hoàn chỉnh")
        print("   • Gõ 'q' hoặc 'quit' để thoát")
        print("   • Gõ 'stats' để xem thống kê")
        print("\n" + "-"*70)
        
        session_stats = {
            'queries': 0,
            'successful': 0,
            'strategies': Counter()
        }
        
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
                    self._show_stats(session_stats)
                    continue
                
                if user_input.lower() == 'help':
                    print("\n📖 Commands:")
                    print("   • Nhập text: Tìm gợi ý")
                    print("   • 'stats': Xem thống kê session")
                    print("   • 'q' / 'quit': Thoát")
                    continue
                
                if not user_input:
                    print("   ⚠️  Vui lòng nhập text")
                    continue
                
                # Get suggestions
                session_stats['queries'] += 1
                suggestions = self.get_suggestions(user_input, top_k=5)
                
                if suggestions:
                    session_stats['successful'] += 1
                    for sugg in suggestions:
                        session_stats['strategies'][sugg['strategy']] += 1
                
                # Display
                self.display_suggestions(suggestions)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
        
        # Final stats
        print("\n" + "="*70)
        self._show_stats(session_stats)
        print("="*70)
    
    def _show_stats(self, stats):
        """Show session statistics"""
        
        print("\n📊 Session Statistics:")
        print(f"   Total queries: {stats['queries']}")
        
        if stats['queries'] > 0:
            success_rate = stats['successful'] / stats['queries'] * 100
            print(f"   Successful: {stats['successful']} ({success_rate:.1f}%)")
        
        if stats['strategies']:
            print(f"\n   Strategies used:")
            for strategy, count in stats['strategies'].most_common():
                print(f"      • {strategy}: {count}")


# ========== MAIN ==========
def main():
    """Main entry point"""
    
    # Setup paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "processed"
    
    # Load data
    print("🔄 Loading data...")
    
    try:
        with open(DATA_DIR / "train.json", 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        # Extract unique sentences
        database = list(set(item['full'] for item in train_data))
        
        print(f"✓ Loaded {len(database)} unique sentences")
        
    except FileNotFoundError:
        print("❌ Data file not found!")
        print(f"   Looking for: {DATA_DIR / 'train.json'}")
        print("\n💡 Using demo data instead...")
        
        # Demo data
        database = [
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
    
    # Create autocomplete system
    autocomplete = InteractiveAutocomplete(database)
    
    # Run interactive mode
    autocomplete.run_interactive()


# ========== QUICK TEST MODE ==========
def quick_test():
    """Quick test without interactive mode"""
    
    print("\n" + "="*70)
    print("🧪 QUICK TEST MODE")
    print("="*70)
    
    # Demo data
    database = [
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
        "gieo nhân nào gặt quả nấy",
    ]
    
    autocomplete = InteractiveAutocomplete(database)
    
    test_cases = [
        "ăn",
        "ăn quả",
        "có công",
        "gần mực",
        "học",
        "xyz"
    ]
    
    for test_input in test_cases:
        print(f"\n{'─'*70}")
        print(f">>> Nhập: {test_input}")
        
        suggestions = autocomplete.get_suggestions(test_input, top_k=3)
        autocomplete.display_suggestions(suggestions)
    
    print("\n" + "="*70)
    print("✅ Test complete!")
    print("\nTo run interactive mode:")
    print("   python interactive_autocomplete.py")
    print("="*70)


if __name__ == "__main__":
    # Check if running in interactive mode or test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        main()