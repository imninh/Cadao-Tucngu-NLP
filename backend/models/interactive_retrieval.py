"""
INTERACTIVE RETRIEVAL TERMINAL
Test Improved Retrieval model với interactive interface

Usage:
    python interactive_retrieval.py
"""

import sys
from pathlib import Path
from collections import Counter
import pickle

import numpy as np


class InteractiveRetrievalTerminal:
    """
    Interactive terminal for Improved Retrieval model testing
    """
    
    def __init__(self, retrieval_model):
        self.model = retrieval_model
        
        self.session_stats = {
            'queries': 0,
            'high_confidence': 0,  # confidence > 0.8
            'medium_confidence': 0,  # 0.5 < confidence < 0.8
            'low_confidence': 0,  # confidence < 0.5
            'avg_confidence': [],
            'avg_score': []
        }
    
    def display_suggestions(self, suggestions):
        """Display suggestions with formatting"""
        
        if not suggestions or suggestions[0].get('text') == '':
            print("   ⚠️  Không tìm thấy câu phù hợp")
            return
        
        print(f"\n   📋 Gợi ý:")
        
        for i, sugg in enumerate(suggestions, 1):
            conf = sugg['confidence']
            score = sugg.get('score', 0)
            
            # Confidence level
            if conf >= 0.8:
                icon = "🟢"
                conf_label = "HIGH"
            elif conf >= 0.5:
                icon = "🟡"
                conf_label = "MED "
            else:
                icon = "🔴"
                conf_label = "LOW "
            
            print(f"   {i}. {icon} [{conf*100:.0f}% {conf_label}] {sugg['text']}")
            print(f"      └─ Score: {score:.3f}")
    
    def show_detailed_analysis(self, input_text):
        """Show detailed analysis"""
        
        print(f"\n{'─'*70}")
        print(f"🔍 DETAILED ANALYSIS")
        print(f"{'─'*70}")
        
        # Get top 10 for analysis
        candidates = self.model.predict_multiple(input_text, top_k=10)
        
        print(f"\n📝 Input: '{input_text}'")
        print(f"   Length: {len(input_text.split())} words")
        
        # Vectorize input to show matching features
        query_vec = self.model.vectorizer.transform([input_text.lower()])
        feature_names = self.model.vectorizer.get_feature_names_out()
        
        # Get non-zero features (ngrams in input)
        nonzero_indices = query_vec.nonzero()[1]
        
        print(f"\n🔑 Features extracted from input (character n-grams):")
        if len(nonzero_indices) > 0:
            for idx in nonzero_indices[:10]:  # Show top 10
                feature = feature_names[idx]
                tfidf_value = query_vec[0, idx]
                print(f"   • '{feature}' (TF-IDF: {tfidf_value:.3f})")
        else:
            print("   ⚠️  No features found (input too short or unknown n-grams)")
        
        # Show candidates
        print(f"\n📊 Top {len(candidates)} candidates:")
        for i, cand in enumerate(candidates, 1):
            conf = cand['confidence']
            score = cand['score']
            text = cand['text']
            
            # Confidence icon
            if conf >= 0.8:
                icon = "🟢"
            elif conf >= 0.5:
                icon = "🟡"
            else:
                icon = "🔴"
            
            print(f"\n   {i}. {icon} {text}")
            print(f"      Confidence: {conf:.1%} | BM25 Score: {score:.3f}")
            
            # Check if prefix match
            if text.lower().startswith(input_text.lower()):
                print(f"      Boost: Exact prefix match (+10)")
            
            # Show matching words
            matching_words = []
            input_words = set(input_text.lower().split())
            candidate_words = set(text.lower().split())
            matches = input_words & candidate_words
            
            if matches:
                print(f"      Matching words: {', '.join(matches)}")
        
        print(f"{'─'*70}")
    
    def show_database_info(self):
        """Show database statistics"""
        
        print(f"\n{'─'*70}")
        print(f"📚 DATABASE INFORMATION")
        print(f"{'─'*70}")
        
        print(f"\n📊 Statistics:")
        print(f"   Total sentences: {len(self.model.database):,}")
        print(f"   Vector dimension: {self.model.term_freqs.shape[1]:,}")
        print(f"   Vocabulary size: {len(self.model.vectorizer.vocabulary_):,}")
        print(f"   Average doc length: {self.model.avg_doc_len:.1f} words")
        
        # Show sample sentences
        print(f"\n📝 Sample sentences (random 10):")
        import random
        samples = random.sample(self.model.database, min(10, len(self.model.database)))
        for i, sent in enumerate(samples, 1):
            print(f"   {i}. {sent}")
        
        # Top features
        feature_names = self.model.vectorizer.get_feature_names_out()
        idf_scores = self.model.idf
        
        print(f"\n🔝 Most common features (low IDF = common):")
        top_common = np.argsort(idf_scores)[:15]
        for i, idx in enumerate(top_common, 1):
            print(f"   {i:2d}. '{feature_names[idx]}' (IDF: {idf_scores[idx]:.2f})")
        
        print(f"\n🔝 Most distinctive features (high IDF = rare):")
        top_rare = np.argsort(idf_scores)[-15:][::-1]
        for i, idx in enumerate(top_rare, 1):
            print(f"   {i:2d}. '{feature_names[idx]}' (IDF: {idf_scores[idx]:.2f})")
    
    def show_stats(self):
        """Show session statistics"""
        
        print(f"\n{'─'*70}")
        print(f"📊 SESSION STATISTICS")
        print(f"{'─'*70}")
        
        stats = self.session_stats
        
        print(f"\n📈 General:")
        print(f"   Total queries: {stats['queries']}")
        
        if stats['queries'] > 0:
            print(f"   High confidence (>80%): {stats['high_confidence']}")
            print(f"   Medium confidence (50-80%): {stats['medium_confidence']}")
            print(f"   Low confidence (<50%): {stats['low_confidence']}")
        
        if stats['avg_confidence']:
            avg_conf = sum(stats['avg_confidence']) / len(stats['avg_confidence'])
            print(f"   Average confidence: {avg_conf:.1%}")
        
        if stats['avg_score']:
            avg_score = sum(stats['avg_score']) / len(stats['avg_score'])
            print(f"   Average BM25 score: {avg_score:.3f}")
    
    def show_help(self):
        """Show help message"""
        
        print("\n📖 Available Commands:")
        print("   • <text>           : Get suggestions")
        print("   • detail:<text>    : Show detailed analysis")
        print("   • db               : Show database info")
        print("   • stats            : Show session statistics")
        print("   • help             : Show this help")
        print("   • q / quit / exit  : Exit program")
        
        print("\n💡 Examples:")
        print("   >>> ăn quả")
        print("   >>> detail:công cha")
        print("   >>> db")
        print("   >>> stats")
    
    def run_interactive(self):
        """Run interactive session"""
        
        print("\n" + "="*70)
        print("🔍 INTERACTIVE IMPROVED RETRIEVAL MODEL")
        print("   BM25 + Prefix Boost")
        print("="*70)
        print(f"\n📚 Database: {len(self.model.database):,} sentences")
        print(f"📊 Vocabulary: {len(self.model.vectorizer.vocabulary_):,} n-grams")
        print("\n📖 Instructions:")
        print("   • Nhập text để nhận suggestions")
        print("   • Gõ 'detail:<text>' để xem phân tích chi tiết")
        print("   • Gõ 'db' để xem thông tin database")
        print("   • Gõ 'stats' để xem thống kê")
        print("   • Gõ 'help' để xem hướng dẫn")
        print("   • Gõ 'q' để thoát")
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
                    self.show_stats()
                    continue
                
                if user_input.lower() == 'db':
                    self.show_database_info()
                    continue
                
                if user_input.lower().startswith('detail:'):
                    text = user_input[7:].strip()
                    if text:
                        self.show_detailed_analysis(text)
                    else:
                        print("   ⚠️  Usage: detail:<text>")
                    continue
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if not user_input:
                    print("   ⚠️  Vui lòng nhập text")
                    continue
                
                # Get suggestions
                self.session_stats['queries'] += 1
                
                suggestions = self.model.predict_multiple(user_input, top_k=5)
                
                # Update stats
                if suggestions:
                    top_conf = suggestions[0]['confidence']
                    top_score = suggestions[0].get('score', 0)
                    
                    self.session_stats['avg_confidence'].append(top_conf)
                    self.session_stats['avg_score'].append(top_score)
                    
                    if top_conf >= 0.8:
                        self.session_stats['high_confidence'] += 1
                    elif top_conf >= 0.5:
                        self.session_stats['medium_confidence'] += 1
                    else:
                        self.session_stats['low_confidence'] += 1
                
                # Display
                self.display_suggestions(suggestions)
                
                # Show tip after first query
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
        self.show_stats()
        print("="*70)


# ============================================================
# MAIN
# ============================================================

def main():
    """Main entry point"""
    
    print("\n" + "="*70)
    print("🚀 LOADING IMPROVED RETRIEVAL MODEL")
    print("="*70)
    
    # Setup paths
    BASE_DIR = Path(__file__).parent.parent
    MODEL_DIR = BASE_DIR / "trained_models"
    MODEL_PATH = MODEL_DIR / "improved_retrieval_model.pkl"
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"\n❌ Model not found at: {MODEL_PATH}")
        print(f"\n💡 Please train the model first:")
        print(f"   python models/retrieval.py")
        print(f"\n   Or run from backend/models/:")
        print(f"   python retrieval.py")
        return
    
    # Load model
    print(f"\n📦 Loading model from {MODEL_PATH}...")
    
    try:
        # Import numpy for database info
        import numpy as np
        
        # Import RetrievalModel
        import sys
        sys.path.append(str(BASE_DIR / "backend" / "models"))
        
        from retrieval import RetrievalModel
        
        model = RetrievalModel()
        model.load(MODEL_PATH)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Database: {len(model.database):,} sentences")
        print(f"   Vector dimension: {model.term_freqs.shape[1]:,}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run interactive terminal
    terminal = InteractiveRetrievalTerminal(model)
    terminal.run_interactive()


if __name__ == "__main__":
    main()