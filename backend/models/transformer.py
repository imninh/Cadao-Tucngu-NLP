import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
import numpy as np
from collections import Counter
import math

class PositionalEncoding(nn.Module):
    """Build Positional Encoding từ đầu (sin/cos)."""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerDecoderLayer(nn.Module):
    """Một layer Decoder từ đầu (Attention + Feed Forward)."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_output))
        # Feed forward
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = self.norm2(tgt + self.dropout2(ff_output))
        return tgt

class TransformerModel(nn.Module):
    """Full Transformer Decoder từ đầu cho text generation."""
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1, max_seq_len=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.max_seq_len = max_seq_len
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        for layer in self.layers:
            src = layer(src, src_mask, src_key_padding_mask)
        output = self.fc_out(src)
        return output
    
    def generate_mask(self, size):
        """Mask cho future tokens (không nhìn trước)."""
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask

class TransformerTrainer:
    """Class để train và predict, tương tự NgramModel."""
    def __init__(self, n=4):  # n không dùng, chỉ để tương thích
        self.model = None
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_len = 50
    
    def build_vocab(self, sentences):
        """Build vocab từ đầu từ list sentences."""
        word_counts = Counter()
        for sentence in sentences:
            words = sentence.strip().split()
            word_counts.update(words)
        vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + list(word_counts.keys())
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(vocab)
        print(f"✓ Built vocab with {self.vocab_size} words")
    
    def prepare_data(self, data):
        """Preprocess data thành tensor."""
        inputs, targets = [], []
        for item in data:
            words = item['full'].strip().split()
            seq = [self.word_to_idx['<SOS>']] + [self.word_to_idx.get(w, self.word_to_idx['<UNK>']) for w in words] + [self.word_to_idx['<EOS>']]
            if len(seq) > self.max_seq_len:
                seq = seq[:self.max_seq_len]
            inputs.append(seq[:-1])  # Input: all but last
            targets.append(seq[1:])  # Target: all but first
        # Pad sequences
        max_len = max(len(s) for s in inputs)
        inputs = [s + [self.word_to_idx['<PAD>']] * (max_len - len(s)) for s in inputs]
        targets = [s + [self.word_to_idx['<PAD>']] * (max_len - len(s)) for s in targets]
        return torch.tensor(inputs, dtype=torch.long).to(self.device), torch.tensor(targets, dtype=torch.long).to(self.device)
    
    def train(self, train_data):
        sentences = [item['full'] for item in train_data]
        self.build_vocab(sentences)
        self.model = TransformerModel(self.vocab_size).to(self.device)
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.word_to_idx['<PAD>'])
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        inputs, targets = self.prepare_data(train_data)
        
        print(f"🔄 Training Transformer from scratch...")
        for epoch in range(10):  # 10 epochs cho dataset nhỏ
            self.model.train()
            optimizer.zero_grad()
            mask = self.model.generate_mask(inputs.size(1)).to(self.device)
            output = self.model(inputs, mask)
            loss = criterion(output.view(-1, self.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
        
        print("✓ Training complete!")
    
    def predict(self, partial_input, max_words=15):
        self.model.eval()
        words = partial_input.strip().split()
        input_seq = [self.word_to_idx['<SOS>']] + [self.word_to_idx.get(w, self.word_to_idx['<UNK>']) for w in words]
        input_tensor = torch.tensor([input_seq], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_words):
                mask = self.model.generate_mask(input_tensor.size(1)).to(self.device)
                output = self.model(input_tensor, mask)
                next_word_idx = output[0, -1].argmax().item()
                if next_word_idx == self.word_to_idx['<EOS>']:
                    break
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_word_idx]]).to(self.device)], dim=1)
        
        predicted_words = [self.idx_to_word[idx.item()] for idx in input_tensor[0][1:]]  # Skip <SOS>
        return ' '.join([w for w in predicted_words if w not in ['<PAD>', '<UNK>', '<EOS>']])
    
    def save(self, file_path):
        data = {
            'model_state': self.model.state_dict(),
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size
        }
        torch.save(data, file_path)
        print(f"✓ Model saved to {file_path}")
    
    def load(self, file_path):
        data = torch.load(file_path, map_location=self.device)
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = data['idx_to_word']
        self.vocab_size = data['vocab_size']
        self.model = TransformerModel(self.vocab_size).to(self.device)
        self.model.load_state_dict(data['model_state'])
        print(f"✓ Model loaded from {file_path}")

# Script train/test
def train_transformer_model():
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "processed"
    
    with open(DATA_DIR / "train.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    model = TransformerTrainer()
    model.train(train_data)
    
    model_path = BASE_DIR / "trained_models" / "transformer_model.pth"
    model.save(model_path)
    
    # Test
    print("\n🧪 Test predictions:")
    test_inputs = ["ăn quả", "có công mài sắt", "gần mực"]
    for inp in test_inputs:
        print(f"Input: '{inp}' → Predicted: '{model.predict(inp)}'")

if __name__ == "__main__":
    train_transformer_model()