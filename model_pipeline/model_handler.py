import torch
import torch.nn as nn

# --- Constants ---
SPECIAL_TOKENS = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
PAD_ID = SPECIAL_TOKENS["<PAD>"]
CHUNK_SIZE = 10

# --- Model Architecture ---
# (Copy the ChunkLSTMClassifier class from train.py and paste it here)
class ChunkLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size=128, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            emb_size, hidden_size, num_layers=num_layers,
            dropout=dropout, bidirectional=True, batch_first=True
        )
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = self.embedding(x)
        outputs, (h_n, c_n) = self.lstm(x)
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        logits = self.classifier(h_n)
        return logits.squeeze(1)

# --- Helper Functions ---
def encode_sequence(seq: list, vocab: dict):
    encoded = [vocab.get(token, SPECIAL_TOKENS["<UNK>"]) for token in seq]
    # Pad or truncate
    encoded = encoded[:CHUNK_SIZE]
    if len(encoded) < CHUNK_SIZE:
        encoded.extend([PAD_ID] * (CHUNK_SIZE - len(encoded)))
    return encoded