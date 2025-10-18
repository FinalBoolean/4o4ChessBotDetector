import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
import pickle
from tqdm import tqdm

# --- 1. Constants and Vocab ---
SPECIAL_TOKENS = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
PAD_ID = SPECIAL_TOKENS["<PAD>"]
CHUNK_SIZE = 10 # 10 moves per chunk

def build_vocab(all_moves_list):
    """Builds a vocabulary from a list of move strings."""
    all_uci_moves = " ".join(all_moves_list).split()
    vocab = {tok: i for i, (tok, count) in enumerate(Counter(all_uci_moves).items(), len(SPECIAL_TOKENS))}
    for tok, i in SPECIAL_TOKENS.items():
        vocab[tok] = i
    print(f"Built vocab with {len(vocab)} tokens.")
    return vocab

def encode_sequence(seq: list, vocab: dict):
    """Encodes a sequence of UCI moves."""
    encoded = [vocab.get(token, SPECIAL_TOKENS["<UNK>"]) for token in seq]
    # No BOS/EOS needed for a fixed-size chunk
    return encoded

# --- 2. PyTorch Dataset ---
class ChessChunkDataset(Dataset):
    def __init__(self, dataframe, vocab):
        self.df = dataframe
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        moves_list = row['uci_moves'].split()
        
        # Pad or truncate to CHUNK_SIZE
        moves_list = moves_list[:CHUNK_SIZE]
        if len(moves_list) < CHUNK_SIZE:
            moves_list.extend(["<PAD>"] * (CHUNK_SIZE - len(moves_list)))
            
        encoded_moves = encode_sequence(moves_list, self.vocab)
        
        return (
            torch.tensor(encoded_moves, dtype=torch.long),
            torch.tensor(row['is_cheat'], dtype=torch.float) # Label
        )

# --- 3. Model Architecture ---
class ChunkLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size=128, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            emb_size, hidden_size, num_layers=num_layers,
            dropout=dropout, bidirectional=True, batch_first=True
        )
        # We output 1 logit (P(Cheat))
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len=10)
        x = self.embedding(x)
        # We don't need pack_padded_sequence since all inputs are fixed to 10
        outputs, (h_n, c_n) = self.lstm(x)
        
        # Concat final forward and backward hidden states
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        logits = self.classifier(h_n)
        return logits.squeeze(1) # Squeeze to shape (batch_size)

# --- 4. Training ---
def train_model():
    # --- Setup ---
    BATCH_SIZE = 128
    EPOCHS = 5 # You have time for more!
    LEARNING_RATE = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- Load Data ---
    df = pd.read_csv("chunks_dataset.csv")
    vocab = build_vocab(df['uci_moves'])
    dataset = ChessChunkDataset(df, vocab)
    
    # Split data: 80% train, 20% validation
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # --- Model, Loss, Optimizer ---
    model = ChunkLSTMClassifier(vocab_size=len(vocab)).to(DEVICE)
    # Binary Cross-Entropy with Logits (handles sigmoid internally)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for moves, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            moves, labels = moves.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(moves)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validation Loop ---
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for moves, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                moves, labels = moves.to(DEVICE), labels.to(DEVICE)
                logits = model(moves)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.sigmoid(logits) > 0.5
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        print(f"Epoch {epoch}: "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val Acc: {100 * correct/total:.2f}%")

    # --- Save Artifacts ---
    print("Training complete. Saving model and vocab...")
    torch.save(model.state_dict(), "model.pth")
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("✅ Model and vocab saved!")

if __name__ == "__main__":
    train_model()