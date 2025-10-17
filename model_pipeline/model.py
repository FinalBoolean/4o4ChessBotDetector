
import os, io, math, random
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import chess
import chess.pgn as pgn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import roc_auc_score
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

# ---------- Repro ----------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ---------- Constants ----------
SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
}

PAD_ID = SPECIAL_TOKENS["<PAD>"]
BOS_ID = SPECIAL_TOKENS["<BOS>"]
EOS_ID = SPECIAL_TOKENS["<EOS>"]
UNK_ID = SPECIAL_TOKENS["<UNK>"]

# ---------- PGN parsing ----------
def iter_games_from_pgn(path_or_bytes):
    """Yield chess.pgn.Game objects from a file path or bytes."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        fh = io.StringIO(path_or_bytes.decode("utf-8", errors="ignore"))
    elif isinstance(path_or_bytes, str):
        fh = open(path_or_bytes, "r", encoding="utf-8", errors="ignore")
    else:
        raise TypeError("Expected path or bytes")
    try:
        while True:
            game = pgn.read_game(fh)
            if game is None:
                break
            yield game
    finally:
        fh.close()

def game_to_uci_sequence(game):
    """Return list of move.uci() strings from a PGN game."""
    seq = []
    board = game.board()
    for mv in game.mainline_moves():
        seq.append(mv.uci())
        board.push(mv)
    return seq

# ---------- Vocabulary ----------
def build_vocab(seqs, min_freq=1):
    cnt = Counter(m for s in seqs for m in s)
    vocab = dict(SPECIAL_TOKENS)
    for move, c in cnt.items():
        if c >= min_freq and move not in vocab:
            vocab[move] = len(vocab)
    return vocab

def encode_sequence(seq, vocab, add_bos_eos=True):
    tok = []
    if add_bos_eos:
        tok.append(vocab["<BOS>"])
    for m in seq:
        tok.append(vocab.get(m, vocab["<UNK>"]))
    if add_bos_eos:
        tok.append(vocab["<EOS>"])
    return tok

# ---------- Dataset ----------
class PgnMovesDataset(Dataset):
    """
    Each sample: (token_tensor, label_tensor)
    label_tensor = [y_white, y_black], each in {0,1}
    """
    def __init__(self, encoded_games: List[List[int]], labels: List[List[int]], max_len=None):
        assert len(encoded_games) == len(labels)
        self.data = encoded_games
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        if self.max_len is not None:
            if self.max_len >= 2:
                seq = seq[: self.max_len]
                if seq and seq[-1] != EOS_ID and self.data[idx][-1] == EOS_ID:
                    if len(seq) >= 2:
                        seq[-1] = EOS_ID
            else:
                seq = seq[: self.max_len]  # degenerate case, but safe
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

def collate_pad(batch):
    """Pad sequences for batching."""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in seqs])
    max_len = int(lengths.max().item())
    padded = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
    labels = torch.stack(labels)
    return padded, lengths, labels

# ---------- Model ----------
class DualSideDetector(nn.Module):
    """
    Two outputs:
        y_white = probability white cheats
        y_black = probability black cheats

    Upgrades:
      - Bidirectional LSTM
      - Side-aware attention pooling over valid move positions only
      - Learned "no-moves" sentinels per side
      - Separate heads (white/black)
    """
    def __init__(self, vocab_size, emb_dim=128, hidden=256, num_layers=1, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            emb_dim, hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.0 if num_layers == 1 else dropout
        )
        self.h_dim = hidden * 2  # bidirectional
        # Attention scorer (per side we reuse the same scorer; we just mask different steps)
        self.attn_w = nn.Linear(self.h_dim, 1, bias=False)
        # Learned replacements when a side has 0 valid moves:
        self.no_white = nn.Parameter(torch.zeros(self.h_dim))
        self.no_black = nn.Parameter(torch.zeros(self.h_dim))
        # Separate heads:
        self.white_head = nn.Linear(self.h_dim, 1)
        self.black_head = nn.Linear(self.h_dim, 1)

    @staticmethod
    def _build_side_masks(x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          valid_moves: (B, T)  boolean mask of positions strictly inside (BOS, EOS)
          white_mask:  (B, T)  whites at 1,3,5,... (within valid range)
          black_mask:  (B, T)  blacks at 2,4,6,... (within valid range)
        """
        B, T = x.shape
        device = x.device
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # 0..T-1
        lengths_exp = lengths.unsqueeze(1)
        valid_moves = (pos > 0) & (pos < (lengths_exp - 1))
        white_mask = valid_moves & (pos % 2 == 1)  # 1,3,5,...
        black_mask = valid_moves & (pos % 2 == 0)  # 2,4,6,...
        return valid_moves, white_mask, black_mask

    def _attn_pool(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        H: (B, T, Hd)
        mask: (B, T) boolean. Only positions with True contribute.
        Returns:
          pooled: (B, Hd)
        """
        # scores (B, T)
        scores = self.attn_w(H).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        # If a row is all -inf (no valid positions), softmax returns NaNs; handle separately
        # detect empties per batch item:
        empty = ~mask.any(dim=1)  # (B,)
        # softmax on non-empty rows
        weights = torch.zeros_like(scores)
        if (~empty).any():
            weights[~empty] = torch.softmax(scores[~empty], dim=1)
        # pooled (B, Hd)
        pooled = (H * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, empty

    def forward(self, x, lengths):
        """
        x: LongTensor (B, T)
        lengths: LongTensor (B,)
        """
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)  # (B, T, Hd)

        _, white_mask, black_mask = self._build_side_masks(x, lengths)

        h_white, white_empty = self._attn_pool(H, white_mask)
        h_black, black_empty = self._attn_pool(H, black_mask)

        if white_empty.any():
            h_white[white_empty] = self.no_white
        if black_empty.any():
            h_black[black_empty] = self.no_black

        logit_white = self.white_head(h_white)  # (B,1)
        logit_black = self.black_head(h_black)  # (B,1)
        logits = torch.cat([logit_white, logit_black], dim=-1)  # (B,2)
        return logits

# ---------- Loading data ----------
def load_pgn_folder_raw(folder) -> Tuple[List[List[str]], List[List[int]]]:
    """
    Returns raw sequences (list of UCI strings) and labels.
    Expects subfolders: 'white', 'black', 'both', 'neither' (case-insensitive startswith ok).
    """
    all_seqs = []
    labels = []
    for sub in os.listdir(folder):
        subdir = os.path.join(folder, sub)
        if not os.path.isdir(subdir):
            continue
        name = sub.lower()
        if name.startswith("white"):
            lbl = [1, 0]
        elif name.startswith("black"):
            lbl = [0, 1]
        elif name.startswith("both"):
            lbl = [1, 1]
        elif name.startswith("neither"):
            lbl = [0, 0]
        else:
            continue
        for fname in os.listdir(subdir):
            if not fname.lower().endswith(".pgn"):
                continue
            for g in iter_games_from_pgn(os.path.join(subdir, fname)):
                seq = game_to_uci_sequence(g)
                if seq:
                    all_seqs.append(seq)
                    labels.append(lbl)
    return all_seqs, labels

def encode_with_vocab(seqs: List[List[str]], vocab: Dict[str,int], add_bos_eos=True) -> List[List[int]]:
    return [encode_sequence(s, vocab, add_bos_eos=add_bos_eos) for s in seqs]

# ---------- Splits & weights ----------
def stratified_split_by_label_pairs(labels: List[List[int]], train_ratio=0.9, seed=42):
    """
    Stratify by the 4 label tuples: [1,0], [0,1], [1,1], [0,0]
    Returns list of train_indices, val_indices
    """
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for i, y in enumerate(labels):
        buckets[tuple(y)].append(i)
    train_idx, val_idx = [], []
    for key, idxs in buckets.items():
        rng.shuffle(idxs)
        k = int(len(idxs) * train_ratio)
        train_idx.extend(idxs[:k])
        val_idx.extend(idxs[k:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx

def compute_pos_weight(labels_tensor: torch.Tensor) -> torch.Tensor:
    """
    pos_weight for BCEWithLogits: weight for positive examples per class.
    pos_weight[c] = N_neg / N_pos  (so positives get higher weight when rare)
    labels_tensor: (N,2) float in {0,1}
    """
    y = labels_tensor
    N = y.shape[0]
    pos = y.sum(dim=0)                 # (2,)
    neg = N - pos
    # Avoid division by zero → if a class has no positives, set weight=1.0 (neutral)
    pw = torch.where(pos > 0, neg / pos, torch.ones_like(pos))
    return pw

# ---------- Training / Eval ----------
def eval_epoch(model, dl, device="cpu"):
    model.eval()
    total = 0.0
    n = 0
    all_probs = []
    all_trues = []
    with torch.no_grad():
        for batch, lengths, y in dl:
            batch = batch.to(device)
            lengths = lengths.to(device)
            y = y.to(device)
            logits = model(batch, lengths)
            loss = F.binary_cross_entropy_with_logits(logits, y, reduction="sum")
            total += loss.item()
            n += batch.size(0)
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_trues.append(y.cpu())
    avg_loss = total / max(n, 1)
    aurocs = {}
    if HAVE_SKLEARN and all_probs:
        probs = torch.cat(all_probs, dim=0).numpy()
        trues = torch.cat(all_trues, dim=0).numpy()
        for i, name in enumerate(["white", "black"]):
            # roc_auc_score requires both classes present
            if len(set(trues[:, i].tolist())) >= 2:
                aurocs[name] = roc_auc_score(trues[:, i], probs[:, i])
            else:
                aurocs[name] = None
    return avg_loss, aurocs

def train_example(
        data_folder="data/pgns",
        min_freq=1,
        max_len=None,
        emb_dim=128,
        hidden=256,
        num_layers=1,
        dropout=0.1,
        epochs=5,
        batch_size=32,
        lr=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
):
    raw_seqs, labels = load_pgn_folder_raw(data_folder)
    assert raw_seqs, "No PGNs found."

    train_idx, val_idx = stratified_split_by_label_pairs(labels, train_ratio=0.9, seed=42)
    train_seqs = [raw_seqs[i] for i in train_idx]
    val_seqs   = [raw_seqs[i] for i in val_idx]
    train_y    = [labels[i]   for i in train_idx]
    val_y      = [labels[i]   for i in val_idx]

    vocab = build_vocab(train_seqs, min_freq=min_freq)

    train_enc = encode_with_vocab(train_seqs, vocab, add_bos_eos=True)
    val_enc   = encode_with_vocab(val_seqs, vocab, add_bos_eos=True)

    train_ds = PgnMovesDataset(train_enc, train_y, max_len=max_len)
    val_ds   = PgnMovesDataset(val_enc,   val_y,   max_len=max_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_pad)

    model = DualSideDetector(
        vocab_size=len(vocab),
        emb_dim=emb_dim,
        hidden=hidden,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_labels_tensor = torch.tensor(train_y, dtype=torch.float)
    pos_weight = compute_pos_weight(train_labels_tensor).to(device)

    best_val = math.inf
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch, lengths, y in train_dl:
            batch = batch.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            logits = model(batch, lengths)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="mean")

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            running += loss.item() * batch.size(0)
            seen += batch.size(0)

        train_loss = running / max(seen, 1)
        val_loss, val_aucs = eval_epoch(model, val_dl, device=device)

        msg = f"Epoch {ep:02d} | train {train_loss:.4f} | val {val_loss:.4f}"
        if HAVE_SKLEARN:
            msg += " | AUROC " + ", ".join(
                f"{k}={('%.3f' % v) if v is not None else 'n/a'}" for k, v in val_aucs.items()
            )
        print(msg)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, vocab

# ---------- Inference ----------
def predict(model: DualSideDetector, vocab: Dict[str,int], pgn_path: str, threshold=0.5, device=None):
    """
    Predict which side(s) cheated for each PGN game in the file.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    for g in iter_games_from_pgn(pgn_path):
        seq = encode_sequence(game_to_uci_sequence(g), vocab, add_bos_eos=True)
        x = torch.tensor([seq], dtype=torch.long).to(device)
        lengths = torch.tensor([len(seq)], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(x, lengths)
            probs = torch.sigmoid(logits)[0].tolist()
        p_white, p_black = probs
        if p_white >= threshold and p_black >= threshold:
            verdict = "Both"
        elif p_white >= threshold:
            verdict = "White"
        elif p_black >= threshold:
            verdict = "Black"
        else:
            verdict = "Neither"
        print(f"White={p_white:.3f}  Black={p_black:.3f}  → {verdict}")

if __name__ == "__main__":
    # Nothing for now as this ia very basic prototype.
    pass

