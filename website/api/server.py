from __future__ import annotations
import io
import os
import re
from typing import List, Dict, Any, Tuple

from flask_cors import CORS
from flask import Flask, request, jsonify, abort

import chess.pgn as pgn
import torch
import torch.nn as nn

SPECIAL = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
PAD_ID, BOS_ID, EOS_ID, UNK_ID = SPECIAL.values()

class Vocab:
    def __init__(self, stoi: Dict[str, int], itos: List[str]):
        self.stoi = dict(stoi)
        self.itos = list(itos)

    def encode(self, seq: List[str]) -> List[int]:
        return [self.stoi.get(t, UNK_ID) for t in seq]

    def __len__(self):
        return len(self.itos)

class HierBiLSTMCheater(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, lstm_dim: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            emb_dim,
            lstm_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.token_head = nn.Sequential(
            nn.Linear(2 * lstm_dim, lstm_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_dim, 1),
        )
        self.side_head = nn.Sequential(
            nn.Linear(4 * lstm_dim, lstm_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_dim, 1),
        )

    def forward(self, x_ids, mask):
        te = self.tok_emb(x_ids)
        lengths = mask.sum(1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(te, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x_ids.size(1))
        tok = self.token_head(out).squeeze(-1)

        B, T = x_ids.shape
        idxs = torch.arange(T, device=x_ids.device).unsqueeze(0).expand(B, -1)
        lengths_bt = mask.sum(1)
        valid = (idxs >= 1) & (idxs < (lengths_bt.unsqueeze(1) - 1))
        Wmask = valid & (((idxs - 1) % 2 == 0)) & mask
        Bmask = valid & (((idxs - 1) % 2 == 1)) & mask

        def pool(h, m):
            m_f = m.float().unsqueeze(-1)
            s = (h * m_f).sum(1)
            c = m_f.sum(1).clamp_min(1.0)
            mean = s / c
            minus_inf = (~m).unsqueeze(-1).float() * -1e9
            mx, _ = (h + minus_inf).max(1)
            return torch.cat([mean, mx], -1)

        Wr, Br = pool(out, Wmask), pool(out, Bmask)
        sw = self.side_head(Wr).squeeze(-1)
        sb = self.side_head(Br).squeeze(-1)
        return tok, sw, sb, (Wmask, Bmask)

def load_checkpoint(p: str):
    import numpy as np
    from torch.serialization import add_safe_globals
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        add_safe_globals({"dtype": np.dtype, "ndarray": np.ndarray, "scalar": np.generic})
        return torch.load(p, map_location="cpu", weights_only=False)

def san_list_from_game(game):
    moves = []
    node = game
    b = game.board()
    while node.variations:
        node = node.variations[0]
        moves.append(b.san(node.move))
        b.push(node.move)
    return moves

def build_ids_mask(vocab: Vocab, moves: List[str]):
    ids = [BOS_ID] + vocab.encode(moves) + [EOS_ID]
    mask = [1] * len(ids)
    X = torch.tensor(ids).unsqueeze(0)
    M = torch.tensor(mask).bool().unsqueeze(0)
    return X, M

def apply_beta(tok, sw, sb, Wmask, Bmask, beta: float):
    if beta == 0:
        return tok
    t = tok.clone()
    t[Wmask] += beta * sw.unsqueeze(1).expand_as(t)[Wmask]
    t[Bmask] += beta * sb.unsqueeze(1).expand_as(t)[Bmask]
    return t


app = Flask(__name__)

CORS(
    app,
    resources={r"/infer": {"origins": [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]}},
    supports_credentials=False,
    methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Length"],
    max_age=86400,
)

CKPT_PATH = os.environ.get("CKPT", "cheat_hier_best.pt")
BETA_SIDE = float(os.environ.get("BETA_SIDE", "0.7"))
THRESHOLDS_JSON = os.environ.get("THRESHOLDS", "")

_model = None
_vocab = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAN_CLEAN = re.compile(r"\s+")

def read_first_game(pgn_text: str) -> pgn.Game:
    """Parse and return the first game from PGN text, or raise 400."""
    fh = io.StringIO(pgn_text)
    game = pgn.read_game(fh)
    if game is None:
        abort(400, description="No valid game found in PGN.")
    return game

def _ensure_loaded():
    global _model, _vocab
    if _model is not None and _vocab is not None:
        return

    if not os.path.exists(CKPT_PATH):
        raise SystemExit(f"Checkpoint not found: {CKPT_PATH}")

    ckpt = load_checkpoint(CKPT_PATH)
    _vocab = Vocab(ckpt["vocab"]["stoi"], ckpt["vocab"]["itos"])
    a = ckpt.get("args", {})
    _model = HierBiLSTMCheater(
        len(_vocab),
        emb_dim=a.get("emb_dim", 128),
        lstm_dim=a.get("lstm_dim", 256),
        num_layers=a.get("layers", 2),
        dropout=a.get("dropout", 0.2),
    )
    _model.load_state_dict(ckpt["model_state"])
    _model.to(_device).eval()

def _infer_side_payloads(game: pgn.Game) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Runs the model on the full game's SAN sequence, then splits per-ply
    probabilities into White/Black tracks. Returns two payloads:
    (white_payload, black_payload).
    """
    _ensure_loaded()

    all_moves = san_list_from_game(game)
    if not all_moves:
        return {"score": 0.0, "moves": []}, {"score": 0.0, "moves": []}

    # Build tensors
    X, M = build_ids_mask(_vocab, all_moves)
    X, M = X.to(_device), M.to(_device)

    with torch.no_grad():
        tok, sw, sb, (Wmask, Bmask) = _model(X, M)
        tok_adj = apply_beta(tok, sw, sb, Wmask, Bmask, BETA_SIDE)

        per_ply_probs = torch.sigmoid(tok_adj)[0, 1 : 1 + len(all_moves)]
        pW = torch.sigmoid(sw).item()
        pB = torch.sigmoid(sb).item()

    white_probs = per_ply_probs[::2].cpu().tolist()
    black_probs = per_ply_probs[1::2].cpu().tolist()

    def to_pct_list(xs: List[float]) -> List[Dict[str, float]]:
        return [{"prob": round(100.0 * x, 1)} for x in xs]

    white_payload = {
        "score": round(100.0 * pW, 1),
        "moves": to_pct_list(white_probs),
    }
    black_payload = {
        "score": round(100.0 * pB, 1),
        "moves": to_pct_list(black_probs),
    }
    return white_payload, black_payload

@app.route("/infer", methods=["POST"])
def infer():
    """
    Accepts:
      - JSON body: {"pgn": "<PGN text>"}
      - OR multipart/form-data with a file named "pgn"

    Returns:
      [
        {"score": float, "moves": [{"prob": float}, ...]},  # White (index 0)
        {"score": float, "moves": [{"prob": float}, ...]},  # Black (index 1)
      ]
    """
    pgn_text: str | None = None

    if request.is_json:
        data = request.get_json(silent=True) or {}
        pgn_text = data.get("pgn")

    if pgn_text is None and "pgn" in request.files:
        file = request.files["pgn"]
        pgn_text = file.read().decode("utf-8", errors="ignore")

    if not pgn_text or not pgn_text.strip():
        abort(400, description="Provide PGN via JSON {'pgn': '...'} or file field 'pgn'.")

    game = read_first_game(pgn_text)
    white_payload, black_payload = _infer_side_payloads(game)

    return jsonify([white_payload, black_payload])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "6767"))
    app.run(host="0.0.0.0", port=port, debug=False)
