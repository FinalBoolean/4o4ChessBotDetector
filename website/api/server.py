#!/usr/bin/env python3
"""
Flask server that accepts a PGN and returns:
[
  {
    "score": <float>,           # game-level score for White
    "moves": [{"prob": <float>}, ...]  # per-move probs for White, in move order
  },
  {
    "score": <float>,           # game-level score for Black
    "moves": [{"prob": <float>}, ...]  # per-move probs for Black, in move order
  }
]

White = 0, Black = 1  (by list index)

POST /infer
- Accepts:
  • JSON: {"pgn": "<PGN text>"}
  • or multipart/form-data with a file field named "pgn" (plain text .pgn)

Notes:
- The `compute_probs_for_side()` function is a placeholder. Replace it with your
  model inference (e.g., load ckpt and return per-ply probabilities 0..100).
- The placeholder is deterministic and based on simple hashing so the endpoint
  works out of the box without your model.
"""

from __future__ import annotations
import io
import os
import re
from typing import List, Dict, Any, Tuple
from flask_cors import CORS
from flask import Flask, request, jsonify, abort
import chess.pgn as pgn

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])
# ----------------------------
# PGN helpers
# ----------------------------
SAN_CLEAN = re.compile(r"\s+")

def read_first_game(pgn_text: str) -> pgn.Game:
    """Parse and return the first game from PGN text, or raise 400."""
    fh = io.StringIO(pgn_text)
    game = pgn.read_game(fh)
    if game is None:
        abort(400, description="No valid game found in PGN.")
    return game

def extract_san_moves_by_side(game: pgn.Game) -> Tuple[List[str], List[str]]:
    """
    Return (white_sans, black_sans) as lists of SAN strings in order.
    """
    white, black = [], []
    node = game
    ply_index = 0
    while node.variations:
        node = node.variation(0)
        san = SAN_CLEAN.sub(" ", node.san()).strip()
        if ply_index % 2 == 0:
            white.append(san)
        else:
            black.append(san)
        ply_index += 1
    return white, black

# ----------------------------
# Scoring / inference stubs
# ----------------------------
def stable_hash_f64(s: str) -> float:
    """Tiny deterministic hash → [0,1)."""
    h = 1469598103934665603  # FNV offset basis
    for c in s.encode("utf-8"):
        h ^= c
        h *= 1099511628211
        h &= (1 << 64) - 1
    return (h % 10_000_000) / 10_000_000.0

def compute_probs_for_side(sans: List[str]) -> List[float]:
    """
    Placeholder per-move probabilities in [0, 100].
    Replace this with your model’s per-ply outputs.
    """
    probs = []
    for i, san in enumerate(sans):
        # a benign, deterministic signal using SAN + index
        base = stable_hash_f64(f"{san}|{i}")  # [0,1)
        # shape it a little
        prob = 100.0 * (0.25 + 0.75 * base)  # [25,100)
        probs.append(round(prob, 1))
    return probs

def summarize_score(probs: List[float]) -> float:
    """
    Game-level score: simple average of per-move probs.
    Replace with your aggregation (e.g., calibrated logit mean, max, etc.).
    """
    if not probs:
        return 0.0
    return round(sum(probs) / len(probs), 1)

def build_side_payload(sans: List[str]) -> Dict[str, Any]:
    move_probs = compute_probs_for_side(sans)
    return {
        "score": summarize_score(move_probs),
        "moves": [{"prob": p} for p in move_probs],
    }

# ----------------------------
# Flask endpoint
# ----------------------------
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

    # JSON
    if request.is_json:
        data = request.get_json(silent=True) or {}
        pgn_text = data.get("pgn")

    # multipart (file upload)
    if pgn_text is None and "pgn" in request.files:
        file = request.files["pgn"]
        pgn_text = file.read().decode("utf-8", errors="ignore")

    if not pgn_text or not pgn_text.strip():
        abort(400, description="Provide PGN via JSON {'pgn': '...'} or file field 'pgn'.")

    game = read_first_game(pgn_text)
    white_sans, black_sans = extract_san_moves_by_side(game)

    white_payload = build_side_payload(white_sans)  # index 0
    black_payload = build_side_payload(black_sans)  # index 1

    return jsonify([white_payload, black_payload])

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    # Run with: python app.py
    # Or: gunicorn -w 2 -b 0.0.0.0:8000 app:app
    port = int(os.environ.get("PORT", "6767"))
    app.run(host="0.0.0.0", port=port, debug=False)
