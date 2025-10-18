import io
import pickle
import torch
import chess.pgn
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import our new model class
from model_handler import ChunkLSTMClassifier, encode_sequence, CHUNK_SIZE

# --- 1. Server Setup ---
app = Flask(__name__)
CORS(app)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Load Model & Vocab ---
try:
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    
    model = ChunkLSTMClassifier(vocab_size=len(vocab))
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("✅ Model and vocab loaded successfully.")
except Exception as e:
    print(f"❌ Critical error: Could not load model. {e}")
    model, vocab = None, None

# --- 3. API Endpoint ---
@app.route("/analyze", methods=["POST"])
def analyze_game():
    if not model or not vocab:
        return jsonify({"error": "Model is not available."}), 503

    try:
        data = request.get_json()
        pgn_string = data["pgn"]
        game = chess.pgn.read_game(io.StringIO(pgn_string))
        if game is None:
            return jsonify({"error": "Invalid PGN."}), 400

        # Get all moves
        moves_uci = [move.uci() for move in game.mainline_moves()]
        white_moves = moves_uci[0::2]
        black_moves = moves_uci[1::2]

        white_probs, black_probs = [], []

        # --- Analyze White's Chunks ---
        for i in range(len(white_moves) - CHUNK_SIZE + 1):
            chunk = white_moves[i:i+CHUNK_SIZE]
            encoded = encode_sequence(chunk, vocab)
            tensor = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                logit = model(tensor)
                prob = torch.sigmoid(logit).item()
            white_probs.append(prob)

        # --- Analyze Black's Chunks ---
        for i in range(len(black_moves) - CHUNK_SIZE + 1):
            chunk = black_moves[i:i+CHUNK_SIZE]
            encoded = encode_sequence(chunk, vocab)
            tensor = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                logit = model(tensor)
                prob = torch.sigmoid(logit).item()
            black_probs.append(prob)

        # For the demo, we return the *highest* probability of cheating found.
        # This is more informative than an average.
        final_white_prob = max(white_probs) if white_probs else 0.0
        final_black_prob = max(black_probs) if black_probs else 0.0

        return jsonify({
            "white_prob": final_white_prob,
            "black_prob": final_black_prob
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

# --- 4. Run the Server ---
if __name__ == "__main__":
    app.run(debug=True, port=5000)