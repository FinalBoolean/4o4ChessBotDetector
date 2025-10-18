import io
import pickle
import torch
import chess.pgn
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Import our new model class
from model_handler import ChunkLSTMClassifier, encode_sequence, CHUNK_SIZE

# --- 1. Server Setup ---
app = Flask(__name__)
CORS(app)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. Load Model & Vocab ---
try:
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    
    model = ChunkLSTMClassifier(vocab_size=len(vocab))
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    logger.info(f"✅ Model loaded on {DEVICE}. Vocab size: {len(vocab)}")
except Exception as e:
    logger.error(f"❌ Critical error: Could not load model. {e}")
    model, vocab = None, None

# --- 3. Health Check Endpoint ---
@app.route("/health", methods=["GET"])
def health_check():
    """Check if the server and model are ready."""
    if model and vocab:
        return jsonify({
            "status": "healthy",
            "device": DEVICE,
            "vocab_size": len(vocab),
            "chunk_size": CHUNK_SIZE
        }), 200
    return jsonify({"status": "unhealthy", "error": "Model not loaded"}), 503

# --- 4. API Endpoint ---
@app.route("/analyze", methods=["POST"])
def analyze_game():
    if not model or not vocab:
        return jsonify({"error": "Model is not available."}), 503

    try:
        data = request.get_json()
        
        # Validate input
        if not data or "pgn" not in data:
            return jsonify({"error": "Missing 'pgn' field in request."}), 400
        
        pgn_string = data["pgn"]
        
        # Parse PGN
        game = chess.pgn.read_game(io.StringIO(pgn_string))
        if game is None:
            return jsonify({"error": "Invalid PGN format."}), 400

        # Get all moves
        moves_uci = [move.uci() for move in game.mainline_moves()]
        
        # Validate we have enough moves
        if len(moves_uci) < CHUNK_SIZE:
            return jsonify({
                "error": f"Game too short. Need at least {CHUNK_SIZE} moves for analysis.",
                "moves_count": len(moves_uci)
            }), 400
        
        white_moves = moves_uci[0::2]
        black_moves = moves_uci[1::2]

        logger.info(f"Analyzing game: {len(white_moves)} White moves, {len(black_moves)} Black moves")

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

        # Return the *highest* probability (most suspicious chunk)
        final_white_prob = max(white_probs) if white_probs else 0.0
        final_black_prob = max(black_probs) if black_probs else 0.0

        logger.info(f"Results: White={final_white_prob:.3f}, Black={final_black_prob:.3f}")

        return jsonify({
            "white_prob": final_white_prob,
            "black_prob": final_black_prob,
            "white_chunks_analyzed": len(white_probs),
            "black_chunks_analyzed": len(black_probs),
            "total_moves": len(moves_uci)
        })

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --- 5. Run the Server ---
if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')