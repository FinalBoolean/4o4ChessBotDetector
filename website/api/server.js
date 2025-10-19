import express from "express";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();

app.use(cors({ origin: true }));              
app.use(express.json({ limit: "2mb" }));      

app.get("/health", (_req, res) => res.json({ ok: true }));

app.post("/analyze", async (req, res) => {
  try {
    const { pgn } = req.body || {};
    if (typeof pgn !== "string" || !pgn.trim()) {
      return res.status(400).json({ error: "Missing or invalid 'pgn' string" });
    }

    const moveCount = countMovesFromPGN(pgn);
    const whiteMoveCount = Math.ceil(moveCount / 2);
    const blackMoveCount = Math.floor(moveCount / 2);

    const whiteMoves = Array.from({ length: whiteMoveCount }, () => ({
      prob: Math.random() * 100
    }));
    
    const blackMoves = Array.from({ length: blackMoveCount }, () => ({
      prob: Math.random() * 100
    }));

    const whiteScore = whiteMoves.reduce((sum, m) => sum + m.prob, 0) / whiteMoveCount;
    const blackScore = blackMoves.reduce((sum, m) => sum + m.prob, 0) / blackMoveCount;

    return res.json([
      { score: whiteScore, moves: whiteMoves },
      { score: blackScore, moves: blackMoves }
    ]);
  } catch (err) {
    console.error("Analyze error:", err);
    return res.status(500).json({ error: "Internal error" });
  }
});

function countMovesFromPGN(pgn) {
  const matches = pgn.match(/\d+\./g);
  return matches ? matches.length : 0;
}

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`API listening on http://localhost:${PORT}`));
