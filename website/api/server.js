// server.js
import express from "express";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();

// Middlewares
app.use(cors({ origin: true }));              // adjust origin in prod
app.use(express.json({ limit: "2mb" }));      // parse JSON bodies

// Health check
app.get("/health", (_req, res) => res.json({ ok: true }));

// POST /analyze — accepts { pgn: string } and returns analysis
app.post("/analyze", async (req, res) => {
  try {
    const { pgn } = req.body || {};
    if (typeof pgn !== "string" || !pgn.trim()) {
      return res.status(400).json({ error: "Missing or invalid 'pgn' string" });
    }

    // TODO: Replace with your real analysis (engine/ML/etc.)
    // For now, stub a deterministic response shape your UI can consume.
    const result = {
      meta: { receivedMoves: countMovesFromPGN(pgn) },
      summary: {
        suspicious: false,
        botProbability: 0.18
      },
      perMove: [] // e.g., [{ ply: 1, cpLoss: 23, best: "e4", played: "e4" }, ...]
    };

    return res.json(result);
  } catch (err) {
    console.error("Analyze error:", err);
    return res.status(500).json({ error: "Internal error" });
  }
});

// --- helpers ---
function countMovesFromPGN(pgn) {
  // very rough count of move numbers like "1.", "2.", etc.; replace with a proper PGN parser later
  const matches = pgn.match(/\d+\./g);
  return matches ? matches.length : 0;
}

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`API listening on http://localhost:${PORT}`));
