# Converts the dataset to a list of PGNs
import pandas as pd
import chess.pgn
import io
from tqdm import tqdm

# Your heuristic: if '1' is in the label chunk, the whole chunk is a cheat.
def get_chunk_label(label_str_chunk):
    return 1 if '1' in label_str_chunk else 0

def create_chunk_dataset(csv_path, output_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # We will collect all our new data points here
    new_data = []
    
    print("Processing games into 10-move chunks...")
    # Use tqdm for a nice progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            pgn_string = row['Game']
            white_labels = str(row['Liste cheat white'])
            black_labels = str(row['Liste cheat black'])
            
            game = chess.pgn.read_game(io.StringIO(pgn_string))
            if game is None:
                continue

            # Get all half-moves (plies) in UCI format
            moves_uci = [move.uci() for move in game.mainline_moves()]
            
            # Combine all labels into one string, alternating white/black
            # This is tricky, let's just process them separately.
            
            # --- Process White's moves ---
            white_moves = moves_uci[0::2] # All even-indexed moves (0, 2, 4...)
            for i in range(0, len(white_moves) - 10):
                move_chunk = " ".join(white_moves[i:i+10])
                label_chunk = white_labels[i:i+10] # Aligns with the moves
                is_cheat = get_chunk_label(label_chunk)
                new_data.append([move_chunk, is_cheat])

            # --- Process Black's moves ---
            black_moves = moves_uci[1::2] # All odd-indexed moves (1, 3, 5...)
            for i in range(0, len(black_moves) - 10):
                move_chunk = " ".join(black_moves[i:i+10])
                label_chunk = black_labels[i:i+10]
                is_cheat = get_chunk_label(label_chunk)
                new_data.append([move_chunk, is_cheat])
                
        except Exception as e:
            # Skip any bad PGNs
            continue

    print(f"\nCreated {len(new_data)} total chunks.")
    
    # Save to a new, clean CSV
    chunk_df = pd.DataFrame(new_data, columns=['uci_moves', 'is_cheat'])
    # Shuffle the dataset
    chunk_df = chunk_df.sample(frac=1).reset_index(drop=True)
    
    chunk_df.to_csv(output_path, index=False)
    print(f"✅ Successfully saved new dataset to {output_path}")

# --- Run the script ---
if __name__ == "__main__":
    create_chunk_dataset(
        csv_path="labelled_data\Games.csv", 
        output_path="chunks_dataset.csv"
    )