#!/usr/bin/env python3
"""
Training Files Checker & Emergency Vocab Generator
Run this to diagnose missing files and create vocab.pkl if needed
"""

import os
import sys
import pickle
import pandas as pd
from pathlib import Path

def check_file(filename, description):
    """Check if a file exists and print status"""
    exists = os.path.exists(filename)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filename}")
    if exists:
        size = os.path.getsize(filename)
        print(f"   Size: {size:,} bytes")
    return exists

def main():
    print("=" * 60)
    print("🔍 Chess Bot Detection - Training Files Check")
    print("=" * 60)
    print()
    
    # Check all required files
    files_status = {
        "Games.csv": check_file("Games.csv", "Training data"),
        "chunks_dataset.csv": check_file("chunks_dataset.csv", "Preprocessed chunks"),
        "vocab.pkl": check_file("vocab.pkl", "Vocabulary file"),
        "model.pth": check_file("model.pth", "Trained model"),
        "preprocess_chunks.py": check_file("preprocess_chunks.py", "Preprocessing script"),
        "train.py": check_file("train.py", "Training script"),
        "model_handler.py": check_file("model_handler.py", "Model handler"),
    }
    
    print()
    print("=" * 60)
    print("📋 DIAGNOSIS")
    print("=" * 60)
    
    # Determine what needs to be done
    if not files_status["Games.csv"]:
        print("❌ CRITICAL: Games.csv is missing!")
        print("   This is your source training data. You need to obtain it first.")
        return
    
    if not files_status["preprocess_chunks.py"]:
        print("⚠️  WARNING: preprocess_chunks.py not found")
        print("   You need this script to create chunks_dataset.csv")
    
    if not files_status["train.py"]:
        print("⚠️  WARNING: train.py not found")
        print("   You need this script to train the model and create vocab.pkl")
    
    if not files_status["chunks_dataset.csv"]:
        print()
        print("📝 STEP 1 NEEDED: Run preprocessing")
        print("   Command: python preprocess_chunks.py")
        print("   This will create chunks_dataset.csv from Games.csv")
        print()
    
    if not files_status["vocab.pkl"] or not files_status["model.pth"]:
        print()
        print("📝 STEP 2 NEEDED: Run training")
        print("   Command: python train.py")
        print("   This will create vocab.pkl and model.pth")
        print()
    
    # Emergency vocab creator
    if files_status["chunks_dataset.csv"] and not files_status["vocab.pkl"]:
        print()
        print("=" * 60)
        print("🚨 EMERGENCY OPTION")
        print("=" * 60)
        print()
        print("I can create vocab.pkl from your chunks_dataset.csv right now!")
        print("This won't train the model, but it will let the server start.")
        print()
        response = input("Create vocab.pkl now? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            create_vocab_from_chunks()
    
    # Check if we can at least create a minimal vocab
    elif not files_status["chunks_dataset.csv"] and files_status["Games.csv"]:
        print()
        print("=" * 60)
        print("🚨 EMERGENCY OPTION")
        print("=" * 60)
        print()
        print("I can create a minimal vocab.pkl directly from Games.csv!")
        print("This is a workaround to let the server start, but you should")
        print("still run the full training pipeline later.")
        print()
        response = input("Create minimal vocab.pkl now? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            create_vocab_from_games()

def create_vocab_from_chunks():
    """Create vocab.pkl from chunks_dataset.csv"""
    try:
        print("\n🔄 Reading chunks_dataset.csv...")
        df = pd.read_csv("chunks_dataset.csv")
        
        print(f"   Found {len(df)} chunks")
        
        # Extract all unique moves
        all_moves = set()
        for moves_str in df['uci_moves']:
            moves = moves_str.split()
            all_moves.update(moves)
        
        # Create vocab: move -> index
        vocab = {move: idx for idx, move in enumerate(sorted(all_moves), start=1)}
        vocab['<PAD>'] = 0  # Add padding token
        
        # Save vocab
        with open("vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
        
        print(f"✅ Created vocab.pkl with {len(vocab)} unique moves!")
        print(f"   Vocabulary size: {len(vocab)}")
        print(f"   Sample moves: {list(vocab.keys())[:10]}")
        
    except Exception as e:
        print(f"❌ Error creating vocab: {e}")
        import traceback
        traceback.print_exc()

def create_vocab_from_games():
    """Create minimal vocab from Games.csv directly (emergency fallback)"""
    try:
        print("\n🔄 Reading Games.csv...")
        print("   (This may take a moment for large files...)")
        
        # Read just the moves columns
        df = pd.read_csv("Games.csv", usecols=['AN'])
        
        print(f"   Found {len(df)} games")
        
        # Extract all unique moves from algebraic notation
        # Note: This is a simplified approach. Your actual preprocessing might differ.
        print("   Extracting unique moves...")
        all_moves = set()
        
        for i, moves_str in enumerate(df['AN']):
            if pd.isna(moves_str):
                continue
            # Split by spaces and clean up
            moves = str(moves_str).split()
            # Remove move numbers (e.g., "1.", "2.")
            moves = [m for m in moves if not m.endswith('.')]
            all_moves.update(moves)
            
            if (i + 1) % 10000 == 0:
                print(f"   Processed {i + 1} games...")
        
        # Create vocab: move -> index
        vocab = {move: idx for idx, move in enumerate(sorted(all_moves), start=1)}
        vocab['<PAD>'] = 0  # Add padding token
        
        # Save vocab
        with open("vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
        
        print(f"✅ Created minimal vocab.pkl with {len(vocab)} unique moves!")
        print(f"   Vocabulary size: {len(vocab)}")
        print()
        print("⚠️  NOTE: This is a simplified vocabulary.")
        print("   You should still run the full training pipeline:")
        print("   1. python preprocess_chunks.py")
        print("   2. python train.py")
        
    except Exception as e:
        print(f"❌ Error creating vocab: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()