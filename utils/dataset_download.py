# Script to download the Chess Games dataset from Hugging Face

# Install the required library if not already installed
# You can run this in your environment: pip install datasets

from datasets import load_dataset

# Load the dataset (this will download it if not already cached)
dataset = load_dataset('angeluriot/chess_games')

# Optionally, save the dataset to disk for offline use
dataset.save_to_disk('chess_games_dataset')

# Print a sample to verify
print("Dataset downloaded successfully!")
print("Sample game:")
print(next(iter(dataset['train'])))