import pandas as pd
from pathlib import Path
from src.document_loader.csv_loader import MasterSubtitleLoader
from src.model.sentiment_analyzer import SentimentAnalyzer
import multiprocessing
import torch

def main():
    # Initialize the loader
    loader = MasterSubtitleLoader()
    
    # Load and filter the data
    print("Loading master subtitles file...")
    data = loader.load()
    # print("data", data)
    
    if not data:
        print("No data loaded. Please check if the New_Drama_Sentiment_Corpus.csv file exists in the processed_subtitles directory.")
        return
    
    print(f"Loaded {len(data)} valid subtitle entries")
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize sentiment analyzer with optimal number of threads and device
    num_threads = min(multiprocessing.cpu_count(), 8)  # Use up to 8 threads
    analyzer = SentimentAnalyzer(num_threads=num_threads, device=device)
    
    # Process the data using multiple threads
    print(f"Starting sentiment analysis using {num_threads} threads...")
    analyzer.analyze_batch(data)

if __name__ == "__main__":
    main()