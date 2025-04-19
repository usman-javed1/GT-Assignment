import pandas as pd
from pathlib import Path
from src.document_loader.csv_loader import MasterSubtitleLoader
from src.model.sentiment_analyzer import SentimentAnalyzer

def main():
    # Initialize the loader
    loader = MasterSubtitleLoader()
    
    # Load and filter the data
    print("Loading master subtitles file...")
    data = loader.load()
    
    if not data:
        print("No data loaded. Please check if the master_aligned_subtitles.csv file exists in the processed_subtitles directory.")
        return
    
    print(f"Loaded {len(data)} valid subtitle entries")
    
    # Initialize sentiment analyzer with batch size 1 for one-by-one processing
    analyzer = SentimentAnalyzer(batch_size=1)
    
    # Process the data one sentence at a time
    print("Starting sentiment analysis...")
    analyzer.analyze_batch(data)

if __name__ == "__main__":
    main()