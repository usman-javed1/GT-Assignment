import pandas as pd
from pathlib import Path
from src.document_loader.csv_loader import SubtitleCSVLoader
from src.model.sentiment_analyzer import SentimentAnalyzer

def main():
    # Initialize the loader
    loader = SubtitleCSVLoader("processed_subtitles")
    
    # Load and filter the data
    print("Loading and filtering subtitle data...")
    data = loader.load()
    print(f"Loaded {len(data)} valid subtitle entries")
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer(batch_size=5)
    
    # Process the data in batches
    print("Analyzing sentiments...")
    results = analyzer.analyze_batch(data)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Ensure output directory exists
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save results
    output_file = output_dir / "sentiment_analysis_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()