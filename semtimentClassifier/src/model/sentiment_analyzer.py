from typing import List, Dict
import pandas as pd
from pathlib import Path
from transformers import pipeline
import time

class SentimentAnalyzer:
    def __init__(self, batch_size: int = 16):  # Increased batch size as local model is faster
        self.model = pipeline("text-classification", model="finiteautomata/bertweet-base-sentiment-analysis")
        self.batch_size = batch_size
        self.results_file = Path("output/sentiment_analysis_results.csv")
        self.processed_count = 0
        
    def _map_sentiment_label(self, label: str) -> str:
        """Map BERTweet labels to our format."""
        label_mapping = {
            'POS': 'positive',
            'NEG': 'negative',
            'NEU': 'neutral'
        }
        return label_mapping.get(label, 'neutral')

    def _analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze a batch of texts using the model."""
        try:
            results = self.model(texts)
            return [
                {
                    "label": self._map_sentiment_label(result['label']),
                    "score": result['score']
                }
                for result in results
            ]
        except Exception as e:
            print(f"Error in batch analysis: {str(e)}")
            return [
                {
                    "label": "neutral",
                    "score": 0.0
                }
                for _ in texts
            ]

    def _print_analysis_results(self, text: str, sentiment_data: Dict, item_number: int):
        print("\n" + "="*80)
        print(f"Analysis Results for Item #{item_number}")
        print("="*80)
        print(f"\nInput Text: {text[:200]}...")
        print(f"Label: {sentiment_data['label'].upper()}")
        print(f"Confidence Score: {sentiment_data['score']:.3f}")
        print("-"*80)
        
    def _save_result(self, result: Dict):
        """Save a single result to CSV file."""
        df = pd.DataFrame([result])
        
        # Create directory if it doesn't exist
        self.results_file.parent.mkdir(exist_ok=True)
        
        # If file doesn't exist, create it with headers
        if not self.results_file.exists():
            df.to_csv(self.results_file, index=False)
        else:
            # Append without headers
            df.to_csv(self.results_file, mode='a', header=False, index=False)
        
        self.processed_count += 1

    def analyze_batch(self, data: List[Dict]) -> None:
        """Process items in batches and save results progressively."""
        total_items = len(data)
        print(f"\nStarting analysis of {total_items} items...")
        
        # Clear or create results file
        self.results_file.parent.mkdir(exist_ok=True)
        if self.results_file.exists():
            self.results_file.unlink()
        
        # Process in batches
        for i in range(0, total_items, self.batch_size):
            batch_data = data[i:i + self.batch_size]
            batch_texts = [item['English Subtitle'] for item in batch_data]
            
            print(f"\nProcessing batch {i//self.batch_size + 1}/{(total_items + self.batch_size - 1)//self.batch_size}")
            
            # Get sentiment analysis for the batch
            sentiment_results = self._analyze_batch(batch_texts)
            
            # Process and save each item in the batch
            for item, sentiment in zip(batch_data, sentiment_results):
                try:
                    # Create the result dictionary
                    result = {
                        **item,
                        "sentiment_label": sentiment['label'],
                        "sentiment_score": sentiment['score'],
                        "processing_status": "success"
                    }
                    
                    # Print analysis results
                    self._print_analysis_results(
                        item['English Subtitle'],
                        {"label": sentiment['label'], "score": sentiment['score']},
                        self.processed_count + 1
                    )
                    
                    # Save the result
                    self._save_result(result)
                    
                except Exception as e:
                    print(f"\nError processing item: {str(e)}")
                    null_result = {
                        **item,
                        "sentiment_label": "neutral",
                        "sentiment_score": 0.0,
                        "processing_status": "error"
                    }
                    self._save_result(null_result)
            
            # Print progress
            print(f"\nProgress: {min(i + self.batch_size, total_items)}/{total_items} items processed "
                  f"({(min(i + self.batch_size, total_items)/total_items)*100:.1f}%)")
        
        print(f"\nAnalysis complete. Results saved to {self.results_file}")
        print(f"Successfully processed: {self.processed_count}/{total_items} items") 