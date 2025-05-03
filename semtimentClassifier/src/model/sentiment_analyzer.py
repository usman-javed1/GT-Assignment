from typing import List, Dict
import pandas as pd
from pathlib import Path
from transformers import pipeline
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import sys

class SentimentAnalyzer:
    def __init__(self, num_threads: int = 4):
        self.model = pipeline("text-classification", model="finiteautomata/bertweet-base-sentiment-analysis")
        self.results_file = Path("output/sentiment_analysis_results.csv")
        self.processed_count = 0
        self.num_threads = num_threads
        self.print_lock = threading.Lock()
        self.save_lock = threading.Lock()
        self.print_queue = Queue()
        self.output_thread = None
        
    def _map_sentiment_label(self, label: str) -> str:
        """Map BERTweet labels to our format."""
        label_mapping = {
            'POS': 'positive',
            'NEG': 'negative',
            'NEU': 'neutral'
        }
        return label_mapping.get(label, 'neutral')

    def _analyze_text(self, text: str) -> Dict:
        """Analyze a single text using the model."""
        try:
            result = self.model(text)[0]
            return {
                "label": self._map_sentiment_label(result['label']),
                "score": result['score']
            }
        except Exception as e:
            print(f"Error in text analysis: {str(e)}")
            return {
                "label": "neutral",
                "score": 0.0
            }

    def _print_analysis_results(self, text: str, sentiment_data: Dict, item_number: int):
        """Queue the results for printing."""
        output = f"\n{'='*80}\n"
        output += f"Analysis Results for Item #{item_number}\n"
        output += f"{'='*80}\n"
        output += f"\nInput Text: {text[:200]}...\n"
        output += f"Label: {sentiment_data['label'].upper()}\n"
        output += f"Confidence Score: {sentiment_data['score']:.3f}\n"
        output += f"{'-'*80}\n"
        self.print_queue.put(output)
        
    def _output_printer(self):
        """Separate thread for handling output printing."""
        while True:
            output = self.print_queue.get()
            if output is None:  # Sentinel value to stop the thread
                break
            sys.stdout.write(output)
            sys.stdout.flush()
            self.print_queue.task_done()

    def _save_result(self, result: Dict):
        """Thread-safe saving of results."""
        with self.save_lock:
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

    def _process_item(self, item: Dict, item_number: int) -> Dict:
        """Process a single item with all necessary steps."""
        try:
            sentiment = self._analyze_text(item['English Sentence'])
            
            result = {
                **item,
                "sentiment_label": sentiment['label'],
                "sentiment_score": sentiment['score'],
                "processing_status": "success"
            }
            
            self._print_analysis_results(
                item['English Sentence'],
                {"label": sentiment['label'], "score": sentiment['score']},
                item_number
            )
            
            self._save_result(result)
            return result
            
        except Exception as e:
            with self.print_lock:
                print(f"\nError processing item {item_number}: {str(e)}")
            null_result = {
                **item,
                "sentiment_label": "neutral",
                "sentiment_score": 0.0,
                "processing_status": "error"
            }
            self._save_result(null_result)
            return null_result

    def analyze_batch(self, data: List[Dict]) -> None:
        """Process items using multiple threads."""
        total_items = len(data)
        print(f"\nStarting analysis of {total_items} items using {self.num_threads} threads...")
        
        # Clear or create results file
        self.results_file.parent.mkdir(exist_ok=True)
        if self.results_file.exists():
            self.results_file.unlink()
            
        # Start the output printer thread
        self.output_thread = threading.Thread(target=self._output_printer)
        self.output_thread.start()
        
        # Process items using thread pool
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self._process_item, item, i + 1): (item, i + 1)
                for i, item in enumerate(data)
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_item):
                completed += 1
                if completed % 10 == 0:  # Print progress every 10 items
                    with self.print_lock:
                        print(f"\nProgress: {completed}/{total_items} items processed "
                              f"({(completed/total_items)*100:.1f}%)")
        
        # Signal the output printer thread to stop
        self.print_queue.put(None)
        self.output_thread.join()
        
        print(f"\nAnalysis complete. Results saved to {self.results_file}")
        print(f"Successfully processed: {self.processed_count}/{total_items} items") 