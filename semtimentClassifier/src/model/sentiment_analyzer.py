from typing import List, Dict
import pandas as pd
from pathlib import Path
from transformers import pipeline
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import sys
from datasets import Dataset

class SentimentAnalyzer:
    def __init__(self, num_threads: int = 4):
        self.model = pipeline(
            "text-classification",
            model="finiteautomata/bertweet-base-sentiment-analysis",
            device=0  # Switch to GPU
        )
        if self.model.device.type == 'cuda':
            print("[WARNING] For best GPU performance, use a HuggingFace Dataset and batch processing.")
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
        """Analyze a single text using the model, with input validation."""
        try:
            # Input validation: skip empty or excessively long strings
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Input text is empty or not a string.")
            if len(text) > 512:
                raise ValueError("Input text is too long for the model (>{} chars).".format(len(text)))
            result = self.model(text)[0]
            return {
                "label": self._map_sentiment_label(result['label']),
                "score": result['score']
            }
        except Exception as e:
            print(f"Error in text analysis for input: {repr(text)[:200]}...\n{str(e)}")
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
            # Only print analysis results if sentiment score is zero
            if sentiment['score'] == 0.0:
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

    def _analyze_batch_texts(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """Analyze a batch of texts using the pipeline's batching capabilities."""
        results = self.model(texts, batch_size=batch_size, truncation=True)
        # Map BERTweet labels to our format
        return [
            {
                "label": self._map_sentiment_label(res['label']),
                "score": res['score']
            }
            for res in results
        ]

    def analyze_batch(self, data: List[Dict], batch_size: int = 32) -> None:
        """Process items in batches using HuggingFace Dataset and pipeline batching for GPU efficiency."""
        total_items = len(data)
        print(f"\nStarting analysis of {total_items} items using GPU batch processing...")
        # Clear or create results file
        self.results_file.parent.mkdir(exist_ok=True)
        if self.results_file.exists():
            self.results_file.unlink()
        # Prepare texts and Dataset
        texts = [item['English Sentence'] for item in data]
        dataset = Dataset.from_dict({"text": texts})
        completed = 0
        def print_progress_bar(completed, total, bar_length=40):
            percent = completed / total
            bar = '=' * int(bar_length * percent) + '-' * (bar_length - int(bar_length * percent))
            print(f'\rProgress: |{bar}| {completed}/{total} ({percent*100:.1f}%)', end='', flush=True)
        # Process in batches
        for start in range(0, total_items, batch_size):
            end = min(start + batch_size, total_items)
            batch_texts = dataset['text'][start:end]
            batch_items = data[start:end]
            batch_results = self._analyze_batch_texts(batch_texts, batch_size=batch_size)
            for i, (item, sentiment) in enumerate(zip(batch_items, batch_results)):
                result = {
                    **item,
                    "sentiment_label": sentiment['label'],
                    "sentiment_score": sentiment['score'],
                    "processing_status": "success"
                }
                # Only print analysis results if sentiment score is zero
                if sentiment['score'] == 0.0:
                    self._print_analysis_results(
                        item['English Sentence'],
                        {"label": sentiment['label'], "score": sentiment['score']},
                        start + i + 1
                    )
                self._save_result(result)
                completed += 1
                print_progress_bar(completed, total_items)
        print()  # Newline after progress bar
        # Signal the output printer thread to stop
        self.print_queue.put(None)
        if self.output_thread:
            self.output_thread.join()
        print(f"\nAnalysis complete. Results saved to {self.results_file}")
        print(f"Successfully processed: {self.processed_count}/{total_items} items") 