from typing import List, Dict
from langchain_core.messages import HumanMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from src.model.openrouter import OpenRouter
import json
import re
import time
import pandas as pd
from pathlib import Path

class SentimentAnalyzer:
    def __init__(self, batch_size: int = 5, max_retries: int = 5, rate_limit_delay: int = 60):
        self.model = OpenRouter(model_name="deepseek/deepseek-r1-distill-llama-70b:free")
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.rate_limit_delay = 60
        self.output_parser = self._create_output_parser()
        self.results_file = Path("output/sentiment_analysis_results.csv")
        self.processed_count = 0
        
    def _create_output_parser(self):
        response_schemas = [
            ResponseSchema(name="positive_score", description="Float score between 0 and 1 indicating positive sentiment"),
            ResponseSchema(name="negative_score", description="Float score between 0 and 1 indicating negative sentiment"),
            ResponseSchema(name="neutral_score", description="Float score between 0 and 1 indicating neutral sentiment"),
            ResponseSchema(name="label", description='One of ["positive", "negative", "neutral"]'),
            ResponseSchema(name="positive_words", description="List of positive words found in the text"),
            ResponseSchema(name="negative_words", description="List of negative words found in the text")
        ]
        return StructuredOutputParser.from_response_schemas(response_schemas)
        
    def _create_prompt(self, text: str) -> str:
        template = """Analyze the sentiment of the following text and provide scores and word lists.
        Text: {text}
        
        {format_instructions}
        
        Your response MUST be in valid JSON format with the following structure:
        {{
            "positive_score": <float between 0 and 1>,
            "negative_score": <float between 0 and 1>,
            "neutral_score": <float between 0 and 1>,
            "label": <"positive" or "negative" or "neutral">,
            "positive_words": <list of positive words>,
            "negative_words": <list of negative words>
        }}
        
        Requirements:
        1. All scores must be between 0 and 1
        2. Scores must sum to 1.0
        3. Extract actual words from the text for positive and negative lists
        4. Label should be the sentiment with highest score
        5. Response must be valid JSON
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["text"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        return prompt.format(text=text)

    def _clean_model_response(self, response: str) -> str:
        """Clean the model's response to ensure it's valid JSON."""
        # Try to extract JSON from the response using regex
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            response = json_match.group(0)
        
        # Remove any markdown code block markers
        response = re.sub(r'```json\s*|\s*```', '', response)
        
        # Ensure lists are properly formatted
        response = re.sub(r"'", '"', response)
        
        return response

    def _parse_sentiment(self, response_text: str) -> Dict:
        """Parse the sentiment response with error handling."""
        try:
            # Clean the response first
            cleaned_response = self._clean_model_response(response_text)
            
            # Try to parse as JSON first
            try:
                data = json.loads(cleaned_response)
                # Validate the required fields
                required_fields = ["positive_score", "negative_score", "neutral_score", 
                                 "label", "positive_words", "negative_words"]
                if all(field in data for field in required_fields):
                    return data
            except json.JSONDecodeError:
                pass
            
            # If JSON parsing fails, use the output parser
            return self.output_parser.parse(cleaned_response)
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print("Using default neutral values")
            return {
                "positive_score": 0.0,
                "negative_score": 0.0,
                "neutral_score": 1.0,
                "label": "neutral",
                "positive_words": [],
                "negative_words": []
            }

    def _print_analysis_results(self, text: str, sentiment_data: Dict, item_number: int):
        print("\n" + "="*80)
        print(f"Analysis Results for Item #{item_number}")
        print("="*80)
        print(f"\nInput Text: {text[:200]}...")
        print("\nSentiment Scores:")
        print(f"  Positive: {sentiment_data['positive_score']:.3f}")
        print(f"  Negative: {sentiment_data['negative_score']:.3f}")
        print(f"  Neutral:  {sentiment_data['neutral_score']:.3f}")
        print(f"\nOverall Label: {sentiment_data['label'].upper()}")
        print("\nPositive Words Found:", ', '.join(sentiment_data['positive_words']) if sentiment_data['positive_words'] else "None")
        print("Negative Words Found:", ', '.join(sentiment_data['negative_words']) if sentiment_data['negative_words'] else "None")
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
        
    def _handle_rate_limit(self, retry_count: int) -> bool:
        """Handle rate limit with exponential backoff."""
        if retry_count >= self.max_retries:
            print(f"Max retries ({self.max_retries}) exceeded. Skipping this item.")
            return False
            
        wait_time = self.rate_limit_delay
        print(f"Rate limit hit. Waiting {wait_time} seconds before retry {retry_count + 1}/{self.max_retries}")
        time.sleep(wait_time)
        return True

    def _analyze_single_item(self, item: Dict, item_number: int) -> Dict:
        """Analyze a single item with retry logic."""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                prompt = self._create_prompt(item['English Subtitle'])
                print(f"\nProcessing item {item_number}")
                print(f"Sending prompt to model...")
                
                response = self.model.invoke(
                    input=[HumanMessage(content=prompt)]
                )
                
                print("\nModel Response:")
                print("-"*30)
                print(response.content)
                print("-"*30)
                
                sentiment_data = self._parse_sentiment(response.content)
                self._print_analysis_results(item['English Subtitle'], sentiment_data, item_number)
                
                # Combine original data with sentiment analysis
                result = {**item, **sentiment_data}
                return result
                
            except Exception as e:
                error_str = str(e)
                print(f"\nError: {error_str}")
                
                if "Rate limit exceeded" in error_str:
                    if not self._handle_rate_limit(retry_count):
                        break
                    retry_count += 1
                else:
                    print(f"Non-rate-limit error: {error_str}")
                    break
        
        # If we get here, either max retries exceeded or non-rate-limit error
        print(f"Unable to process item {item_number}. Using neutral values.")
        null_sentiment = {
            "positive_score": 0.0,
            "negative_score": 0.0,
            "neutral_score": 1.0,
            "label": "neutral",
            "positive_words": [],
            "negative_words": [],
            "processing_status": "failed"
        }
        return {**item, **null_sentiment}

    def analyze_batch(self, data: List[Dict]) -> None:
        """Process items and save results progressively."""
        total_items = len(data)
        print(f"\nStarting analysis of {total_items} items...")
        
        # Clear or create results file
        self.results_file.parent.mkdir(exist_ok=True)
        if self.results_file.exists():
            self.results_file.unlink()
        
        for i, item in enumerate(data, 1):
            try:
                result = self._analyze_single_item(item, i)
                result['processing_status'] = 'success' if result['label'] != 'neutral' else 'failed'
                self._save_result(result)
                
                # Print progress
                print(f"\nProgress: {i}/{total_items} items processed ({(i/total_items)*100:.1f}%)")
                
                # Add a small delay between items to avoid hitting rate limits
                if i < total_items:
                    time.sleep(3)  # 3-second delay between items
                    
            except Exception as e:
                print(f"\nError processing item {i}: {str(e)}")
                null_result = {
                    **item,
                    "positive_score": 0.0,
                    "negative_score": 0.0,
                    "neutral_score": 1.0,
                    "label": "neutral",
                    "positive_words": [],
                    "negative_words": [],
                    "processing_status": "error"
                }
                self._save_result(null_result)
        
        print(f"\nAnalysis complete. Results saved to {self.results_file}")
        print(f"Successfully processed: {self.processed_count}/{total_items} items") 