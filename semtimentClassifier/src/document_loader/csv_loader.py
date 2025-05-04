from typing import List, Dict
from pathlib import Path
import pandas as pd
from langchain_community.document_loaders.base import BaseLoader
import re

def clean_subtitle_text(text):
    if not isinstance(text, str):
        return text
    cleaned = re.sub(r'<.*?>', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

class SubtitleCSVLoader(BaseLoader):
    def __init__(self, directory_path: str):
        self.directory_path = Path(directory_path)
        
    def load(self) -> List[Dict]:
        all_data = []
        
        # Get all CSV files in the directory
        csv_files = list(self.directory_path.glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Clean the English Subtitle column
                if 'English Subtitle' in df.columns:
                    df['English Subtitle'] = df['English Subtitle'].apply(clean_subtitle_text)
                
                # Filter rows where English Length is >= 5 words
                df['English_Word_Count'] = df['English Subtitle'].str.split().str.len()
                df = df[df['English_Word_Count'] >= 5]
                
                # Convert DataFrame rows to dictionaries
                records = df.to_dict('records')
                all_data.extend(records)
                
            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")
                
        return all_data 

class MasterSubtitleLoader:
    def __init__(self, file_path: str = "processed_subtitles/master_aligned_subtitles.csv"):
        self.file_path = Path(file_path)
        
    def load(self) -> List[Dict]:
        """Load and filter the master subtitles file."""
        try:
            # Read the CSV file
            df = pd.read_csv(self.file_path)
            
            print(df.head())
            # Clean the English Sentence column
            if 'English Sentence' in df.columns:
                df['English Sentence'] = df['English Sentence'].apply(clean_subtitle_text)
            # Filter rows where English Length is >= 5 words
            df['English_Word_Count'] = df['English Sentence'].str.split().str.len()
            df = df[df['English_Word_Count'] >= 3]
            df.reset_index(drop=True, inplace=True)
            df = df[df.index > 95860]
            # Drop the word count column as it's no longer needed
            df = df.drop('English_Word_Count', axis=1)
            
            # Convert DataFrame rows to dictionaries
            records = df.to_dict('records')
            
            print(f"Loaded {len(records)} valid subtitle entries")
            return records
            
        except Exception as e:
            print(f"Error loading file {self.file_path}: {str(e)}")
            return [] 