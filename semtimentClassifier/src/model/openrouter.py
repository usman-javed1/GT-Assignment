from langchain_openai import ChatOpenAI
from typing import Optional, Dict, Any, List
import os
from pydantic import Field, BaseModel


class KeyRotator:
    def __init__(self, keys_file: str):
        with open(keys_file, 'r') as f:
            self.keys = [key.strip() for key in f.readlines() if key.strip()]
        self.current_index = 0
        
    def get_next_key(self) -> str:
        key = self.keys[self.current_index]
        print(f"Using API Key {self.current_index + 1}/{len(self.keys)}: {key}")
        self.current_index = (self.current_index + 1) % len(self.keys)
        return key


class OpenRouter(ChatOpenAI):
    def __init__(
        self,
        keys_file: str = "src/model/keys.txt",
        **kwargs
    ):
        # Create key rotator first
        key_rotator = KeyRotator(keys_file)
        initial_key = key_rotator.get_next_key()
        
        # Initialize with default settings
        super().__init__(
            model_name=kwargs.get("model_name", "google/gemini-2.5-pro-exp-03-25:free"),
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=initial_key,
            max_tokens=400,  # Limit tokens to stay within free tier
            default_headers={
                "HTTP-Referer": "https://github.com/",  # Required for OpenRouter
                "X-Title": "Python Script"  # Required for OpenRouter
            }
        )
        
        # Store key rotator after parent initialization
        self._key_rotator = key_rotator
    
    def _rotate_key(self) -> None:
        """Rotate to the next API key"""
        if self._key_rotator:
            new_key = self._key_rotator.get_next_key()
            self.openai_api_key = new_key
    
    def invoke(self, input: List, **kwargs) -> Any:
        # Always rotate to next key before making a request
        self._rotate_key()
        try:
            return super().invoke(input=input, **kwargs)
        except Exception as e:
            print(f"Error with current key, trying next key... Error: {str(e)}")
            # If there's an error, try with the next key
            self._rotate_key()
            return super().invoke(input=input, **kwargs)


    
