# =============================================================================
# core/preprocess.py
# =============================================================================
import re
import unicodedata
from config import MambaConfig
from typing import List, Dict, Any

class TextPreprocessor:
    def __init__(self, config: MambaConfig):
        self.config = config
        self.max_length = config.max_seq_len
        
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = None) -> List[str]:
        """Split text into chunks for distributed processing"""
        if chunk_size is None:
            chunk_size = self.max_length // 2
            
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts"""
        return [self.clean_text(text) for text in texts]
 