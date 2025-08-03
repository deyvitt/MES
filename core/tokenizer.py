# =============================================================================
# core/tokenizer.py
# =============================================================================
from transformers import AutoTokenizer
import torch
from config import MambaConfig
from typing import List, Dict, Union

class MambaTokenizer:
    def __init__(self, config: MambaConfig, tokenizer_name: str = "gpt2"):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.vocab_size = len(self.tokenizer)
        
    def encode(self, text: str, max_length: int = None) -> Dict[str, torch.Tensor]:
        """Encode text to token ids"""
        if max_length is None:
            max_length = self.config.max_seq_len
            
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }
    
    def encode_batch(self, texts: List[str], max_length: int = None) -> Dict[str, torch.Tensor]:
        """Encode batch of texts"""
        if max_length is None:
            max_length = self.config.max_seq_len
            
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token ids to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def decode_batch(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """Decode batch of token ids"""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
 