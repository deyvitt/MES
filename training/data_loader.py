# =============================================================================
# training/data_loader.py
# =============================================================================
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Iterator
import json
import random
from core.tokenizer import MambaTokenizer
from core.preprocess import TextPreprocessor

class MambaDataset(Dataset):
    """Dataset for Mamba training"""
    
    def __init__(self, data_path: str, tokenizer: MambaTokenizer, 
                 preprocessor: TextPreprocessor, config):
        self.config = config
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_length = config.max_seq_len
        
        # Load data
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[str]:
        """Load training data from file"""
        data = []
        
        try:
            if data_path.endswith('.json'):
                with open(data_path, 'r') as f:
                    raw_data = json.load(f)
                    if isinstance(raw_data, list):
                        data = [item['text'] if isinstance(item, dict) else str(item) 
                               for item in raw_data]
                    else:
                        data = [raw_data['text']]
                        
            elif data_path.endswith('.txt'):
                with open(data_path, 'r') as f:
                    content = f.read()
                    # Split into chunks
                    data = self.preprocessor.chunk_text(content, self.max_length)
            
            print(f"Loaded {len(data)} training examples")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create dummy data for testing
            data = [f"This is example text number {i}." for i in range(1000)]
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training example"""
        text = self.data[idx]
        
        # Preprocess text
        clean_text = self.preprocessor.clean_text(text)
        
        # Tokenize
        encoded = self.tokenizer.encode(clean_text, max_length=self.max_length)
        
        # Create input and target sequences
        input_ids = encoded['input_ids'].squeeze(0)  # [seq_len]
        
        # For language modeling, target is input shifted by 1
        target_ids = torch.cat([input_ids[1:], torch.tensor([self.tokenizer.tokenizer.eos_token_id])])
        
        return {
            'input_ids': input_ids[:-1],  # [seq_len-1]
            'target_ids': target_ids[:-1],  # [seq_len-1]
            'attention_mask': encoded['attention_mask'].squeeze(0)[:-1]
        }

class DomainSpecificDataset(Dataset):
    """Dataset for domain-specific specialist training"""
    
    def __init__(self, domain_data: Dict[str, List[str]], domain_id: int,
                 tokenizer: MambaTokenizer, preprocessor: TextPreprocessor, config):
        self.domain_id = domain_id
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.config = config
        
        # Get domain-specific data
        domain_name = f"domain_{domain_id}"
        self.data = domain_data.get(domain_name, [])
        
        if not self.data:
            # Create synthetic domain data for testing
            self.data = [f"Domain {domain_id} specific text example {i}." 
                        for i in range(100)]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get domain-specific training example"""
        text = self.data[idx]
        
        # Preprocess and tokenize
        clean_text = self.preprocessor.clean_text(text)
        encoded = self.tokenizer.encode(clean_text, max_length=self.config.max_seq_len)
        
        input_ids = encoded['input_ids'].squeeze(0)
        target_ids = torch.cat([input_ids[1:], torch.tensor([self.tokenizer.tokenizer.eos_token_id])])
        
        return {
            'input_ids': input_ids[:-1],
            'target_ids': target_ids[:-1],
            'attention_mask': encoded['attention_mask'].squeeze(0)[:-1],
            'domain_id': self.domain_id
        }

def create_data_loaders(config, tokenizer: MambaTokenizer, 
                       preprocessor: TextPreprocessor) -> Dict[str, DataLoader]:
    """Create data loaders for training"""
    
    # Main training dataset
    train_dataset = MambaDataset(
        data_path=getattr(config, 'train_data_path', 'train_data.txt'),
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        config=config
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Domain-specific datasets for specialist training
    domain_loaders = {}
    
    # Load domain-specific data (placeholder)
    domain_data = {}  # Should load actual domain-specific datasets
    
    for domain_id in range(config.num_specialists):
        domain_dataset = DomainSpecificDataset(
            domain_data=domain_data,
            domain_id=domain_id,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            config=config
        )
        
        domain_loader = DataLoader(
            domain_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        domain_loaders[domain_id] = domain_loader
    
    return {
        'main': train_loader,
        'domains': domain_loaders
    } 