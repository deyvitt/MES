# =============================================================================
# routing/router.py
# =============================================================================
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re
from utils.domain_configs import DomainConfigs

class TopicRouter(nn.Module):
    def __init__(self, config, domain_configs: List[Dict]):
        super().__init__()
        self.config = config
        self.domain_configs = domain_configs
        self.num_specialists = len(domain_configs)
        
        # Build keyword mappings
        self.keyword_to_domains = defaultdict(list)
        self.domain_keywords = {}
        
        for domain in domain_configs:
            domain_id = domain["id"]
            keywords = domain["keywords"]
            self.domain_keywords[domain_id] = keywords
            
            for keyword in keywords:
                self.keyword_to_domains[keyword.lower()].append(domain_id)
        
        # Neural router for complex routing decisions
        self.neural_router = nn.Sequential(
            nn.Linear(config.d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_specialists)
        )
        
        # Text similarity threshold
        self.similarity_threshold = 0.1
        
    def keyword_based_routing(self, text: str) -> Dict[int, float]:
        """Route based on keyword matching"""
        text_lower = text.lower()
        domain_scores = defaultdict(float)
        
        # Count keyword matches for each domain
        for domain_id, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Weight by keyword frequency and length
                    count = text_lower.count(keyword)
                    weight = len(keyword) / 10.0  # Longer keywords get higher weight
                    domain_scores[domain_id] += count * weight
        
        # Normalize scores
        total_score = sum(domain_scores.values())
        if total_score > 0:
            domain_scores = {k: v/total_score for k, v in domain_scores.items()}
        
        return dict(domain_scores)
    
    def neural_routing(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Neural network based routing"""
        # Use mean pooling of embeddings
        pooled = embeddings.mean(dim=1)  # [batch, d_model]
        scores = self.neural_router(pooled)  # [batch, num_specialists]
        return torch.softmax(scores, dim=-1)
    
    def route_text(self, text: str, embeddings: torch.Tensor = None, 
                   max_specialists: int = 10) -> List[Tuple[int, float]]:
        """
        Route text to appropriate specialists
        
        Args:
            text: Input text to route
            embeddings: Text embeddings [1, seq_len, d_model]
            max_specialists: Maximum number of specialists to activate
            
        Returns:
            List of (specialist_id, confidence) tuples
        """
        # Keyword-based routing
        keyword_scores = self.keyword_based_routing(text)
        
        # Neural routing (if embeddings provided)
        neural_scores = {}
        if embeddings is not None:
            neural_weights = self.neural_routing(embeddings)
            neural_scores = {i: float(neural_weights[0, i]) 
                           for i in range(self.num_specialists)}
        
        # Combine scores
        final_scores = {}
        for i in range(self.num_specialists):
            keyword_score = keyword_scores.get(i, 0.0)
            neural_score = neural_scores.get(i, 0.0)
            
            # Weighted combination
            final_scores[i] = 0.7 * keyword_score + 0.3 * neural_score
        
        # Sort by score and take top specialists
        sorted_specialists = sorted(final_scores.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)
        
        # Filter by threshold and limit
        active_specialists = []
        for specialist_id, score in sorted_specialists:
            if score > self.similarity_threshold and len(active_specialists) < max_specialists:
                active_specialists.append((specialist_id, score))
        
        # Ensure at least one specialist is active
        if not active_specialists and sorted_specialists:
            active_specialists = [sorted_specialists[0]]
        
        return active_specialists
    
    def chunk_and_route(self, text: str, chunk_size: int = 512) -> List[Dict]:
        """
        Split text into chunks and route each chunk
        
        Returns:
            List of dicts with 'text', 'specialists', 'chunk_id'
        """
        # Simple sentence-based chunking
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Route current chunk
                specialists = self.route_text(current_chunk)
                chunks.append({
                    'text': current_chunk.strip(),
                    'specialists': specialists,
                    'chunk_id': chunk_id
                })
                current_chunk = sentence
                chunk_id += 1
            else:
                current_chunk += sentence + ". "
        
        # Handle last chunk
        if current_chunk.strip():
            specialists = self.route_text(current_chunk)
            chunks.append({
                'text': current_chunk.strip(),
                'specialists': specialists,
                'chunk_id': chunk_id
            })
        
        return chunks
