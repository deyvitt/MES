# =============================================================================
# routing/tlm_manager.py
# =============================================================================
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from core.model import MambaModel
from core.config import MambaConfig
from utils.domain_configs import DomainConfigs

class SpecialistTLM:
    """Individual Specialist Mamba TLM"""
    def __init__(self, specialist_id: int, config: MambaConfig, domain_info: Dict):
        self.specialist_id = specialist_id
        self.config = config
        self.domain_info = domain_info
        self.model = MambaModel(config)
        self.device = config.device
        
        # Move to device
        self.model.to(self.device)
        
    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode input and return hidden states"""
        self.model.eval()
        with torch.no_grad():
            # Get embeddings
            x = self.model.embedding(input_ids)
            
            # Pass through Mamba layers
            for layer in self.model.layers:
                x = layer(x)
            
            # Apply final norm
            x = self.model.norm_f(x)
            
            # Return pooled representation
            return x.mean(dim=1)  # [batch, d_model]
    
    def get_memory_usage(self) -> int:
        """Get model memory usage in bytes"""
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

class TLMManager:
    """Manages 100 specialist Mamba TLMs"""
    
    def __init__(self, config: MambaConfig):
        self.config = config
        self.device = config.device
        
        # Create domain configurations
        self.domain_configs = DomainConfigs.get_domain_configs(config.num_specialists)
        
        # Initialize specialists
        self.specialists = {}
        self._initialize_specialists()
        
        # Shared components
        self.shared_embedding = None
        if config.shared_embedding:
            self.shared_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.shared_embedding.to(self.device)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=min(32, config.num_specialists))
        
    def _initialize_specialists(self):
        """Initialize all specialist TLMs"""
        print("Initializing 100 specialist TLMs...")
        
        for domain_config in self.domain_configs:
            specialist_id = domain_config["id"]
            
            # Create specialist-specific config
            specialist_config = DomainConfigs.create_specialist_config(
                self.config, specialist_id
            )
            
            # Create specialist TLM
            specialist = SpecialistTLM(
                specialist_id=specialist_id,
                config=specialist_config,
                domain_info=domain_config
            )
            
            self.specialists[specialist_id] = specialist
            
            if specialist_id % 10 == 0:
                print(f"Initialized {specialist_id + 1}/100 specialists")
        
        print("All specialists initialized!")
        
        # Apply weight sharing if enabled
        if self.config.hierarchical_sharing:
            self._apply_weight_sharing()
    
    def _apply_weight_sharing(self):
        """Apply hierarchical weight sharing between specialists"""
        print("Applying hierarchical weight sharing...")
        
        # Share embedding layers
        if self.shared_embedding is not None:
            for specialist in self.specialists.values():
                specialist.model.embedding.token_embedding = self.shared_embedding
        
        # Group specialists by domain similarity and share lower layers
        domain_groups = self._group_domains_by_similarity()
        
        for group in domain_groups:
            if len(group) > 1:
                # Use first specialist's weights as shared weights for the group
                reference_specialist = self.specialists[group[0]]
                shared_layers = reference_specialist.model.layers[:self.config.n_layers//2]
                
                for specialist_id in group[1:]:
                    specialist = self.specialists[specialist_id]
                    for i, layer in enumerate(shared_layers):
                        specialist.model.layers[i] = layer
    
    def _group_domains_by_similarity(self) -> List[List[int]]:
        """Group domains by similarity for weight sharing"""
        # Simple grouping based on domain categories
        groups = {
            'stem': [],
            'programming': [],
            'language': [],
            'business': [],
            'other': []
        }
        
        for domain_config in self.domain_configs:
            domain_name = domain_config["name"].lower()
            specialist_id = domain_config["id"]
            
            if any(x in domain_name for x in ['math', 'physics', 'chemistry', 'biology']):
                groups['stem'].append(specialist_id)
            elif any(x in domain_name for x in ['python', 'javascript', 'systems']):
                groups['programming'].append(specialist_id)
            elif any(x in domain_name for x in ['writing', 'translation']):
                groups['language'].append(specialist_id)
            elif any(x in domain_name for x in ['business', 'legal']):
                groups['business'].append(specialist_id)
            else:
                groups['other'].append(specialist_id)
        
        return [group for group in groups.values() if len(group) > 1]
    
    def encode_parallel(self, routing_results: List[Dict]) -> List[Dict]:
        """
        Encode chunks in parallel using appropriate specialists
        
        Args:
            routing_results: List of routing results from router
            
        Returns:
            List of encoded results with specialist outputs
        """
        futures = []
        
        for chunk_info in routing_results:
            chunk_text = chunk_info['text']
            specialists = chunk_info['specialists']
            chunk_id = chunk_info['chunk_id']
            
            # Create encoding task for each relevant specialist
            for specialist_id, confidence in specialists:
                if specialist_id in self.specialists:
                    future = self.executor.submit(
                        self._encode_chunk,
                        chunk_text,
                        specialist_id,
                        confidence,
                        chunk_id
                    )
                    futures.append(future)
        
        # Collect results
        encoded_results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                encoded_results.append(result)
            except Exception as e:
                print(f"Error in specialist encoding: {e}")
        
        # Group results by chunk_id
        grouped_results = {}
        for result in encoded_results:
            chunk_id = result['chunk_id']
            if chunk_id not in grouped_results:
                grouped_results[chunk_id] = []
            grouped_results[chunk_id].append(result)
        
        return grouped_results
    
    def _encode_chunk(self, text: str, specialist_id: int, confidence: float, 
                     chunk_id: int) -> Dict:
        """Encode a single chunk with a specific specialist"""
        try:
            specialist = self.specialists[specialist_id]
            
            # Tokenize text (simplified - should use proper tokenizer)
            # This is a placeholder - integrate with actual tokenizer
            input_ids = torch.randint(0, 1000, (1, 100)).to(self.device)
            
            # Encode with specialist
            encoding = specialist.encode(input_ids)
            
            return {
                'chunk_id': chunk_id,
                'specialist_id': specialist_id,
                'confidence': confidence,
                'encoding': encoding,
                'domain': specialist.domain_info['name']
            }
            
        except Exception as e:
            print(f"Error encoding chunk {chunk_id} with specialist {specialist_id}: {e}")
            return None
    
    def get_active_specialists(self) -> List[int]:
        """Get list of currently active specialist IDs"""
        return list(self.specialists.keys())
    
    def get_specialist_info(self, specialist_id: int) -> Dict:
        """Get information about a specific specialist"""
        if specialist_id in self.specialists:
            specialist = self.specialists[specialist_id]
            return {
                'id': specialist_id,
                'domain': specialist.domain_info,
                'params': specialist.model.get_num_params(),
                'memory': specialist.get_memory_usage()
            }
        return None
    
    def get_total_parameters(self) -> int:
        """Get total parameters across all specialists"""
        total = 0
        for specialist in self.specialists.values():
            total += specialist.model.get_num_params()
        return total 