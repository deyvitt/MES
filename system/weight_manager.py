# =============================================================================
# system/weight_manager.py
# =============================================================================
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import os
from pathlib import Path

class WeightManager:
    """Manages hierarchical weight sharing and loading/saving"""
    
    def __init__(self, config, tlm_manager):
        self.config = config
        self.tlm_manager = tlm_manager
        
        # Track shared weights
        self.shared_embeddings = None
        self.shared_foundation_layers = {}
        
    def setup_hierarchical_sharing(self):
        """Setup hierarchical weight sharing between specialists"""
        print("Setting up hierarchical weight sharing...")
        
        # Create shared embedding if enabled
        if self.config.shared_embedding:
            self.shared_embeddings = nn.Embedding(
                self.config.vocab_size, 
                self.config.d_model
            ).to(self.config.device)
            
            # Share embedding across all specialists
            for specialist in self.tlm_manager.specialists.values():
                specialist.model.embedding.token_embedding = self.shared_embeddings
        
        # Setup foundation layer sharing
        self._setup_foundation_sharing()
        
        print("Hierarchical weight sharing setup complete!")
    
    def _setup_foundation_sharing(self):
        """Setup sharing of foundation layers"""
        num_shared_layers = self.config.n_layers // 2
        
        # Group specialists by domain similarity
        domain_groups = self._group_specialists_by_domain()
        
        for group_name, specialist_ids in domain_groups.items():
            if len(specialist_ids) > 1:
                # Create shared foundation layers for this group
                reference_specialist = self.tlm_manager.specialists[specialist_ids[0]]
                shared_layers = reference_specialist.model.layers[:num_shared_layers]
                
                # Share with other specialists in the group
                for specialist_id in specialist_ids[1:]:
                    specialist = self.tlm_manager.specialists[specialist_id]
                    for i in range(num_shared_layers):
                        specialist.model.layers[i] = shared_layers[i]
                
                self.shared_foundation_layers[group_name] = shared_layers
    
    def _group_specialists_by_domain(self) -> Dict[str, List[int]]:
        """Group specialists by domain for weight sharing"""
        domain_groups = {
            'stem': [],
            'programming': [],
            'language': [],
            'business': [],
            'general': []
        }
        
        for specialist_id, specialist in self.tlm_manager.specialists.items():
            domain_name = specialist.domain_info['name'].lower()
            
            if any(x in domain_name for x in ['math', 'physics', 'chemistry', 'biology']):
                domain_groups['stem'].append(specialist_id)
            elif any(x in domain_name for x in ['python', 'javascript', 'systems']):
                domain_groups['programming'].append(specialist_id)
            elif any(x in domain_name for x in ['writing', 'translation']):
                domain_groups['language'].append(specialist_id)
            elif any(x in domain_name for x in ['business', 'legal']):
                domain_groups['business'].append(specialist_id)
            else:
                domain_groups['general'].append(specialist_id)
        
        return {k: v for k, v in domain_groups.items() if len(v) > 1}
    
    def save_weights(self, save_path: str):
        """Save all weights with hierarchical structure"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save shared embeddings
        if self.shared_embeddings is not None:
            torch.save(
                self.shared_embeddings.state_dict(),
                save_path / "shared_embeddings.pt"
            )
        
        # Save shared foundation layers
        for group_name, layers in self.shared_foundation_layers.items():
            group_state = {}
            for i, layer in enumerate(layers):
                group_state[f"layer_{i}"] = layer.state_dict()
            torch.save(group_state, save_path / f"shared_foundation_{group_name}.pt")
        
        # Save specialist-specific weights
        specialists_path = save_path / "specialists"
        specialists_path.mkdir(exist_ok=True)
        
        for specialist_id, specialist in self.tlm_manager.specialists.items():
            torch.save(
                specialist.model.state_dict(),
                specialists_path / f"specialist_{specialist_id}.pt"
            )
        
        print(f"Weights saved to {save_path}")
    
    def load_weights(self, load_path: str):
        """Load weights with hierarchical structure"""
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Weight path {load_path} not found")
        
        # Load shared embeddings
        embeddings_path = load_path / "shared_embeddings.pt"
        if embeddings_path.exists() and self.shared_embeddings is not None:
            self.shared_embeddings.load_state_dict(torch.load(embeddings_path))
        
        # Load shared foundation layers
        for group_name in self.shared_foundation_layers.keys():
            foundation_path = load_path / f"shared_foundation_{group_name}.pt"
            if foundation_path.exists():
                group_state = torch.load(foundation_path)
                for i, layer in enumerate(self.shared_foundation_layers[group_name]):
                    if f"layer_{i}" in group_state:
                        layer.load_state_dict(group_state[f"layer_{i}"])
        
        # Load specialist weights
        specialists_path = load_path / "specialists"
        if specialists_path.exists():
            for specialist_id, specialist in self.tlm_manager.specialists.items():
                specialist_path = specialists_path / f"specialist_{specialist_id}.pt"
                if specialist_path.exists():
                    specialist.model.load_state_dict(torch.load(specialist_path))
        
        print(f"Weights loaded from {load_path}")
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage breakdown"""
        usage = {}
        
        # Shared embedding memory
        if self.shared_embeddings is not None:
            usage['shared_embeddings'] = sum(
                p.numel() * p.element_size() 
                for p in self.shared_embeddings.parameters()
            )
        
        # Shared foundation layer memory
        total_foundation = 0
        for layers in self.shared_foundation_layers.values():
            for layer in layers:
                total_foundation += sum(
                    p.numel() * p.element_size()
                    for p in layer.parameters()
                )