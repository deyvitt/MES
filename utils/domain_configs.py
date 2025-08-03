# =============================================================================
# utils/domain_configs.py
# =============================================================================
from typing import Dict, List
from core.config import MambaConfig

class DomainConfigs:
    """Configurations for different specialist domains"""
    
    DOMAINS = {
        # STEM domains
        "mathematics": {
            "keywords": ["equation", "theorem", "proof", "calculate", "derivative", "integral", "matrix", "algebra", "geometry", "statistics"],
            "description": "Mathematical reasoning and computation"
        },
        "physics": {
            "keywords": ["force", "energy", "momentum", "quantum", "relativity", "particle", "wave", "thermodynamics", "mechanics"],
            "description": "Physics concepts and problems"
        },
        "chemistry": {
            "keywords": ["molecule", "atom", "reaction", "compound", "bond", "element", "organic", "inorganic", "catalyst"],
            "description": "Chemistry and molecular science"
        },
        "biology": {
            "keywords": ["cell", "DNA", "protein", "organism", "evolution", "genetics", "ecology", "anatomy", "physiology"],
            "description": "Biological sciences"
        },
        
        # Programming domains
        "python": {
            "keywords": ["def", "class", "import", "python", "pandas", "numpy", "matplotlib", "sklearn", "tensorflow"],
            "description": "Python programming and data science"
        },
        "javascript": {
            "keywords": ["function", "var", "let", "const", "javascript", "react", "node", "async", "promise"],
            "description": "JavaScript and web development"
        },
        "systems": {
            "keywords": ["linux", "server", "network", "database", "docker", "kubernetes", "cloud", "devops"],
            "description": "Systems programming and infrastructure"
        },
        
        # Language domains
        "writing": {
            "keywords": ["essay", "article", "story", "paragraph", "thesis", "narrative", "prose", "literature"],
            "description": "Creative and technical writing"
        },
        "translation": {
            "keywords": ["translate", "language", "spanish", "french", "german", "chinese", "japanese", "korean"],
            "description": "Language translation and linguistics"
        },
        
        # Business domains
        "business": {
            "keywords": ["market", "strategy", "finance", "management", "revenue", "profit", "customer", "sales"],
            "description": "Business and economics"
        },
        "legal": {
            "keywords": ["law", "contract", "court", "legal", "attorney", "judge", "case", "statute", "regulation"],
            "description": "Legal reasoning and analysis"
        },
        
        # Other domains
        "history": {
            "keywords": ["war", "empire", "civilization", "century", "ancient", "medieval", "revolution", "dynasty"],
            "description": "Historical knowledge and analysis"
        },
        "philosophy": {
            "keywords": ["ethics", "moral", "logic", "metaphysics", "epistemology", "consciousness", "existence"],
            "description": "Philosophical reasoning"
        },
        "medical": {
            "keywords": ["patient", "diagnosis", "treatment", "disease", "medicine", "surgery", "therapy", "symptom"],
            "description": "Medical knowledge and healthcare"
        },
        "arts": {
            "keywords": ["painting", "music", "sculpture", "artist", "gallery", "museum", "aesthetic", "culture"],
            "description": "Arts and cultural topics"
        }
    }
    
    @classmethod
    def get_domain_configs(cls, num_specialists: int = 100) -> List[Dict]:
        """Generate configurations for specialist domains"""
        configs = []
        base_domains = list(cls.DOMAINS.keys())
        
        # Create configurations
        for i in range(num_specialists):
            if i < len(base_domains):
                # Use predefined domains
                domain_name = base_domains[i]
                domain_info = cls.DOMAINS[domain_name]
            else:
                # Create sub-specializations or general domains
                base_idx = i % len(base_domains)
                domain_name = f"{base_domains[base_idx]}_sub_{i}"
                domain_info = cls.DOMAINS[base_domains[base_idx]]
            
            config = {
                "id": i,
                "name": domain_name,
                "keywords": domain_info["keywords"],
                "description": domain_info["description"],
                "weight": 1.0  # Can be adjusted based on importance
            }
            configs.append(config)
        
        return configs
    
    @classmethod
    def create_specialist_config(cls, base_config: MambaConfig, domain_id: int) -> MambaConfig:
        """Create a specialist configuration for a specific domain"""
        specialist_config = MambaConfig(**base_config.__dict__)
        specialist_config.specialist_id = domain_id
        return specialist_config 