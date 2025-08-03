# =============================================================================
# system/mambaSwarm.py - Unified Scalable Mamba Swarm Engine
# =============================================================================
import torch
import time
import os
import asyncio
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer

# Core imports
from core.config import MambaConfig, MambaSwarmConfig, auto_detect_tier
from core.tokenizer import MambaTokenizer
from core.preprocess import TextPreprocessor
from core.model import MambaModel
from core.mamba_swarm_integration import MambaEncoderSwarmModel, create_swarm_from_existing_config

# Routing imports
from routing.router import TopicRouter, ContentBasedRouter
from routing.tlm_manager import TLMManager
from routing.aggregator import AttentionAggregator, WeightedAggregator
from utils.domain_configs import DomainConfigs


class UnifiedMambaSwarm:
    """
    Unified Mamba Swarm Engine combining the best of both architectures:
    - Scalable tier-based system with auto-detection
    - Production-ready async processing and monitoring
    - Graceful fallback to simulation mode
    - Support for both custom and pre-trained models
    """
    
    def __init__(self, 
                 tier: Optional[str] = None,
                 config: Optional[Union[MambaConfig, MambaSwarmConfig]] = None,
                 use_pretrained: bool = True,
                 config_override: Optional[Dict] = None):
        """
        Initialize the unified swarm engine
        
        Args:
            tier: Scaling tier (demo/small/medium/large/full) or None for auto-detect
            config: Either MambaConfig for custom models or MambaSwarmConfig for scaling
            use_pretrained: Whether to use HuggingFace pretrained models
            config_override: Dictionary to override config settings
        """
        # Auto-detect tier if not specified
        if tier is None:
            tier = auto_detect_tier()
            print(f"Auto-detected tier: {tier}")
        
        self.tier = tier
        self.use_pretrained = use_pretrained
        
        # Initialize configuration
        if config is None:
            if use_pretrained:
                self.swarm_config = MambaSwarmConfig(tier=tier)
                if config_override:
                    self.swarm_config.config.update(config_override)
                self.config = self._create_legacy_config()
            else:
                # Use custom config for legacy components
                self.config = MambaConfig()  # Default config
                self.swarm_config = None
        else:
            if isinstance(config, MambaSwarmConfig):
                self.swarm_config = config
                self.config = self._create_legacy_config()
            else:
                self.config = config
                self.swarm_config = None
        
        self.device = getattr(self.config, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # System properties
        if self.swarm_config:
            self.num_encoders = self.swarm_config.config["num_encoders"]
            self.encoder_size = self.swarm_config.config["encoder_size"]
        else:
            self.num_encoders = getattr(self.config, 'num_specialists', 5)
            self.encoder_size = "130M"
        
        # Initialize components
        self.encoders = []
        self.tokenizer = None
        self.preprocessor = None
        self.router = None
        self.aggregator = None
        self.tlm_manager = None
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'total_tokens_processed': 0,
            'avg_response_time': 0.0,
            'specialist_usage': {i: 0 for i in range(self.num_encoders)},
            'simulation_mode': False,
            'model_load_errors': 0
        }
        
        # Initialize system
        self._initialize_system()
        
        print(f"✅ Unified Mamba Swarm initialized: {self.tier} tier, {self.num_encoders} encoders")
    
    def _create_legacy_config(self) -> MambaConfig:
        """Create legacy MambaConfig from SwarmConfig for compatibility"""
        legacy_config = MambaConfig()
        if self.swarm_config:
            legacy_config.num_specialists = self.swarm_config.config["num_encoders"]
            legacy_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return legacy_config
    
    def _initialize_system(self):
        """Initialize the complete swarm system"""
        try:
            # Initialize tokenizer and preprocessor
            self._initialize_tokenizer()
            self._initialize_preprocessor()
            
            # Initialize encoders/specialists
            if self.use_pretrained:
                self._initialize_pretrained_encoders()
            else:
                self._initialize_custom_specialists()
            
            # Initialize routing system
            self._initialize_routing()
            
            # Initialize aggregation system
            self._initialize_aggregation()
            
            print(f"🚀 System initialization complete!")
            
        except Exception as e:
            print(f"⚠️  Error during initialization: {e}")
            self._fallback_to_simulation()
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer based on mode"""
        if self.use_pretrained:
            base_model_name = self._get_base_model_name()
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"📝 Loaded HuggingFace tokenizer: {base_model_name}")
            except:
                print("⚠️  HuggingFace tokenizer failed, using custom tokenizer")
                self.tokenizer = MambaTokenizer(self.config)
        else:
            self.tokenizer = MambaTokenizer(self.config)
    
    def _initialize_preprocessor(self):
        """Initialize text preprocessor"""
        self.preprocessor = TextPreprocessor(self.config)
    
    def _get_base_model_name(self):
        """Get the appropriate base model for current tier"""
        model_mapping = {
            "130M": "state-spaces/mamba-130m",
            "370M": "state-spaces/mamba-370m", 
            "790M": "state-spaces/mamba-790m",
            "1.4B": "state-spaces/mamba-1.4b",
            "2.8B": "state-spaces/mamba-2.8b"
        }
        return model_mapping.get(self.encoder_size, "state-spaces/mamba-130m")
    
    def _initialize_pretrained_encoders(self):
        """Initialize pretrained encoder swarm"""
        print(f"🔄 Loading {self.num_encoders} pretrained encoders...")
        
        base_model_name = self._get_base_model_name()
        
        try:
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if self.num_encoders > 5 else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            # Create encoder instances
            for i in range(self.num_encoders):
                domain_info = self.swarm_config.domain_assignments[i] if self.swarm_config else {
                    "domain": f"general_{i}", "specialty": "general"
                }
                
                if self.tier == "demo" or self.num_encoders <= 5:
                    # Share model instance for smaller configurations
                    encoder = {
                        "id": i,
                        "model": base_model,
                        "domain": domain_info["domain"],
                        "specialty": domain_info["specialty"],
                        "shared": True
                    }
                else:
                    # Separate instances for larger configurations
                    encoder = {
                        "id": i,
                        "model": AutoModelForCausalLM.from_pretrained(
                            base_model_name,
                            torch_dtype=torch.float16,
                            device_map="auto"
                        ),
                        "domain": domain_info["domain"],
                        "specialty": domain_info["specialty"],
                        "shared": False
                    }
                
                self.encoders.append(encoder)
                print(f"  ✓ Encoder {i}: {encoder['domain']} specialist")
                
        except Exception as e:
            print(f"❌ Failed to load pretrained models: {e}")
            self.stats['model_load_errors'] += 1
            self._create_simulated_encoders()
    
    def _initialize_custom_specialists(self):
        """Initialize custom TLM specialists or native Mamba swarm"""
        try:
            if hasattr(self, 'use_native_swarm') and self.use_native_swarm:
                # Use the native Mamba swarm integration
                self.native_swarm_model = create_swarm_from_existing_config(
                    self.config, num_encoders=self.num_encoders
                )
                print(f"✓ Initialized native Mamba swarm with {self.num_encoders} encoders")
            else:
                # Use TLM manager (legacy approach)
                self.tlm_manager = TLMManager(self.config)
                print(f"✓ Initialized {self.num_encoders} custom specialists")
        except Exception as e:
            print(f"⚠️  Custom specialists failed: {e}")
            self._create_simulated_encoders()
    
    def _create_simulated_encoders(self):
        """Create simulated encoders for demonstration/fallback"""
        print("🎭 Creating simulated encoders...")
        self.stats['simulation_mode'] = True
        
        for i in range(self.num_encoders):
            domain_info = self.swarm_config.domain_assignments[i] if self.swarm_config else {
                "domain": f"general_{i}", "specialty": "general"
            }
            
            encoder = {
                "id": i,
                "model": None,
                "domain": domain_info["domain"],
                "specialty": domain_info["specialty"],
                "simulated": True
            }
            self.encoders.append(encoder)
    
    def _initialize_routing(self):
        """Initialize routing system"""
        try:
            if self.use_pretrained and self.swarm_config:
                # Use content-based router for pretrained models
                router_config = self.swarm_config.get_router_config()
                self.router = ContentBasedRouter(
                    num_encoders=self.num_encoders,
                    domain_assignments=self.swarm_config.domain_assignments,
                    config=router_config
                )
            else:
                # Use topic router for custom models
                domain_configs = DomainConfigs.get_domain_configs(self.num_encoders)
                self.router = TopicRouter(self.config, domain_configs)
                if hasattr(self.router, 'to'):
                    self.router.to(self.device)
            
            print("🧭 Router initialized")
            
        except Exception as e:
            print(f"⚠️  Router initialization failed: {e}")
            # Create basic fallback router
            self.router = self._create_fallback_router()
    
    def _initialize_aggregation(self):
        """Initialize aggregation system"""
        try:
            if self.use_pretrained:
                self.aggregator = WeightedAggregator(
                    num_encoders=self.num_encoders,
                    hidden_dim=768
                )
            else:
                self.aggregator = AttentionAggregator(self.config)
                if hasattr(self.aggregator, 'to'):
                    self.aggregator.to(self.device)
            
            print("🔄 Aggregator initialized")
            
        except Exception as e:
            print(f"⚠️  Aggregator initialization failed: {e}")
            self.aggregator = None
    
    def _create_fallback_router(self):
        """Create a simple fallback router"""
        class FallbackRouter:
            def __init__(self, num_encoders):
                self.num_encoders = num_encoders
            
            def route(self, text):
                # Simple round-robin routing
                import random
                num_selected = min(3, self.num_encoders)
                return {
                    "selected_encoders": random.sample(range(self.num_encoders), num_selected)
                }
            
            def chunk_and_route(self, text):
                return [{"specialists": [(0, 1.0)], "chunk": text}]
        
        return FallbackRouter(self.num_encoders)
    
    def _fallback_to_simulation(self):
        """Complete fallback to simulation mode"""
        print("🎭 Entering full simulation mode")
        self.stats['simulation_mode'] = True
        self._create_simulated_encoders()
        if not self.router:
            self.router = self._create_fallback_router()
    
    # =============================================================================
    # MAIN PROCESSING METHODS
    # =============================================================================
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.7, 
                show_routing: bool = True) -> Dict:
        """
        Generate response using the swarm (from swarmEngine2 style)
        
        Args:
            prompt: Input text prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            show_routing: Whether to display routing information
            
        Returns:
            Dict with response and metadata
        """
        start_time = time.time()
        
        try:
            # Route to appropriate encoders
            if hasattr(self.router, 'route'):
                routing_decision = self.router.route(prompt)
                selected_encoders = routing_decision.get("selected_encoders", [0])
            else:
                # Fallback routing
                selected_encoders = [0]
            
            if show_routing:
                print(f"🔀 Routing: Selected {len(selected_encoders)} encoders")
                for enc_id in selected_encoders[:3]:
                    if enc_id < len(self.encoders):
                        domain = self.encoders[enc_id]["domain"]
                        print(f"   Encoder {enc_id}: {domain}")
            
            # Generate response
            if self.stats['simulation_mode'] or any(enc.get("simulated") for enc in self.encoders):
                response = self._simulate_generation(prompt, selected_encoders, max_length)
            else:
                response = self._real_generation(prompt, selected_encoders, max_length, temperature)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats_simple(prompt, selected_encoders, processing_time)
            
            return {
                "response": response,
                "processing_time": processing_time,
                "routing_info": {
                    "selected_encoders": selected_encoders,
                    "num_active": len(selected_encoders),
                    "total_encoders": self.num_encoders,
                    "domains": [self.encoders[i]["domain"] for i in selected_encoders 
                               if i < len(self.encoders)]
                },
                "success": True
            }
            
        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "processing_time": time.time() - start_time,
                "success": False,
                "error": str(e)
            }
    
    def process_request(self, text: str, max_new_tokens: int = 100) -> Dict:
        """
        Process request using traditional pipeline (from swarm_engine style)
        
        Args:
            text: Input text to process
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dict with response and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Preprocess input
            if self.preprocessor:
                clean_text = self.preprocessor.clean_text(text)
            else:
                clean_text = text
            
            # Step 2: Route to specialists
            if hasattr(self.router, 'chunk_and_route'):
                routing_results = self.router.chunk_and_route(clean_text)
            else:
                # Fallback for content-based router
                routing_decision = self.router.route(clean_text)
                routing_results = [{"specialists": [(enc_id, 1.0) for enc_id in routing_decision["selected_encoders"]], 
                                 "chunk": clean_text}]
            
            # Step 3: Process chunks
            if self.tlm_manager and not self.stats['simulation_mode']:
                specialist_outputs = self.tlm_manager.encode_parallel(routing_results)
            else:
                # Simulate processing
                specialist_outputs = [{"response": f"Processed chunk: {res['chunk'][:50]}..."} 
                                    for res in routing_results]
            
            # Step 4: Aggregate results
            if self.aggregator and not self.stats['simulation_mode']:
                response = self.aggregator.generate_response(specialist_outputs, max_new_tokens)
            else:
                # Simple aggregation fallback
                response = " ".join([out.get("response", "") for out in specialist_outputs])
            
            # Update stats
            processing_time = time.time() - start_time
            self._update_stats(text, routing_results, processing_time)
            
            return {
                'response': response,
                'processing_time': processing_time,
                'chunks_processed': len(routing_results),
                'specialists_used': self._get_specialists_used(routing_results),
                'success': True
            }
            
        except Exception as e:
            return {
                'response': f"Error processing request: {str(e)}",
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    # =============================================================================
    # ASYNC AND BATCH PROCESSING
    # =============================================================================
    
    async def process_request_async(self, text: str, max_new_tokens: int = 100) -> Dict:
        """Async version of process_request"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor, self.process_request, text, max_new_tokens
            )
        
        return result
    
    async def generate_async(self, prompt: str, max_length: int = 100, 
                           temperature: float = 0.7) -> Dict:
        """Async version of generate"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor, self.generate, prompt, max_length, temperature, False
            )
        
        return result
    
    def batch_process(self, texts: List[str], max_new_tokens: int = 100, 
                     method: str = "process") -> List[Dict]:
        """
        Process multiple texts in batch
        
        Args:
            texts: List of input texts
            max_new_tokens: Maximum tokens to generate
            method: "process" or "generate" for processing method
        """
        results = []
        
        for text in texts:
            if method == "generate":
                result = self.generate(text, max_new_tokens, show_routing=False)
            else:
                result = self.process_request(text, max_new_tokens)
            results.append(result)
        
        return results
    
    # =============================================================================
    # GENERATION METHODS
    # =============================================================================
    
    def _simulate_generation(self, prompt: str, selected_encoders: List[int], max_length: int) -> str:
        """Simulate generation for demo/fallback purposes"""
        import random
        
        # Determine response type based on selected encoder domains
        domains = [self.encoders[i]["domain"] for i in selected_encoders if i < len(self.encoders)]
        
        if any("code" in domain.lower() for domain in domains):
            return f"Here's a solution for '{prompt[:30]}...':\n\n```python\ndef solution():\n    # Implementation here\n    return result\n```"
        elif any("medical" in domain.lower() for domain in domains):
            return f"Regarding '{prompt[:30]}...': This medical topic requires careful consideration. Please consult healthcare professionals."
        elif any("science" in domain.lower() for domain in domains):
            return f"From a scientific perspective on '{prompt[:30]}...': Current research indicates several key factors..."
        else:
            return f"Thank you for asking about '{prompt[:30]}...'. Based on expertise from {len(selected_encoders)} specialized domains, here's a comprehensive response..."
    
    def _real_generation(self, prompt: str, selected_encoders: List[int], 
                        max_length: int, temperature: float) -> str:
        """Real generation using loaded models"""
        if not selected_encoders or selected_encoders[0] >= len(self.encoders):
            return "No valid encoders available for generation."
        
        try:
            # Use primary encoder for generation
            primary_encoder = self.encoders[selected_encoders[0]]
            
            if primary_encoder.get("simulated") or not primary_encoder["model"]:
                return self._simulate_generation(prompt, selected_encoders, max_length)
            
            # Tokenize input
            if hasattr(self.tokenizer, 'encode'):
                inputs = self.tokenizer(prompt, return_tensors="pt")
            else:
                # Fallback tokenization
                return self._simulate_generation(prompt, selected_encoders, max_length)
            
            # Generate with model
            with torch.no_grad():
                outputs = primary_encoder["model"].generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 0
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove original prompt from response
            response = response[len(prompt):].strip()
            
            return response if response else "Generated response was empty."
            
        except Exception as e:
            print(f"⚠️  Real generation failed: {e}")
            return self._simulate_generation(prompt, selected_encoders, max_length)
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def _get_specialists_used(self, routing_results: List[Dict]) -> List[int]:
        """Extract specialist IDs used in routing"""
        specialists_used = set()
        
        for chunk_info in routing_results:
            if 'specialists' in chunk_info:
                for specialist_id, _ in chunk_info['specialists']:
                    specialists_used.add(specialist_id)
        
        return list(specialists_used)
    
    def _update_stats(self, text: str, routing_results: List[Dict], processing_time: float):
        """Update detailed performance statistics"""
        self.stats['total_requests'] += 1
        self.stats['total_tokens_processed'] += len(text.split())
        
        # Update average response time
        prev_avg = self.stats['avg_response_time']
        n = self.stats['total_requests']
        self.stats['avg_response_time'] = (prev_avg * (n-1) + processing_time) / n
        
        # Update specialist usage
        specialists_used = self._get_specialists_used(routing_results)
        for specialist_id in specialists_used:
            if specialist_id in self.stats['specialist_usage']:
                self.stats['specialist_usage'][specialist_id] += 1
    
    def _update_stats_simple(self, text: str, selected_encoders: List[int], processing_time: float):
        """Update simple statistics for generate method"""
        self.stats['total_requests'] += 1
        self.stats['total_tokens_processed'] += len(text.split())
        
        # Update average response time
        prev_avg = self.stats['avg_response_time']
        n = self.stats['total_requests']
        self.stats['avg_response_time'] = (prev_avg * (n-1) + processing_time) / n
        
        # Update encoder usage
        for enc_id in selected_encoders:
            if enc_id in self.stats['specialist_usage']:
                self.stats['specialist_usage'][enc_id] += 1
    
    # =============================================================================
    # SCALING AND MANAGEMENT
    # =============================================================================
    
    def scale_up(self, new_tier: str):
        """Scale up to a higher tier"""
        if new_tier not in ["demo", "small", "medium", "large", "full"]:
            raise ValueError(f"Invalid tier: {new_tier}")
        
        print(f"🚀 Scaling from {self.tier} to {new_tier}")
        
        # Preserve current stats
        old_stats = self.stats.copy()
        
        # Reinitialize with new tier
        self.__init__(tier=new_tier, use_pretrained=self.use_pretrained)
        
        # Restore relevant stats
        self.stats['total_requests'] = old_stats['total_requests']
        self.stats['total_tokens_processed'] = old_stats['total_tokens_processed']
        self.stats['avg_response_time'] = old_stats['avg_response_time']
    
    def get_system_info(self) -> Dict:
        """Get comprehensive system information"""
        info = {
            "tier": self.tier,
            "num_encoders": self.num_encoders,
            "encoder_size": self.encoder_size,
            "use_pretrained": self.use_pretrained,
            "simulation_mode": self.stats['simulation_mode'],
            "device": self.device,
            "domains": list(set(enc["domain"] for enc in self.encoders)),
        }
        
        if self.swarm_config:
            info.update({
                "total_parameters": self.swarm_config.config["total_params"],
                "memory_estimate": self.swarm_config.config["memory_estimate"],
                "hardware_recommendation": self.swarm_config.config["hardware"]
            })
        
        return info
    
    def get_stats(self) -> Dict:
        """Get current performance statistics"""
        return self.stats.copy()
    
    def load_models(self, checkpoint_path: str):
        """Load trained models from checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load aggregator
            if self.aggregator and 'aggregator_state' in checkpoint:
                self.aggregator.load_state_dict(checkpoint['aggregator_state'])
            
            # Load specialists (if using custom models)
            if self.tlm_manager and 'specialist_states' in checkpoint:
                for specialist_id, state_dict in checkpoint['specialist_states'].items():
                    if specialist_id in self.tlm_manager.specialists:
                        self.tlm_manager.specialists[specialist_id].model.load_state_dict(state_dict)
            
            print(f"✅ Models loaded from {checkpoint_path}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def set_eval_mode(self):
        """Set all models to evaluation mode"""
        if self.tlm_manager:
            for specialist in self.tlm_manager.specialists.values():
                if hasattr(specialist, 'model'):
                    specialist.model.eval()
        
        if self.aggregator and hasattr(self.aggregator, 'eval'):
            self.aggregator.eval()
        
        if self.router and hasattr(self.router, 'eval'):
            self.router.eval()
        
        # Set pretrained encoders to eval mode
        for encoder in self.encoders:
            if encoder.get("model") and hasattr(encoder["model"], 'eval'):
                encoder["model"].eval()
    
    def set_train_mode(self):
        """Set all models to training mode"""
        if self.tlm_manager:
            for specialist in self.tlm_manager.specialists.values():
                if hasattr(specialist, 'model'):
                    specialist.model.train()
        
        if self.aggregator and hasattr(self.aggregator, 'train'):
            self.aggregator.train()
        
        if self.router and hasattr(self.router, 'train'):
            self.router.train()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_mamba_swarm(tier: str = "auto", use_pretrained: bool = True, 
                      config_override: Optional[Dict] = None) -> UnifiedMambaSwarm:
    """
    Factory function to create appropriately configured swarm
    
    Args:
        tier: Scaling tier or "auto" for auto-detection
        use_pretrained: Whether to use pretrained HuggingFace models
        config_override: Dictionary to override default config
    
    Returns:
        Configured UnifiedMambaSwarm instance
    """
    if tier == "auto":
        tier = auto_detect_tier()
    
    return UnifiedMambaSwarm(
        tier=tier, 
        use_pretrained=use_pretrained,
        config_override=config_override
    )


def create_production_swarm(tier: str = "medium") -> UnifiedMambaSwarm:
    """Create production-ready swarm with optimal settings"""
    return UnifiedMambaSwarm(
        tier=tier,
        use_pretrained=True,
        config_override={
            "batch_size": 32,
            "max_sequence_length": 2048
        }
    )


def create_development_swarm() -> UnifiedMambaSwarm:
    """Create development swarm with simulation fallback"""
    return UnifiedMambaSwarm(
        tier="demo",
        use_pretrained=True,
        config_override={
            "simulation_fallback": True
        }
    )


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🧪 Testing Unified Mamba Swarm...")
    
    # Create swarm instance
    swarm = create_mamba_swarm(tier="demo")
    
    # Display system info
    print("\n📊 System Information:")
    info = swarm.get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test both processing methods
    test_prompts = [
        "Write a Python function to calculate fibonacci numbers",
        "Explain the process of photosynthesis",
        "What are the symptoms of diabetes?"
    ]
    
    print("\n🧪 Testing generate method:")
    for prompt in test_prompts[:2]:
        result = swarm.generate(prompt, max_length=150)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {result['response'][:100]}...")
        print(f"Processing time: {result['processing_time']:.3f}s")
        print(f"Routing: {result['routing_info']['domains']}")
    
    print("\n🧪 Testing process_request method:")
    result = swarm.process_request(test_prompts[2])
    print(f"Response: {result['response'][:100]}...")
    print(f"Success: {result['success']}")
    
    # Test batch processing
    print("\n🧪 Testing batch processing:")
    batch_results = swarm.batch_process(test_prompts, method="generate")
    print(f"Processed {len(batch_results)} requests in batch")
    
    # Display final stats
    print("\n📈 Final Statistics:")
    stats = swarm.get_stats()
    for key, value in stats.items():
        if key != 'specialist_usage':
            print(f"  {key}: {value}")
    
    print("\n✅ Testing complete!")
