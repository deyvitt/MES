"""
Model Evaluator for Mamba Swarm
Comprehensive evaluation system for model performance and quality
"""

import time
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict
import math
import re
from datetime import datetime
from pathlib import Path
import asyncio
import concurrent.futures

# Evaluation metrics
@dataclass
class EvaluationResult:
    metric_name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class BenchmarkResult:
    benchmark_name: str
    overall_score: float
    individual_metrics: List[EvaluationResult]
    execution_time: float
    model_info: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

"""
Model Evaluator for Mamba Swarm
Comprehensive evaluation system for model performance and quality
"""

import time
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict
import math
import re
from datetime import datetime
from pathlib import Path
import asyncio
import concurrent.futures

# Evaluation metrics
@dataclass
class EvaluationResult:
    metric_name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class BenchmarkResult:
    benchmark_name: str
    overall_score: float
    individual_metrics: List[EvaluationResult]
    execution_time: float
    model_info: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

class PerplexityCalculator:
    """Calculate perplexity for language models"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def calculate_perplexity(self, text: str, max_length: int = 512) -> float:
        """Calculate perplexity for given text"""
        # Tokenize text
        tokens = self.tokenizer.encode(text, return_tensors="pt", max_length=max_length, truncation=True)
        tokens = tokens.to(self.device)
        
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(tokens)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Calculate cross-entropy loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tokens[..., 1:].contiguous()
            
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Convert to perplexity
            perplexity = torch.exp(loss)
            
        return perplexity.item()

class BLEUScore:
    """BLEU score calculator for text generation"""
    
    def __init__(self, n_grams: int = 4):
        self.n_grams = n_grams
    
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score between reference and candidate"""
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)
        
        if len(cand_tokens) == 0:
            return 0.0
        
        # Calculate n-gram precisions
        precisions = []
        for n in range(1, self.n_grams + 1):
            precision = self._calculate_n_gram_precision(ref_tokens, cand_tokens, n)
            precisions.append(precision)
        
        # Brevity penalty
        bp = self._brevity_penalty(len(ref_tokens), len(cand_tokens))
        
        # Calculate BLEU score
        if 0 in precisions:
            return 0.0
        
        log_precisions = [math.log(p) for p in precisions]
        bleu = bp * math.exp(sum(log_precisions) / len(log_precisions))
        
        return bleu
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.lower().split()
    
    def _calculate_n_gram_precision(self, ref_tokens: List[str], cand_tokens: List[str], n: int) -> float:
        """Calculate n-gram precision"""
        if len(cand_tokens) < n:
            return 0.0
        
        # Get n-grams
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        cand_ngrams = self._get_ngrams(cand_tokens, n)
        
        if len(cand_ngrams) == 0:
            return 0.0
        
        # Count matches
        matches = 0
        for ngram in cand_ngrams:
            if ngram in ref_ngrams:
                matches += min(cand_ngrams[ngram], ref_ngrams[ngram])
        
        return matches / sum(cand_ngrams.values())
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        """Get n-gram counts"""
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def _brevity_penalty(self, ref_len: int, cand_len: int) -> float:
        """Calculate brevity penalty"""
        if cand_len > ref_len:
            return 1.0
        elif cand_len == 0:
            return 0.0
        else:
            return math.exp(1 - ref_len / cand_len)

class ROUGEScore:
    """ROUGE score calculator"""
    
    def __init__(self):
        pass
    
    def calculate_rouge_l(self, reference: str, candidate: str) -> float:
        """Calculate ROUGE-L score"""
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        # Calculate LCS
        lcs_length = self._lcs_length(ref_tokens, cand_tokens)
        
        if lcs_length == 0:
            return 0.0
        
        # Calculate precision and recall
        precision = lcs_length / len(cand_tokens)
        recall = lcs_length / len(ref_tokens)
        
        # Calculate F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]

class CoherenceAnalyzer:
    """Analyze text coherence and quality"""
    
    def __init__(self):
        pass
    
    def analyze_coherence(self, text: str) -> Dict[str, float]:
        """Analyze text coherence"""
        sentences = self._split_sentences(text)
        
        if len(sentences) < 2:
            return {"coherence_score": 1.0, "repetition_score": 1.0, "diversity_score": 0.5}
        
        # Calculate coherence metrics
        coherence_score = self._calculate_coherence(sentences)
        repetition_score = self._calculate_repetition(text)
        diversity_score = self._calculate_diversity(text)
        
        return {
            "coherence_score": coherence_score,
            "repetition_score": repetition_score,
            "diversity_score": diversity_score
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_coherence(self, sentences: List[str]) -> float:
        """Calculate coherence score based on sentence similarity"""
        if len(sentences) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(sentences) - 1):
            sim = self._sentence_similarity(sentences[i], sentences[i+1])
            similarities.append(sim)
        
        return sum(similarities) / len(similarities)
    
    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences"""
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_repetition(self, text: str) -> float:
        """Calculate repetition score (lower is better)"""
        words = text.lower().split()
        if len(words) < 2:
            return 1.0
        
        unique_words = set(words)
        repetition_ratio = len(words) / len(unique_words)
        
        # Normalize to 0-1 scale (1 is best, no repetition)
        return 1.0 / repetition_ratio
    
    def _calculate_diversity(self, text: str) -> float:
        """Calculate lexical diversity"""
        words = text.lower().split()
        if len(words) == 0:
            return 0.0
        
        unique_words = set(words)
        return len(unique_words) / len(words)

class LatencyBenchmark:
    """Benchmark model latency and throughput"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def benchmark_inference_speed(self, prompts: List[str], max_length: int = 100, num_runs: int = 5) -> Dict[str, float]:
        """Benchmark inference speed"""
        latencies = []
        token_counts = []
        
        for _ in range(num_runs):
            for prompt in prompts:
                start_time = time.time()
                
                # Tokenize input
                inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=max_length,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                end_time = time.time()
                
                # Calculate metrics
                latency = end_time - start_time
                generated_tokens = outputs.shape[1] - inputs.shape[1]
                
                latencies.append(latency)
                token_counts.append(generated_tokens)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        total_tokens = sum(token_counts)
        total_time = sum(latencies)
        throughput = total_tokens / total_time if total_time > 0 else 0
        
        return {
            "avg_latency_ms": avg_latency * 1000,
            "p95_latency_ms": p95_latency * 1000,
            "throughput_tokens_per_sec": throughput,
            "total_runs": len(latencies)
        }

class QualityEvaluator:
    """Comprehensive quality evaluation"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.perplexity_calc = PerplexityCalculator(model, tokenizer)
        self.bleu_calc = BLEUScore()
        self.rouge_calc = ROUGEScore()
        self.coherence_analyzer = CoherenceAnalyzer()
        self.latency_benchmark = LatencyBenchmark(model, tokenizer)
    
    def evaluate_generation_quality(self, prompts: List[str], references: Optional[List[str]] = None, max_length: int = 100) -> List[EvaluationResult]:
        """Evaluate generation quality"""
        results = []
        
        for i, prompt in enumerate(prompts):
            # Generate text
            generated_text = self._generate_text(prompt, max_length)
            
            # Calculate perplexity
            try:
                perplexity = self.perplexity_calc.calculate_perplexity(generated_text)
                results.append(EvaluationResult(
                    metric_name="perplexity",
                    score=perplexity,
                    details={"prompt_index": i, "generated_text": generated_text[:100]}
                ))
            except Exception as e:
                logging.warning(f"Failed to calculate perplexity: {e}")
            
            # Calculate coherence metrics
            coherence_metrics = self.coherence_analyzer.analyze_coherence(generated_text)
            for metric_name, score in coherence_metrics.items():
                results.append(EvaluationResult(
                    metric_name=metric_name,
                    score=score,
                    details={"prompt_index": i}
                ))
            
            # Calculate BLEU and ROUGE if references are provided
            if references and i < len(references):
                reference = references[i]
                
                bleu_score = self.bleu_calc.calculate_bleu(reference, generated_text)
                results.append(EvaluationResult(
                    metric_name="bleu_score",
                    score=bleu_score,
                    details={"prompt_index": i, "reference": reference[:100]}
                ))
                
                rouge_score = self.rouge_calc.calculate_rouge_l(reference, generated_text)
                results.append(EvaluationResult(
                    metric_name="rouge_l",
                    score=rouge_score,
                    details={"prompt_index": i}
                ))
        
        return results
    
    def _generate_text(self, prompt: str, max_length: int) -> str:
        """Generate text from prompt"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(next(self.model.parameters()).device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt
        generated_text = generated_text[len(prompt):].strip()
        
        return generated_text

class MambaSwarmEvaluator:
    """Main evaluator for Mamba Swarm models"""
    
    def __init__(self, swarm_engine, config: Optional[Dict[str, Any]] = None):
        self.swarm_engine = swarm_engine
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize evaluators
        self.quality_evaluator = None
        self._initialize_evaluators()
        
        # Benchmark datasets
        self.benchmark_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "The most important challenge facing humanity today is",
            "Scientific discoveries have always been driven by",
            "The relationship between humans and machines will"
        ]
    
    def _initialize_evaluators(self):
        """Initialize quality evaluators"""
        try:
            # Get model and tokenizer from swarm engine
            model = self.swarm_engine.get_model()
            tokenizer = self.swarm_engine.get_tokenizer()
            
            if model and tokenizer:
                self.quality_evaluator = QualityEvaluator(model, tokenizer)
        except Exception as e:
            self.logger.warning(f"Failed to initialize evaluators: {e}")
    
    def run_comprehensive_evaluation(self) -> BenchmarkResult:
        """Run comprehensive evaluation of the Mamba Swarm"""
        start_time = time.time()
        all_results = []
        
        # Performance benchmarks
        performance_results = self._evaluate_performance()
        all_results.extend(performance_results)
        
        # Quality benchmarks
        if self.quality_evaluator:
            quality_results = self._evaluate_quality()
            all_results.extend(quality_results)
        
        # Scalability benchmarks
        scalability_results = self._evaluate_scalability()
        all_results.extend(scalability_results)
        
        # Resource utilization
        resource_results = self._evaluate_resource_utilization()
        all_results.extend(resource_results)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(all_results)
        
        execution_time = time.time() - start_time
        
        # Get model info
        model_info = self.swarm_engine.get_model_info()
        
        return BenchmarkResult(
            benchmark_name="comprehensive_evaluation",
            overall_score=overall_score,
            individual_metrics=all_results,
            execution_time=execution_time,
            model_info=model_info
        )
    
    def _evaluate_performance(self) -> List[EvaluationResult]:
        """Evaluate performance metrics"""
        results = []
        
        try:
            # Latency benchmark
            if self.quality_evaluator:
                latency_metrics = self.quality_evaluator.latency_benchmark.benchmark_inference_speed(
                    self.benchmark_prompts[:3]  # Use subset for speed
                )
                
                for metric_name, score in latency_metrics.items():
                    results.append(EvaluationResult(
                        metric_name=f"performance_{metric_name}",
                        score=score,
                        details={"category": "performance"}
                    ))
            
            # Throughput test
            throughput = self._measure_throughput()
            results.append(EvaluationResult(
                metric_name="throughput_requests_per_sec",
                score=throughput,
                details={"category": "performance"}
            ))
            
        except Exception as e:
            self.logger.error(f"Performance evaluation failed: {e}")
        
        return results
    
    def _evaluate_quality(self) -> List[EvaluationResult]:
        """Evaluate generation quality"""
        results = []
        
        try:
            # Quality evaluation
            quality_results = self.quality_evaluator.evaluate_generation_quality(
                self.benchmark_prompts
            )
            
            # Add category to results
            for result in quality_results:
                result.details["category"] = "quality"
                results.append(result)
            
        except Exception as e:
            self.logger.error(f"Quality evaluation failed: {e}")
        
        return results
    
    def _evaluate_scalability(self) -> List[EvaluationResult]:
        """Evaluate scalability metrics"""
        results = []
        
        try:
            # Test with different loads
            load_levels = [1, 5, 10]
            
            for load in load_levels:
                start_time = time.time()
                
                # Simulate concurrent requests
                tasks = []
                for _ in range(load):
                    task = self._simulate_inference_request()
                    tasks.append(task)
                
                # Wait for completion
                success_count = sum(1 for task in tasks if task)
                total_time = time.time() - start_time
                
                # Calculate metrics
                success_rate = success_count / load
                avg_response_time = total_time / load
                
                results.append(EvaluationResult(
                    metric_name=f"scalability_success_rate_load_{load}",
                    score=success_rate,
                    details={"category": "scalability", "load_level": load}
                ))
                
                results.append(EvaluationResult(
                    metric_name=f"scalability_avg_response_time_load_{load}",
                    score=avg_response_time,
                    details={"category": "scalability", "load_level": load}
                ))
            
        except Exception as e:
            self.logger.error(f"Scalability evaluation failed: {e}")
        
        return results
    
    def _evaluate_resource_utilization(self) -> List[EvaluationResult]:
        """Evaluate resource utilization"""
        results = []
        
        try:
            # Get memory stats
            memory_stats = self.swarm_engine.memory_manager.get_memory_stats()
            
            results.append(EvaluationResult(
                metric_name="memory_utilization_gb",
                score=memory_stats.used_memory,
                details={"category": "resources", "type": "memory"}
            ))
            
            results.append(EvaluationResult(
                metric_name="gpu_memory_utilization_gb",
                score=memory_stats.gpu_memory,
                details={"category": "resources", "type": "gpu_memory"}
            ))
            
            # Encoder utilization
            active_encoders = len(self.swarm_engine.get_active_encoders())
            total_encoders = 100  # As specified in requirements
            
            results.append(EvaluationResult(
                metric_name="encoder_utilization_ratio",
                score=active_encoders / total_encoders,
                details={"category": "resources", "active": active_encoders, "total": total_encoders}
            ))
            
        except Exception as e:
            self.logger.error(f"Resource evaluation failed: {e}")
        
        return results
    
    def _measure_throughput(self) -> float:
        """Measure system throughput"""
        try:
            num_requests = 10
            start_time = time.time()
            
            for _ in range(num_requests):
                self._simulate_inference_request()
            
            total_time = time.time() - start_time
            throughput = num_requests / total_time
            
            return throughput
        except Exception as e:
            self.logger.error(f"Throughput measurement failed: {e}")
            return 0.0
    
    def _simulate_inference_request(self) -> bool:
        """Simulate an inference request"""
        try:
            prompt = "This is a test prompt for evaluation."
            result = self.swarm_engine.generate(prompt, max_length=50)
            return result is not None
        except Exception as e:
            self.logger.error(f"Simulated request failed: {e}")
            return False
    
    def _calculate_overall_score(self, results: List[EvaluationResult]) -> float:
        """Calculate overall benchmark score"""
        if not results:
            return 0.0
        
        # Weight different categories
        weights = {
            "performance": 0.3,
            "quality": 0.4,
            "scalability": 0.2,
            "resources": 0.1
        }
        
        category_scores = defaultdict(list)
        
        for result in results:
            category = result.details.get("category", "other")
            
            # Normalize scores based on metric type
            normalized_score = self._normalize_score(result)
            category_scores[category].append(normalized_score)
        
        # Calculate weighted average
        total_score = 0.0
        total_weight = 0.0
        
        for category, scores in category_scores.items():
            if category in weights and scores:
                avg_score = sum(scores) / len(scores)
                weight = weights[category]
                total_score += avg_score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _normalize_score(self, result: EvaluationResult) -> float:
        """Normalize score to 0-1 range"""
        metric_name = result.metric_name
        score = result.score
        
        # Define normalization rules for different metrics
        if "perplexity" in metric_name:
            # Lower is better, normalize to 0-1 where 1 is best
            return max(0.0, 1.0 - min(score / 100.0, 1.0))
        elif "latency" in metric_name or "response_time" in metric_name:
            # Lower is better, normalize based on reasonable thresholds
            return max(0.0, 1.0 - min(score / 1000.0, 1.0))  # 1 second threshold
        elif "throughput" in metric_name:
            # Higher is better, normalize based on expected range
            return min(score / 100.0, 1.0)  # 100 requests/sec as max
        elif "success_rate" in metric_name or "utilization" in metric_name:
            # Already in 0-1 range
            return score
        else:
            # Default: assume higher is better and clamp to 0-1
            return min(max(score, 0.0), 1.0)
    
    def export_evaluation_report(self, result: BenchmarkResult, filename: Optional[str] = None) -> str:
        """Export evaluation report to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mamba_swarm_evaluation_{timestamp}.json"
        
        # Convert to serializable format
        report = {
            "benchmark_name": result.benchmark_name,
            "overall_score": result.overall_score,
            "execution_time": result.execution_time,
            "timestamp": result.timestamp,
            "model_info": result.model_info,
            "metrics": [
                {
                    "name": metric.metric_name,
                    "score": metric.score,
                    "details": metric.details,
                    "timestamp": metric.timestamp
                }
                for metric in result.individual_metrics
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation report saved to {filename}")
        return filename

# Example usage
if __name__ == "__main__":
    # This would be used with actual SwarmEngine instance
    # evaluator = MambaSwarmEvaluator(swarm_engine)
    # result = evaluator.run_comprehensive_evaluation()
    # report_file = evaluator.export_evaluation_report(result)
    
    # Demo of individual components
    print("Mamba Swarm Evaluator components initialized successfully")
    
    # Example BLEU calculation
    bleu_calc = BLEUScore()
    reference = "The quick brown fox jumps over the lazy dog"
    candidate = "The fast brown fox leaps over the sleepy dog"
    bleu_score = bleu_calc.calculate_bleu(reference, candidate)
    print(f"BLEU score: {bleu_score:.3f}")
    
    # Example coherence analysis
    coherence_analyzer = CoherenceAnalyzer()
    text = "This is a coherent text. It flows well from sentence to sentence. The ideas are connected logically."
    coherence_metrics = coherence_analyzer.analyze_coherence(text)
    print(f"Coherence metrics: {coherence_metrics}") 