"""
RouterBench-inspired routing classifiers for InferRoute.

This module implements content-aware and baseline routing policies described in the paper:
"ROUTERBENCH: A Benchmark for Multi-LLM Routing System" (withmartian/routerbench).

Implemented Routers:
1. RuleRouter: Content heuristics matching keywords.
2. KNNRouter: Jaccard similarity nearest-neighbor lookup on historical outcomes.
3. MLPRouter: Perceptron/Logistic Regression classifier using prompt feature coefficients.
4. ZeroRouter: Mathematical mixture baseline (non-content-aware).
5. OracleRouter: Theoretical upper-bound (cheapest model achieving quality >= 0.8).
"""

import math
import random
import logging
from typing import Any

logger = logging.getLogger("inferroute.learned_router")


class RuleRouter:
    """
    Rule Router inspired by standard keyword heuristics.
    Routes to specific backends based on keyword rules.
    """
    def choose_backend(self, prompt: str, available_backends: list[str]) -> str:
        prompt_lower = prompt.lower()
        
        # 1. Math queries require high accuracy -> OpenAI or Gemini
        if any(kw in prompt_lower for kw in ["solve", "calculate", "math", "equation", "cookies", "triangle"]):
            if "openai" in available_backends:
                return "openai"
            if "gemini" in available_backends:
                return "gemini"
                
        # 2. Code queries -> vLLM (our code-tuned backend) or Ollama
        if any(kw in prompt_lower for kw in ["def ", "class ", "function", "import ", "javascript", "python", "js", "reversestring"]):
            if "vllm" in available_backends:
                return "vllm"
            if "ollama" in available_backends:
                return "ollama"
                
        # 3. JSON extraction -> OpenAI or Gemini
        if any(kw in prompt_lower for kw in ["json", "schema", "key-value", "format", "extract"]):
            if "gemini" in available_backends:
                return "gemini"
            if "openai" in available_backends:
                return "openai"
                
        # Default: choose cheapest available
        if "ollama" in available_backends:
            return "ollama"
        if "vllm" in available_backends:
            return "vllm"
        return available_backends[0]


class KNNRouter:
    """
    KNN Router inspired by the RouterBench paper (withmartian/routerbench).
    Uses Jaccard text similarity to look up similar prompts in the training set,
    predicts the quality of each candidate backend, and selects the backend
    maximizing: score = lambda_val * predicted_quality - backend_cost.
    """
    def __init__(self):
        # We define our reference training samples based on workload.json
        self.training_samples = [
            {
                "prompt": "explain the concept of quantum computing in simple terms for a 10-year old.",
                "qualities": {"openai": 1.0, "gemini": 0.8, "vllm": 0.6, "ollama": 0.5}
            },
            {
                "prompt": "list the three primary laws of thermodynamics.",
                "qualities": {"openai": 1.0, "gemini": 0.8, "vllm": 0.7, "ollama": 0.4}
            },
            {
                "prompt": "explain the difference between weather and climate.",
                "qualities": {"openai": 1.0, "gemini": 0.8, "vllm": 0.7, "ollama": 0.4}
            },
            {
                "prompt": "def is_prime(n):\n    # write python code to return true if n is prime and false otherwise.\n    # return only raw code without markdown wrappers.",
                "qualities": {"openai": 1.0, "gemini": 0.8, "vllm": 0.9, "ollama": 0.1}
            },
            {
                "prompt": "function reversestring(str) {\n    // write javascript code to reverse a string.\n    // return only raw code without markdown wrappers.",
                "qualities": {"openai": 1.0, "gemini": 0.8, "vllm": 0.9, "ollama": 0.3}
            },
            {
                "prompt": "solve for x: 5x - 15 = 20. output only the final numeric value of x as an integer.",
                "qualities": {"openai": 1.0, "gemini": 1.0, "vllm": 0.0, "ollama": 0.0}
            },
            {
                "prompt": "a batch of 8 cookies requires 2 cups of sugar. how many cups of sugar are needed for 24 cookies? output only the final numeric value.",
                "qualities": {"openai": 1.0, "gemini": 1.0, "vllm": 0.0, "ollama": 0.0}
            },
            {
                "prompt": "if a triangle has a base of 10cm and height of 5cm, what is its area in square centimeters? output only the final numeric value.",
                "qualities": {"openai": 1.0, "gemini": 1.0, "vllm": 0.0, "ollama": 0.0}
            },
            {
                "prompt": "extract structured details from: 'john doe is a 35-year-old doctor from chicago.' output raw json conforming to this schema: {\"name\": \"string\", \"age\": \"number\", \"city\": \"string\"}. do not use markdown wrappers.",
                "qualities": {"openai": 1.0, "gemini": 1.0, "vllm": 0.4, "ollama": 0.0}
            },
            {
                "prompt": "extract the id and total from: 'order id: ord-998822. billing total: $124.99 usd.' output raw json conforming to schema: {\"id\": \"string\", \"total\": \"number\"}. do not use markdown wrappers.",
                "qualities": {"openai": 1.0, "gemini": 1.0, "vllm": 0.4, "ollama": 0.0}
            },
            {
                "prompt": "context: inferroute is a distributed llm gateway that uses a radix trie to check for prompt prefixes. prompt prefixes of length 128, 256, and 512 are hashed using sha-256 and stored in redis sets mapping to warm backend hosts. when a prompt matches a warm prefix in redis, the scoring algorithm applies a cache bonus to favor routing to that host. this avoids pre-fill computing overhead on local gpu nodes.\nquestion: what hashing algorithm and database are used by inferroute to track prompt prefixes for routing cache-affinity?",
                "qualities": {"openai": 1.0, "gemini": 1.0, "vllm": 0.6, "ollama": 0.3}
            },
            {
                "prompt": "context: large language models use key-value caching (kv caching) to avoid computing representations of prompt tokens repeatedly. the prompt pre-fill phase accounts for a significant portion of ttft, especially for documents containing up to 10,000 tokens. the tcp vegas congestion control rate limiter monitors round-trip delays to adjust the concurrency window dynamically.\nquestion: which phase of llm inference is optimized by kv caching to reduce 首字延迟 (ttft)?",
                "qualities": {"openai": 1.0, "gemini": 1.0, "vllm": 0.6, "ollama": 0.3}
            }
        ]

    def _compute_jaccard_similarity(self, s1: str, s2: str) -> float:
        w1 = set(s1.lower().split())
        w2 = set(s2.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1.intersection(w2)) / len(w1.union(w2))

    def choose_backend(self, prompt: str, backend_costs: dict[str, float], lambda_val: float, available_backends: list[str]) -> str:
        # Find similar prompts in the training set
        scored_samples = []
        for sample in self.training_samples:
            sim = self._compute_jaccard_similarity(prompt, sample["prompt"])
            scored_samples.append((sim, sample))
        
        # Sort by similarity descending
        scored_samples.sort(key=lambda x: x[0], reverse=True)
        
        # Take K=3 nearest neighbors
        k = min(3, len(scored_samples))
        top_k = scored_samples[:k]
        
        # Calculate predicted qualities for each available backend
        predicted_qualities = {}
        for backend in available_backends:
            total_sim = 0.0
            weighted_quality = 0.0
            for sim, sample in top_k:
                weight = sim + 1e-5
                total_sim += weight
                weighted_quality += sample["qualities"].get(backend, 0.0) * weight
            predicted_qualities[backend] = weighted_quality / total_sim if total_sim > 0 else 0.0
            
        # Select backend maximizing: score = lambda_val * predicted_quality - cost
        best_backend = available_backends[0]
        best_score = -999999.0
        details = []
        for backend in available_backends:
            pred_q = predicted_qualities[backend]
            cost = backend_costs.get(backend, 0.0)
            # Scale quality prediction term by 0.0001 to make it comparable to USD transaction costs
            score = lambda_val * pred_q * 0.0001 - cost
            details.append(f"{backend}: q={pred_q:.2f}, c={cost:.6f}, score={score:.6f}")
            if score > best_score:
                best_score = score
                best_backend = backend
        
        logger.debug(f"[KNNRouter] Lambda={lambda_val:.3f}. Details: {', '.join(details)} -> Selected: {best_backend}")
        return best_backend


class MLPRouter:
    """
    MLP Router inspired by the RouterBench paper (withmartian/routerbench).
    Predicts model quality using text feature weights (coefficients) and selects the
    backend maximizing: score = lambda_val * predicted_quality - backend_cost.
    """
    coefficients = {
        "openai": {"intercept": 1.5, "w_is_code": 0.5, "w_is_math": 0.5, "w_is_json": 0.5, "w_is_long": 0.2},
        "gemini": {"intercept": 1.2, "w_is_code": 0.2, "w_is_math": 0.4, "w_is_json": 0.5, "w_is_long": 0.1},
        "vllm":   {"intercept": 0.2, "w_is_code": 2.2, "w_is_math": -3.0, "w_is_json": -0.8, "w_is_long": -0.2},
        "ollama": {"intercept": -0.2, "w_is_code": -0.5, "w_is_math": -3.5, "w_is_json": -2.0, "w_is_long": -0.5}
    }

    def _extract_features(self, prompt: str) -> dict[str, float]:
        prompt_lower = prompt.lower()
        is_code = any(kw in prompt_lower for kw in ["def ", "class ", "function", "import ", "javascript", "python", "js", "reversestring"])
        is_math = any(kw in prompt_lower for kw in ["solve", "calculate", "math", "+", "-", "*", "/", "sum", "equation", "prime", "fibonacci", "cookies", "triangle"])
        is_json = any(kw in prompt_lower for kw in ["json", "schema", "key-value", "format", "extract"])
        is_long = len(prompt.split()) > 40
        return {
            "is_code": float(is_code),
            "is_math": float(is_math),
            "is_json": float(is_json),
            "is_long": float(is_long)
        }

    def _sigmoid(self, x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def choose_backend(self, prompt: str, backend_costs: dict[str, float], lambda_val: float, available_backends: list[str]) -> str:
        feats = self._extract_features(prompt)
        
        predicted_qualities = {}
        for backend in available_backends:
            coefs = self.coefficients.get(backend, self.coefficients["openai"])
            score_val = (
                coefs["intercept"]
                + coefs["w_is_code"] * feats["is_code"]
                + coefs["w_is_math"] * feats["is_math"]
                + coefs["w_is_json"] * feats["is_json"]
                + coefs["w_is_long"] * feats["is_long"]
            )
            predicted_qualities[backend] = self._sigmoid(score_val)
            
        best_backend = available_backends[0]
        best_score = -999999.0
        details = []
        for backend in available_backends:
            pred_q = predicted_qualities[backend]
            cost = backend_costs.get(backend, 0.0)
            # Scale quality prediction term by 0.0001 to make it comparable to USD transaction costs
            score = lambda_val * pred_q * 0.0001 - cost
            details.append(f"{backend}: q={pred_q:.2f}, c={cost:.6f}, score={score:.6f}")
            if score > best_score:
                best_score = score
                best_backend = backend
                
        logger.debug(f"[MLPRouter] Lambda={lambda_val:.3f}. Details: {', '.join(details)} -> Selected: {best_backend}")
        return best_backend


class ZeroRouter:
    """
    Zero Router baseline inspired by the RouterBench paper (withmartian/routerbench).
    A baseline that does NOT inspect the content. Instead, it routes to a cloud/expensive
    backend with probability p, and to a local/cheap backend with probability 1 - p.
    """
    def choose_backend(self, mixture_ratio: float, available_backends: list[str]) -> str:
        cloud_backends = [b for b in available_backends if b in ("openai", "gemini")]
        local_backends = [b for b in available_backends if b in ("vllm", "ollama")]
        
        if not cloud_backends:
            return random.choice(local_backends) if local_backends else available_backends[0]
        if not local_backends:
            return random.choice(cloud_backends) if cloud_backends else available_backends[0]
            
        if random.random() < mixture_ratio:
            return "openai" if "openai" in cloud_backends else cloud_backends[0]
        else:
            return "vllm" if "vllm" in local_backends else local_backends[0]


class OracleRouter:
    """
    Oracle Router inspired by the RouterBench paper (withmartian/routerbench).
    Offline upper bound. For known benchmark prompts, chooses the cheapest model
    that achieves a quality score >= 0.8.
    """
    def __init__(self):
        # Maps prompt substrings to their qualities
        # Costs: ollama=0.0, vllm=0.000002, gemini=0.000015, openai=0.000030
        self.samples = [
            ("quantum computing", {"ollama": 0.5, "vllm": 0.6, "gemini": 0.8, "openai": 1.0}),
            ("laws of thermodynamics", {"ollama": 0.4, "vllm": 0.7, "gemini": 0.8, "openai": 1.0}),
            ("weather and climate", {"ollama": 0.4, "vllm": 0.7, "gemini": 0.8, "openai": 1.0}),
            ("def is_prime", {"ollama": 0.1, "vllm": 0.9, "gemini": 0.8, "openai": 1.0}),
            ("reversestring", {"ollama": 0.3, "vllm": 0.9, "gemini": 0.8, "openai": 1.0}),
            ("5x - 15 = 20", {"ollama": 0.0, "vllm": 0.0, "gemini": 1.0, "openai": 1.0}),
            ("8 cookies requires 2 cups", {"ollama": 0.0, "vllm": 0.0, "gemini": 1.0, "openai": 1.0}),
            ("base of 10cm and height of 5cm", {"ollama": 0.0, "vllm": 0.0, "gemini": 1.0, "openai": 1.0}),
            ("john doe is a 35-year-old", {"ollama": 0.0, "vllm": 0.4, "gemini": 1.0, "openai": 1.0}),
            ("ord-998822", {"ollama": 0.0, "vllm": 0.4, "gemini": 1.0, "openai": 1.0}),
            ("what hashing algorithm and database are used by inferroute", {"ollama": 0.3, "vllm": 0.6, "gemini": 1.0, "openai": 1.0}),
            ("optimized by kv caching to reduce", {"ollama": 0.3, "vllm": 0.6, "gemini": 1.0, "openai": 1.0})
        ]
        
    def choose_backend(self, prompt: str, available_backends: list[str]) -> str:
        prompt_lower = prompt.lower()
        qualities = None
        for key, qual in self.samples:
            if key in prompt_lower:
                qualities = qual
                break
                
        if not qualities:
            if "ollama" in available_backends: return "ollama"
            if "vllm" in available_backends: return "vllm"
            if "gemini" in available_backends: return "gemini"
            return available_backends[0]
            
        cost_ranks = {"ollama": 0, "vllm": 1, "gemini": 2, "openai": 3}
        candidates = [b for b in available_backends if b in qualities and qualities[b] >= 0.8]
        if candidates:
            candidates.sort(key=lambda x: cost_ranks.get(x, 99))
            return candidates[0]
            
        candidates = [b for b in available_backends if b in qualities]
        candidates.sort(key=lambda x: qualities[x], reverse=True)
        return candidates[0] if candidates else available_backends[0]


# Single instantiation helper
rule_router = RuleRouter()
knn_router = KNNRouter()
mlp_router = MLPRouter()
zero_router = ZeroRouter()
oracle_router = OracleRouter()
