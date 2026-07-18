"""
RouteLLM-inspired Preference Router for InferRoute.

Calculates the Bradley-Terry pairwise preference probability:
P(M_strong > M_cheap | x) = sigmoid(w^T x + b)

If the probability is >= threshold theta, selects the strong cloud model;
otherwise, selects the cheap local model.
"""
import math
import logging
from typing import Any

logger = logging.getLogger("inferroute.preference_router")


class PreferenceRouter:
    """
    Preference Router simulating human-preferred win rates trained on LMSYS Arena data.
    """
    # Emulates linear weights trained on human preferences
    weights = {
        "intercept": 0.15,
        "w_length": 0.005,          # longer prompts slightly favor strong models
        "w_code_density": 0.45,     # code requests strongly favor strong models
        "w_math_density": 0.65,     # math reasoning strongly favors strong models
        "w_json_density": 0.35,     # structured extraction favors strong models
        "w_complexity_words": 0.25  # academic/complex terms favor strong models
    }

    def _extract_features(self, prompt: str) -> dict[str, float]:
        prompt_lower = prompt.lower()
        words = prompt_lower.split()
        num_words = len(words)
        
        # Heuristics
        code_words = ["def ", "class ", "function", "import ", "const ", "var ", "return ", "{", "}", "javascript", "python", "js", "cpp", "java"]
        math_words = ["solve", "calculate", "math", "+", "-", "*", "/", "sum", "equation", "prime", "fibonacci", "cookies", "triangle", "area", "percent"]
        json_words = ["json", "schema", "key-value", "format", "extract", "parse"]
        complex_words = ["concept", "explain", "describe", "compare", "contrast", "architecture", "design", "optimize", "system", "difference", "quantum", "thermodynamics"]

        code_count = sum(1 for kw in code_words if kw in prompt_lower)
        math_count = sum(1 for kw in math_words if kw in prompt_lower)
        json_count = sum(1 for kw in json_words if kw in prompt_lower)
        complex_count = sum(1 for kw in complex_words if kw in prompt_lower)

        return {
            "length": float(num_words),
            "code_density": float(code_count / (num_words + 1)),
            "math_density": float(math_count / (num_words + 1)),
            "json_density": float(json_count / (num_words + 1)),
            "complexity_words": float(complex_count / (num_words + 1))
        }

    def _sigmoid(self, val: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-val))
        except OverflowError:
            return 0.0 if val < 0 else 1.0

    def predict_preference_probability(self, prompt: str) -> float:
        """
        Calculates P(M_strong > M_cheap | prompt)
        """
        feats = self._extract_features(prompt)
        w = self.weights
        
        score = (
            w["intercept"]
            + w["w_length"] * feats["length"]
            + w["w_code_density"] * feats["code_density"]
            + w["w_math_density"] * feats["math_density"]
            + w["w_json_density"] * feats["json_density"]
            + w["w_complexity_words"] * feats["complexity_words"]
        )
        
        prob = self._sigmoid(score)
        logger.debug(
            f"[RouteLLM] Features: {feats} -> Logit Score: {score:.4f} -> P(Strong > Cheap): {prob:.4f}"
        )
        return prob

    def choose_backend(self, prompt: str, threshold: float, available_backends: list[str]) -> str:
        """
        Routes to cloud model (openai/gemini) if P(Strong > Cheap) >= threshold.
        Otherwise routes to local model (ollama/vllm).
        """
        cloud_candidates = [b for b in available_backends if b in ("openai", "gemini")]
        local_candidates = [b for b in available_backends if b in ("vllm", "ollama")]

        # Edge cases where one list is empty
        if not cloud_candidates:
            return local_candidates[0] if local_candidates else available_backends[0]
        if not local_candidates:
            return cloud_candidates[0] if cloud_candidates else available_backends[0]

        prob = self.predict_preference_probability(prompt)
        if prob >= threshold:
            # Prefer OpenAI if available, else Gemini
            selected = "openai" if "openai" in cloud_candidates else cloud_candidates[0]
            logger.info(f"[RouteLLM] P={prob:.3f} >= threshold={threshold:.3f}. Selecting Cloud: {selected}")
            return selected
        else:
            # Prefer vLLM if available, else Ollama
            selected = "vllm" if "vllm" in local_candidates else local_candidates[0]
            logger.info(f"[RouteLLM] P={prob:.3f} < threshold={threshold:.3f}. Selecting Local: {selected}")
            return selected


preference_router = PreferenceRouter()
