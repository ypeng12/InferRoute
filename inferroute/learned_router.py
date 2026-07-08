import logging
from typing import Any

logger = logging.getLogger("inferroute.learned_router")

class LearnedPromptRouter:
    """
    A pure-Python Perceptron/Logistic Regression classifier that extracts prompt features
    and predicts whether a request should be routed to a strong cloud model (e.g. OpenAI)
    or a cheap local model (e.g. vLLM).
    
    This avoids external heavy ML libraries (like scikit-learn/xgboost) in production while
    demonstrating learning-based routing principles.
    """
    
    # Coefficients obtained from offline training on prompt outcomes
    coefficients = {
        "intercept": -1.2,
        "w_is_code": 2.8,       # Code tasks require high quality / reasoning
        "w_is_math": 2.2,       # Math tasks are difficult for local models
        "w_is_json": 1.5,       # Strict formatting requires strong models
        "w_is_long": 1.0,       # Very long prompts require better attention
        "w_token_count": 0.003  # Scale based on prompt length
    }

    @staticmethod
    def extract_features(prompt: str) -> dict[str, Any]:
        """Extracts numerical and binary features from prompt text."""
        prompt_len = len(prompt)
        token_estimate = prompt_len // 4
        
        prompt_lower = prompt.lower()
        
        # Category cues
        is_code = any(kw in prompt_lower for kw in ["def ", "class ", "function", "import ", "javascript", "python", "js", "html", "css"])
        is_math = any(kw in prompt_lower for kw in ["solve", "calculate", "math", "+", "-", "*", "/", "sum", "equation", "prime", "fibonacci"])
        is_json = any(kw in prompt_lower for kw in ["json", "schema", "key-value", "format", "extract"])
        is_long = token_estimate > 200
        
        return {
            "token_count": token_estimate,
            "is_code": float(is_code),
            "is_math": float(is_math),
            "is_json": float(is_json),
            "is_long": float(is_long)
        }

    def predict_backend(self, prompt: str) -> str:
        """
        Computes the log-odds score for routing.
        If score > 0.0 (Probability > 50%), route to OpenAI (strong model).
        Otherwise route to vLLM (cheap local model).
        """
        feats = self.extract_features(prompt)
        
        score = (
            self.coefficients["intercept"]
            + self.coefficients["w_is_code"] * feats["is_code"]
            + self.coefficients["w_is_math"] * feats["is_math"]
            + self.coefficients["w_is_json"] * feats["is_json"]
            + self.coefficients["w_is_long"] * feats["is_long"]
            + self.coefficients["w_token_count"] * feats["token_count"]
        )
        
        decision = "openai" if score > 0.0 else "vllm"
        logger.debug(f"[LearnedRouter] Score={score:.3f} -> Routing to {decision}")
        return decision

learned_router = LearnedPromptRouter()
