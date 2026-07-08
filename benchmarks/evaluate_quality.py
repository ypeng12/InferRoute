import json
import re
import ast
from typing import List, Optional

def evaluate_response_quality(
    category: str,
    output: str,
    requires_json: bool = False,
    expected_keys: Optional[List[str]] = None,
    reference_keywords: Optional[List[str]] = None
) -> float:
    """
    Evaluates the quality of an LLM response string on a scale from 0.0 to 1.0.
    Uses structural, syntactic, and semantic-keyword indicators.
    """
    if not output or not isinstance(output, str):
        return 0.0
        
    output_clean = output.strip()
    if len(output_clean) < 2:
        return 0.0
        
    # Check for excessive repetition (indicates loop failure / garbage)
    words = output_clean.lower().split()
    if len(words) > 10:
        # Check if a single word constitutes more than 40% of the output
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        max_freq = max(word_counts.values())
        if max_freq / len(words) > 0.40:
            return 0.05  # Loop penalty

    # 1. JSON Extraction Task Evaluation
    if requires_json:
        # Attempt to extract JSON if wrapped in markdown blocks
        json_str = output_clean
        if "```json" in json_str:
            match = re.search(r"```json\s*(.*?)\s*```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1)
        elif "```" in json_str:
            match = re.search(r"```\s*(.*?)\s*```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1)
                
        # Clean up outer brackets
        json_str = json_str.strip()
        
        try:
            parsed = json.loads(json_str)
            if not expected_keys:
                return 1.0
            
            # Check presence of expected keys
            found_keys = 0
            for key in expected_keys:
                if key in parsed:
                    # Check that value is not empty or null
                    val = parsed[key]
                    if val is not None and str(val).strip() != "":
                        found_keys += 1
            
            key_score = found_keys / len(expected_keys)
            return max(0.1, key_score)
        except Exception:
            return 0.0  # Invalid JSON format

    # 2. Code Generation Task Evaluation
    if category == "code":
        # Check for Python syntax compile
        if "def " in output_clean or "import " in output_clean:
            try:
                # Strip markdown blocks to compile raw python code
                py_code = output_clean
                if "```python" in py_code:
                    py_code = re.search(r"```python\s*(.*?)\s*```", py_code, re.DOTALL).group(1)
                elif "```" in py_code:
                    py_code = re.search(r"```\s*(.*?)\s*```", py_code, re.DOTALL).group(1)
                
                ast.parse(py_code.strip())
                syntax_score = 0.5
            except Exception:
                syntax_score = 0.1  # Syntax error
        else:
            # Non-python code syntax (e.g. JS), give base score
            syntax_score = 0.3
            
        # Keyword checks
        keyword_score = 0.0
        if reference_keywords:
            matched = sum(1 for kw in reference_keywords if kw.lower() in output_clean.lower())
            keyword_score = (matched / len(reference_keywords)) * 0.5
            
        return syntax_score + keyword_score

    # 3. Math and General QA Evaluation
    if category == "math":
        # Extract last numbers or single numbers from response to match numeric answer
        numbers = re.findall(r"\d+", output_clean)
        if reference_keywords and numbers:
            expected_num = reference_keywords[0]
            if expected_num in numbers:
                # Direct match gets 1.0, otherwise 0.0
                return 1.0
        return 0.0

    # General / Long Context Keyword Evaluation
    if reference_keywords:
        matched = sum(1 for kw in reference_keywords if kw.lower() in output_clean.lower())
        return matched / len(reference_keywords)
        
    return 0.8  # Default positive score for general coherent answers
