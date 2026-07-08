import json
import logging
from typing import Any, Optional, NamedTuple
from jsonschema import validate, ValidationError
from inferroute.observability import VALIDATION_FAIL_TOTAL

logger = logging.getLogger("inferroute.validator")

class ValidationResult(NamedTuple):
    ok: bool
    reason: Optional[str] = None

class OutputValidator:
    def validate_schema(self, content: str, schema: dict[str, Any]) -> ValidationResult:
        """
        Validates content string parses as JSON and conforms to the specified JSON schema.
        """
        try:
            parsed_json = json.loads(content)
        except json.JSONDecodeError as jde:
            reason = f"JSON decode error: {jde}"
            VALIDATION_FAIL_TOTAL.labels(reason="invalid_json").inc()
            return ValidationResult(ok=False, reason=reason)
            
        try:
            validate(instance=parsed_json, schema=schema)
            return ValidationResult(ok=True)
        except ValidationError as ve:
            reason = f"JSON Schema validation error: {ve.message}"
            VALIDATION_FAIL_TOTAL.labels(reason="schema_violation").inc()
            return ValidationResult(ok=False, reason=reason)

    def validate_response(self, req: dict[str, Any], resp: dict[str, Any]) -> ValidationResult:
        """
        Validates an LLM response based on the gateway request specifications.
        Checks JSON schemas and code compilation syntax.
        """
        choices = resp.get("choices", [])
        if not choices:
            reason = "No choices returned in response to validate."
            VALIDATION_FAIL_TOTAL.labels(reason="empty_choices").inc()
            return ValidationResult(ok=False, reason=reason)
            
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            reason = "Empty content in message choices."
            VALIDATION_FAIL_TOTAL.labels(reason="empty_content").inc()
            return ValidationResult(ok=False, reason=reason)

        # 1. Syntactic check for Code tasks
        if req.get("category") == "code" or "def " in content or "import " in content:
            import ast
            try:
                py_code = content
                if "```python" in py_code:
                    py_code = py_code.split("```python", 1)[1].split("```", 1)[0]
                elif "```" in py_code:
                    py_code = py_code.split("```", 1)[1].split("```", 1)[0]
                ast.parse(py_code.strip())
            except Exception as e:
                reason = f"Python syntax error: {e}"
                VALIDATION_FAIL_TOTAL.labels(reason="syntax_error").inc()
                return ValidationResult(ok=False, reason=reason)

        # 2. Check JSON schemas
        response_format = req.get("response_format")
        if not response_format:
            return ValidationResult(ok=True)
            
        fmt_type = response_format.get("type")
        if fmt_type != "json_schema":
            return ValidationResult(ok=True)
            
        schema_dict = response_format.get("json_schema", {}).get("schema")
        if not schema_dict:
            return ValidationResult(ok=True)
            
        return self.validate_schema(content, schema_dict)
        
        
    def validate_stream_chunk(self, req: dict[str, Any], accumulated_content: str) -> ValidationResult:
        """
        Optional validator for checking partial stream updates.
        Allows testing if the final accumulated stream complies.
        """
        response_format = req.get("response_format")
        if not response_format or response_format.get("type") != "json_schema":
            return ValidationResult(ok=True)
            
        schema_dict = response_format.get("json_schema", {}).get("schema")
        if not schema_dict:
            return ValidationResult(ok=True)
            
        return self.validate_schema(accumulated_content, schema_dict)

    def validate_speculative_quality(self, content: str) -> ValidationResult:
        """
        Evaluates the quality of a speculative generation (e.g., from small models).
        Checks for empty content, repetitive loops, and system error leakage.
        """
        if not content or len(content.strip()) < 5:
            return ValidationResult(ok=False, reason="Too short or empty content")
            
        words = content.split()
        if len(words) > 10:
            # Check for repetitive loops (common in small models)
            for i in range(len(words) - 5):
                sub = words[i:i+3]
                occurrences = 0
                for j in range(len(words) - 2):
                    if words[j:j+3] == sub:
                        occurrences += 1
                if occurrences > 3:
                    VALIDATION_FAIL_TOTAL.labels(reason="repetitive_loop").inc()
                    return ValidationResult(ok=False, reason="Repetitive generation loop detected")
                    
        # Check for system error leaks or traceback exposures
        for pattern in ("traceback (most recent call", "exception:", "internal server error", "connection error"):
            if pattern in content.lower():
                VALIDATION_FAIL_TOTAL.labels(reason="error_leak").inc()
                return ValidationResult(ok=False, reason=f"Suspected error leak: {pattern}")
                
        return ValidationResult(ok=True)
