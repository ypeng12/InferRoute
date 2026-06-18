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
        Currently checks JSON schemas.
        """
        # If no JSON schema is requested, we pass by default
        response_format = req.get("response_format")
        if not response_format:
            return ValidationResult(ok=True)
            
        fmt_type = response_format.get("type")
        if fmt_type != "json_schema":
            return ValidationResult(ok=True)
            
        schema_dict = response_format.get("json_schema", {}).get("schema")
        if not schema_dict:
            return ValidationResult(ok=True)
            
        # Extract response text to validate
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
