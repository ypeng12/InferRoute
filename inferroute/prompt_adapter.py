"""
Prompt Adaptation module for InferRoute (inspired by FrugalGPT).

This module implements techniques to adapt the prompt to low-cost models.
Specifically, it focuses on "Prompt Selection / Compression" to reduce the number 
of few-shot examples when routing to cheap local models (Ollama, vLLM), thereby
minimizing input token costs and latency.
"""
import re
import logging
from typing import Any

logger = logging.getLogger("inferroute.prompt_adapter")


def compress_few_shot_examples(prompt_text: str, max_examples: int = 1) -> str:
    """
    Heuristically identifies and trims few-shot examples in a prompt text.
    Keeps at most `max_examples` and appends the final query instructions.
    
    Supports formats like:
    - Example 1: ... Example 2: ...
    - Q: ... A: ... Q: ... A: ...
    - Input: ... Output: ...
    """
    # 1. Look for "Example X:" or "Example X\n" patterns
    example_blocks = re.split(r"(?i)\bexample\s*\d+\s*[:\-\n]", prompt_text)
    if len(example_blocks) > 2:
        # The first split part is usually the introduction/context
        intro = example_blocks[0]
        # Keep up to max_examples from the middle blocks
        selected_examples = []
        for block in example_blocks[1:max_examples + 1]:
            selected_examples.append(block.strip())
        
        # The last block typically contains the final question/query
        final_query = example_blocks[-1].strip()
        
        reconstructed = intro
        for idx, ex in enumerate(selected_examples):
            reconstructed += f"\n\nExample {idx+1}:\n{ex}"
        
        if final_query and final_query not in selected_examples:
            reconstructed += f"\n\n{final_query}"
        
        logger.info(f"[PromptAdapter] Compressed prompt from {len(prompt_text)} to {len(reconstructed)} characters by trimming Example blocks.")
        return reconstructed

    # 2. Look for "Q:" and "A:" patterns (standard QA few-shot)
    qa_blocks = re.split(r"(?i)\b(?:q|question|input)\s*[:\-\n]", prompt_text)
    if len(qa_blocks) > 2:
        # The first split part is instructions
        intro = qa_blocks[0]
        selected_qa = []
        # Each item in qa_blocks after the first contains the question and answer
        # except the last one which contains the final question
        for block in qa_blocks[1:max_examples + 1]:
            selected_qa.append(block.strip())
            
        # The last block contains the final prompt query
        final_query = qa_blocks[-1].strip()
        
        reconstructed = intro
        for idx, qa in enumerate(selected_qa):
            # Try to format it cleanly
            reconstructed += f"\n\nQ: {qa}"
            
        if final_query and final_query not in selected_qa:
            reconstructed += f"\n\nQ: {final_query}"
            
        logger.info(f"[PromptAdapter] Compressed prompt from {len(prompt_text)} to {len(reconstructed)} characters by trimming Q&A blocks.")
        return reconstructed

    return prompt_text


def adapt_prompt(messages: list[dict[str, str]], target_backend: str) -> list[dict[str, str]]:
    """
    Adapts the messages list for the target backend.
    For local cheap backends (ollama, vllm), it applies few-shot compression.
    For premium cloud models (openai, gemini), it keeps the full rich few-shots.
    """
    if target_backend not in ["ollama", "vllm"]:
        # Do not compress for premium models to maintain high quality
        return messages

    adapted_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "user" and len(content) > 300:
            # Compress long user messages that might contain few-shots
            compressed_content = compress_few_shot_examples(content, max_examples=1)
            adapted_messages.append({"role": role, "content": compressed_content})
        else:
            adapted_messages.append({"role": role, "content": content})
            
    return adapted_messages
