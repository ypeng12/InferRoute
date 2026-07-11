import re

def generate_mock_reply(prompt: str, provider: str) -> str:
    """Generates context-aware, highly realistic responses for Simulation Mode."""
    prompt_clean = prompt.lower().strip()
    
    # 1. general_01
    if "quantum computing" in prompt_clean and "10-year old" in prompt_clean:
        if provider == "openai":
            return "Quantum computing is like a super-powered computer. Regular computers use bits that are like light switches (either ON or OFF). Quantum computers use qubits, which can be both ON and OFF at the same time using superposition. This lets them solve huge puzzles in seconds."
        elif provider == "gemini":
            return "A quantum computer is a special computer. It doesn't use bits; it uses qubits. Because of superposition, a qubit can be both 0 and 1, making it a very fast computer."
        elif provider == "vllm":
            return "Quantum computing is a new kind of computer technology. Instead of normal switches, it uses quantum qubits. It is a powerful type of computer."
        else: # ollama
            return "Quantum computing is computing that uses superposition. It does not use regular computer parts."

    # 2. general_02
    if "laws of thermodynamics" in prompt_clean:
        if provider == "openai":
            return "The three primary laws of thermodynamics are: 1. Conservation of energy: Energy cannot be created or destroyed. 2. Entropy: The entropy of an isolated system always increases. 3. Absolute zero temperature: Entropy approaches a constant value at absolute zero."
        elif provider == "gemini":
            return "Here are the laws of thermodynamics: 1. Energy is conserved. 2. Systems tend toward higher entropy. 3. Zero temperature means minimum entropy."
        elif provider == "vllm":
            return "Thermodynamics laws: 1. Conservation of energy. 2. System entropy increases. 3. Temperature cannot drop below absolute zero."
        else: # ollama
            return "Thermodynamics has laws. Energy is conserved. Also entropy increases. Temperature is related."

    # 3. general_03
    if "weather and climate" in prompt_clean:
        if provider == "openai":
            return "Weather refers to short-term atmospheric conditions like rain or temperature today. Climate is the long-term average of those weather patterns over time, typically measured over 30 years."
        elif provider == "gemini":
            return "Weather describes the atmosphere today. Climate describes long-term trends over a long time."
        elif provider == "vllm":
            return "Weather changes day to day in the atmosphere. Climate is a long-term weather average."
        else: # ollama
            return "Weather is what happens today. Climate is general."

    # 4. code_01
    if "def is_prime" in prompt_clean:
        if provider in ("openai", "gemini", "vllm"):
            return "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
        else: # ollama
            # returns code with syntax error (missing colons)
            return "def is_prime(n)\n    if n <= 1 return False\n    for i in range(2, n)\n        if n % i == 0 return False\n    return True"

    # 5. code_02
    if "reversestring" in prompt_clean:
        if provider in ("openai", "gemini", "vllm"):
            return "function reverseString(str) {\n    return str.split('').reverse().join('');\n}"
        else: # ollama
            return "function reverseString(str) {\n    return str.reverse();\n}"

    # 6. math_01
    if "5x - 15 = 20" in prompt_clean:
        if provider in ("openai", "gemini"):
            return "7"
        else: # vllm, ollama
            return "5"

    # 7. math_02
    if "8 cookies requires 2 cups" in prompt_clean:
        if provider in ("openai", "gemini"):
            return "6"
        else: # vllm, ollama
            return "12"

    # 8. math_03
    if "base of 10cm and height of 5cm" in prompt_clean:
        if provider in ("openai", "gemini"):
            return "25"
        else: # vllm, ollama
            return "50"

    # 9. extraction_01
    if "john doe is a 35-year-old" in prompt_clean:
        if provider in ("openai", "gemini"):
            return '{"name": "John Doe", "age": 35, "city": "Chicago"}'
        elif provider == "vllm":
            return '{"name": "John Doe", "city": "Chicago"}'  # missing 'age' key
        else: # ollama
            return "John Doe is a 35 year old doctor from Chicago."  # invalid JSON

    # 10. extraction_02
    if "ord-998822" in prompt_clean:
        if provider in ("openai", "gemini"):
            return '{"id": "ORD-998822", "total": 124.99}'
        elif provider == "vllm":
            return '{"id": "ORD-998822"}'  # missing 'total' key
        else: # ollama
            return "The ID is ORD-998822 and total is 124.99."  # invalid JSON

    # 11. long_context_01
    if "what hashing algorithm and database are used by inferroute" in prompt_clean:
        if provider == "openai":
            return "InferRoute uses the SHA-256 hashing algorithm and Redis database along with a Radix Trie to track prompt prefixes for KV cache affinity."
        elif provider == "gemini":
            return "It uses SHA-256 hashes and Redis with a Radix Trie for routing prefix cache affinity."
        elif provider == "vllm":
            return "InferRoute uses a Radix Trie and Redis database." # missing SHA-256
        else: # ollama
            return "InferRoute uses Radix Trie for caching." # missing SHA-256 and Redis

    # 12. long_context_02
    if "optimized by kv caching to reduce" in prompt_clean:
        if provider == "openai":
            return "KV caching optimizes the prompt pre-fill phase of LLM inference to reduce 首字延迟 (TTFT)."
        elif provider == "gemini":
            return "The pre-fill phase is optimized by KV caching to reduce TTFT."
        elif provider == "vllm":
            return "It optimizes the prefill phase." # missing TTFT and KV cache
        else: # ollama
            return "It optimizes the response latency." # missing prefill, TTFT, and KV cache

    # Existing test/demo overrides
    # Repetitive loop trigger
    if "loop" in prompt_clean or "repeat" in prompt_clean or "spam" in prompt_clean:
        return "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello"

    if re.search(r"\b(hello|hi|hey|nihao|你好|哈喽)\b", prompt_clean):
        return (
            f"Hello there! I am a simulated {provider.upper()} model running in InferRoute's offline sandbox. "
            f"How can I assist you today? Feel free to ask me questions, test code examples, or type 'loop' to trigger a validation failure!"
        )
        
    return (
        f"You queried: '{prompt[:37]}...'. This is a simulated response from {provider.upper()}."
    )
