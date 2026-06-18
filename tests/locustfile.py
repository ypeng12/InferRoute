"""
Locust load test file for InferRoute gateway.

Scenarios:
  - NormalUser:  Mixed blocking + streaming at realistic rates
  - BurstUser:   High-concurrency burst to stress routing/CB
  - CostUser:    Cost-optimized requests (forces local backends)
  - ChaosUser:   Invalid keys, oversized payloads, bad JSON

Run:
  locust -f tests/locustfile.py --headless -u 50 -r 5 -t 60s --host http://localhost:8080
  locust -f tests/locustfile.py  # interactive UI at http://localhost:8089
"""
import json
import random
import time
from locust import HttpUser, task, between, events, constant_throughput

# Test credentials
VALID_KEY = "sk-inferroute-demo"
INVALID_KEY = "sk-invalid-key-chaos"

# Sample prompts of varying complexity
PROMPTS = [
    "What is machine learning?",
    "Explain the difference between REST and GraphQL APIs.",
    "Write a Python function to calculate the nth Fibonacci number recursively.",
    "What are the SOLID principles in software engineering? Give brief examples.",
    "How does Redis handle cache eviction? Describe the LRU algorithm in detail with a concrete example.",
    "Summarize the key differences between SQL and NoSQL databases for a technical audience.",
    "Explain CAP theorem and how it applies to distributed system design.",
]

SYSTEM_PROMPT = "You are a helpful AI assistant. Keep answers concise and accurate."


def _chat_body(prompt: str, model: str = "edge/auto", stream: bool = False, **routing_opts) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": stream,
        "max_output_tokens": 150,
    }
    if routing_opts:
        body["routing"] = routing_opts
    return body


class NormalUser(HttpUser):
    """
    Simulates a typical application user: mostly blocking requests,
    occasional streaming, realistic think time.
    """
    wait_time = between(1, 3)
    weight = 60  # 60% of users

    @task(7)
    def blocking_completion(self):
        prompt = random.choice(PROMPTS)
        with self.client.post(
            "/v1/chat/completions",
            json=_chat_body(prompt),
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            catch_response=True,
            name="/v1/chat/completions [blocking]",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "choices" not in data:
                    resp.failure(f"Missing choices in response: {data}")
            elif resp.status_code in (429, 503):
                resp.success()  # expected under load
            else:
                resp.failure(f"Unexpected status {resp.status_code}: {resp.text[:200]}")

    @task(2)
    def streaming_completion(self):
        prompt = random.choice(PROMPTS)
        start_time = time.time()
        ttft_ms = None
        chunk_count = 0

        with self.client.post(
            "/v1/chat/completions",
            json=_chat_body(prompt, stream=True),
            headers={
                "Authorization": f"Bearer {VALID_KEY}",
                "Accept": "text/event-stream",
            },
            stream=True,
            catch_response=True,
            name="/v1/chat/completions [streaming]",
        ) as resp:
            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if line.startswith(b"data: "):
                        payload = line[6:].strip()
                        if payload == b"[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                            chunk_count += 1
                            if ttft_ms is None:
                                choices = chunk.get("choices", [])
                                if choices and choices[0].get("delta", {}).get("content"):
                                    ttft_ms = (time.time() - start_time) * 1000.0
                        except json.JSONDecodeError:
                            pass
                resp.success()
            elif resp.status_code in (429, 503):
                resp.success()
            else:
                resp.failure(f"Stream error {resp.status_code}")

    @task(1)
    def check_health(self):
        self.client.get("/healthz", name="/healthz")

    @task(1)
    def check_routing_status(self):
        self.client.get(
            "/v1/routing/status",
            name="/v1/routing/status",
        )


class BurstUser(HttpUser):
    """
    High-frequency burst user to stress circuit breakers and fallback logic.
    """
    wait_time = between(0.1, 0.5)
    weight = 20  # 20% of users

    @task(10)
    def rapid_blocking(self):
        prompt = random.choice(PROMPTS[:3])  # short prompts for speed
        with self.client.post(
            "/v1/chat/completions",
            json=_chat_body(prompt, max_output_tokens=50),
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            catch_response=True,
            name="/v1/chat/completions [burst]",
        ) as resp:
            if resp.status_code in (200, 429, 503):
                resp.success()
            else:
                resp.failure(f"Burst error {resp.status_code}")


class CostUser(HttpUser):
    """
    Cost-optimized user: forces local-only routing to minimize API costs.
    """
    wait_time = between(2, 5)
    weight = 15  # 15% of users

    @task(5)
    def cost_optimized_request(self):
        with self.client.post(
            "/v1/chat/completions",
            json=_chat_body(
                random.choice(PROMPTS),
                policy="cost",
                allow_local=True,
                allow_cloud=False,
            ),
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            catch_response=True,
            name="/v1/chat/completions [cost-policy]",
        ) as resp:
            if resp.status_code in (200, 429, 502, 503):
                resp.success()
            else:
                resp.failure(f"Cost request error {resp.status_code}")

    @task(2)
    def cloud_only_request(self):
        with self.client.post(
            "/v1/chat/completions",
            json=_chat_body(
                random.choice(PROMPTS),
                policy="latency",
                allow_local=False,
                allow_cloud=True,
            ),
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            catch_response=True,
            name="/v1/chat/completions [cloud-only]",
        ) as resp:
            if resp.status_code in (200, 429, 502, 503):
                resp.success()
            else:
                resp.failure(f"Cloud request error {resp.status_code}")


class ChaosUser(HttpUser):
    """
    Chaos user: invalid auth keys, malformed payloads, oversized prompts.
    Verifies that the gateway correctly rejects and rate-limits bad requests.
    """
    wait_time = between(3, 8)
    weight = 5  # 5% of users

    @task(3)
    def invalid_api_key(self):
        with self.client.post(
            "/v1/chat/completions",
            json=_chat_body("hello"),
            headers={"Authorization": f"Bearer {INVALID_KEY}"},
            catch_response=True,
            name="/v1/chat/completions [invalid-auth]",
        ) as resp:
            if resp.status_code == 401:
                resp.success()  # expected
            else:
                resp.failure(f"Expected 401, got {resp.status_code}")

    @task(2)
    def malformed_json(self):
        with self.client.post(
            "/v1/chat/completions",
            data="not valid json at all {{{",
            headers={
                "Authorization": f"Bearer {VALID_KEY}",
                "Content-Type": "application/json",
            },
            catch_response=True,
            name="/v1/chat/completions [bad-json]",
        ) as resp:
            if resp.status_code in (400, 422):
                resp.success()  # expected
            else:
                resp.failure(f"Expected 400/422, got {resp.status_code}")

    @task(1)
    def oversized_prompt(self):
        huge_prompt = "Explain this concept in extreme detail. " * 500
        with self.client.post(
            "/v1/chat/completions",
            json=_chat_body(huge_prompt),
            headers={"Authorization": f"Bearer {VALID_KEY}"},
            catch_response=True,
            name="/v1/chat/completions [oversized]",
        ) as resp:
            if resp.status_code in (200, 400, 413, 422, 429, 502):
                resp.success()  # any handled response is acceptable
            else:
                resp.failure(f"Oversized request unexpected {resp.status_code}")
