"""
Thin wrapper around Groq so every call in the app goes through one place.
Makes it trivial to swap models later or add cost/latency logging.
"""
import time
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def call_llm(prompt: str, system: str = "", temperature: float = 0.2) -> dict:
    """
    Returns a dict with the text response plus basic telemetry
    (latency, token counts) so the eval harness can use it directly.
    """
    if _client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    start = time.time()
    response = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
    )
    latency = time.time() - start

    usage = response.usage
    return {
        "text": response.choices[0].message.content,
        "latency_seconds": round(latency, 3),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
    }
