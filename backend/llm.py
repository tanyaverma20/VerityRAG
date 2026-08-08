"""
Thin wrapper around Groq so every call in the app goes through one place.
Makes it trivial to swap models later or add cost/latency logging.
"""
import time
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from query_transform import with_model_fallback

_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def call_llm(prompt: str, system: str = "", temperature: float = 0.2) -> dict:
    """
    Returns a dict with the text response plus basic telemetry
    (latency, token counts) so the eval harness can use it directly.

    Tries GROQ_MODEL first; on a temporary failure only, retries once against
    GROQ_FALLBACK_MODEL (see query_transform.with_model_fallback).
    """
    if _client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    def _attempt(model: str):
        return _client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

    start = time.time()
    response = with_model_fallback(_attempt)
    latency = time.time() - start

    usage = response.usage
    return {
        "text": response.choices[0].message.content,
        "latency_seconds": round(latency, 3),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
    }
