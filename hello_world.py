"""
Week 1, Day 1 — First Contact.

This is the shape every API call in this repo will follow:
  1. Load secrets from .env (never hardcode)
  2. Handle the four failure modes (auth, rate limit, network, bad model name)
  3. Print AND save the response (so re-runs are auditable)

Run: python hello_world.py
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic, APIError as AnthropicError
from openai import OpenAI, APIError as OpenAIError

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()

PROMPT = "In one sentence: what is the single hardest thing about building production LLM apps?"

OUTPUT_DIR = Path("outputs/day_01")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def must_have(key: str) -> str:
    """Fail loudly if a required env var is missing. Never silently default."""
    value = os.getenv(key)
    if not value:
        sys.exit(f"Missing {key} in .env -- set it and try again.")
    return value


def save_response(provider: str, model: str, prompt: str, response: str) -> Path:
    """Persist every response. Reproducibility is the foundation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"{provider}_{timestamp}.json"
    payload = {
        "provider": provider,
        "model": model,
        "timestamp": timestamp,
        "prompt": prompt,
        "response": response,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


# ── Claude ────────────────────────────────────────────────────────────────────
def call_claude(prompt: str) -> str:
    client = Anthropic(api_key=must_have("ANTHROPIC_API_KEY"))
    try:
        msg = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except AnthropicError as e:
        return f"[Anthropic error: {e}]"


# ── OpenAI ────────────────────────────────────────────────────────────────────
def call_openai(prompt: str) -> str:
    client = OpenAI(api_key=must_have("OPENAI_API_KEY"))
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    except OpenAIError as e:
        return f"[OpenAI error: {e}]"


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"\nPrompt: {PROMPT}\n")
    print("-" * 60)

    claude_reply = call_claude(PROMPT)
    print(f"\nClaude (claude-opus-4-7):\n{claude_reply}")
    claude_path = save_response("anthropic", "claude-opus-4-7", PROMPT, claude_reply)
    print(f"   -> saved: {claude_path}")

    print("-" * 60)

    openai_reply = call_openai(PROMPT)
    print(f"\nOpenAI (gpt-4o-mini):\n{openai_reply}")
    openai_path = save_response("openai", "gpt-4o-mini", PROMPT, openai_reply)
    print(f"   -> saved: {openai_path}")

    print("\nFirst contact made. Screenshot this terminal when you re-run it with real keys.\n")


if __name__ == "__main__":
    main()