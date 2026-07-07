"""LLM client abstraction for session plan generation.

Supports Anthropic Claude and OpenAI GPT. Provider is selected automatically
based on which API key is set, or explicitly via ATOM_LLM_PROVIDER.

Environment variables:
  ATOM_LLM_PROVIDER  — "anthropic" or "openai" (auto-detected if not set)
  ANTHROPIC_API_KEY  — required for Anthropic/Claude
  OPENAI_API_KEY     — required for OpenAI/GPT
  ATOM_LLM_MODEL     — override model name
                       default Anthropic: claude-haiku-4-5-20251001
                       default OpenAI:    gpt-4o-mini
"""

from __future__ import annotations

import json
import os


DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class LLMClient:
    """Async LLM client supporting Anthropic and OpenAI."""

    def __init__(self, model: str | None = None):
        provider = os.getenv("ATOM_LLM_PROVIDER", "").lower()

        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        # Auto-detect provider if not explicitly set
        if not provider:
            if openai_key and not anthropic_key:
                provider = "openai"
            elif anthropic_key:
                provider = "anthropic"
            else:
                raise RuntimeError(
                    "No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
                )

        self.provider = provider

        if provider == "openai":
            if not openai_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Set it or switch to Anthropic."
                )
            import openai
            self.model = model or os.getenv("ATOM_LLM_MODEL", DEFAULT_OPENAI_MODEL)
            self._client = openai.AsyncOpenAI(api_key=openai_key)

        elif provider == "anthropic":
            if not anthropic_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not set. Set it or switch to OpenAI."
                )
            import anthropic
            self.model = model or os.getenv("ATOM_LLM_MODEL", DEFAULT_ANTHROPIC_MODEL)
            self._client = anthropic.AsyncAnthropic(api_key=anthropic_key)

        else:
            raise RuntimeError(
                f"Unknown provider '{provider}'. Use 'anthropic' or 'openai'."
            )

    async def generate_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 4000,
        temperature: float = 0.8,
    ) -> dict:
        """Generate a structured JSON response. Returns parsed dict."""
        if self.provider == "openai":
            return await self._generate_openai(system, user, max_tokens, temperature)
        return await self._generate_anthropic(system, user, max_tokens, temperature)

    async def _generate_anthropic(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> dict:
        message = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self._log_usage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
        text = message.content[0].text.strip()
        return self._parse_json(text)

    async def _generate_openai(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> dict:
        response = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        self._log_usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        text = response.choices[0].message.content.strip()
        return self._parse_json(text)

    def _log_usage(self, input_tokens: int, output_tokens: int) -> None:
        # Prices per 1M tokens (USD)
        PRICES: dict[str, tuple[float, float]] = {
            "claude-haiku-4-5-20251001": (0.80, 4.00),
            "claude-haiku-4-5":          (0.80, 4.00),
            "claude-sonnet-4-5":         (3.00, 15.00),
            "claude-opus-4-5":           (15.00, 75.00),
            "gpt-4o-mini":               (0.15, 0.60),
            "gpt-4o":                    (5.00, 15.00),
        }
        price_in, price_out = PRICES.get(self.model, (0.0, 0.0))
        cost_usd = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
        cost_krw = cost_usd * 1_380  # approximate KRW rate
        print(
            f"[LLM] {self.model} | "
            f"in={input_tokens} out={output_tokens} tokens | "
            f"${cost_usd:.4f} (₩{cost_krw:.1f})"
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        # Strip markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\n\nResponse:\n{text}")
