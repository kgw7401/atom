"""
LLM Client Abstraction.

Provides async interface to LLM APIs (Claude, GPT-4) with retry logic
and structured output parsing.
"""

import json
import os
from typing import Any, Dict, List, Optional

import aiohttp


class LLMClient:
    """Async LLM client supporting Claude and GPT-4.

    Args:
        provider: 'anthropic' or 'openai'
        model: Model name (e.g., 'claude-3-haiku-20240307', 'gpt-4o-mini')
        api_key: API key (if None, reads from env)
        max_retries: Number of retries on failure
    """

    def __init__(
        self,
        provider: str = 'anthropic',
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 2,
    ):
        self.provider = provider.lower()
        self.max_retries = max_retries

        if self.provider == 'anthropic':
            self.model = model or 'claude-3-haiku-20240307'
            self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            self.api_url = 'https://api.anthropic.com/v1/messages'
        elif self.provider == 'openai':
            self.model = model or 'gpt-4o-mini'
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.api_url = 'https://api.openai.com/v1/chat/completions'
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        if not self.api_key:
            raise ValueError(f"API key required for {provider}")

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """Generate completion from prompt.

        Args:
            prompt: User prompt
            system: Optional system prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        for attempt in range(self.max_retries + 1):
            try:
                if self.provider == 'anthropic':
                    return await self._complete_anthropic(
                        prompt, system, max_tokens, temperature
                    )
                else:  # openai
                    return await self._complete_openai(
                        prompt, system, max_tokens, temperature
                    )
            except Exception as e:
                if attempt == self.max_retries:
                    raise
                print(f"LLM API error (attempt {attempt + 1}): {e}")

    async def _complete_anthropic(
        self, prompt: str, system: Optional[str], max_tokens: int, temperature: float
    ) -> str:
        """Claude API completion."""
        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        }

        messages = [{'role': 'user', 'content': prompt}]

        data = {
            'model': self.model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'messages': messages,
        }

        if system:
            data['system'] = system

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url, headers=headers, json=data
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result['content'][0]['text']

    async def _complete_openai(
        self, prompt: str, system: Optional[str], max_tokens: int, temperature: float
    ) -> str:
        """OpenAI API completion."""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})

        data = {
            'model': self.model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url, headers=headers, json=data
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result['choices'][0]['message']['content']

    async def complete_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate JSON-structured completion.

        Args:
            prompt: User prompt
            system: Optional system prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature

        Returns:
            Parsed JSON dict

        Raises:
            ValueError: If response is not valid JSON
        """
        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nRespond with valid JSON only. No markdown, no explanation."

        response = await self.complete(json_prompt, system, max_tokens, temperature)

        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            text = response.strip()
            if text.startswith('```'):
                # Extract content between ```json and ```
                lines = text.split('\n')
                text = '\n'.join(lines[1:-1])

            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}\n\nResponse:\n{response}")
