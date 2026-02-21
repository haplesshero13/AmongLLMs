"""Token counting utilities for LongContextAgent.

Uses tiktoken for accurate token counting when available, falls back to
character-based estimation otherwise.
"""

import json
from typing import Optional

# Module-level cache: model_id -> context_length
_context_length_cache: dict[str, int] = {}

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


def count_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    """Count tokens in a messages list using tiktoken.
    
    Falls back to character-based estimate if tiktoken is not available
    or if the model is not recognized.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: Model name for encoding selection
        
    Returns:
        Estimated token count
    """
    if _TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            # Use cl100k_base as default for OpenRouter models
            enc = tiktoken.get_encoding("cl100k_base")
        
        total = 0
        for msg in messages:
            # Per-message overhead (approximate)
            total += 4
            for key, value in msg.items():
                if isinstance(value, str):
                    total += len(enc.encode(value))
                elif isinstance(value, list):
                    # Encode list as JSON string
                    total += len(enc.encode(json.dumps(value)))
        # Reply priming overhead
        total += 2
        return total
    else:
        # Fallback: character-based estimate (1 token ≈ 4 characters)
        text = json.dumps(messages)
        return len(text) // 4


def estimate_tokens_chars(text: str) -> int:
    """Rough estimate of tokens from character count.
    
    Uses the heuristic that 1 token ≈ 4 characters.
    
    Args:
        text: Input text to estimate
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


async def get_context_length(model: str, api_key: str) -> Optional[int]:
    """Fetch context_length for a model from OpenRouter's /api/v1/models endpoint.
    
    Returns None if the model is not found or fetch fails (caller handles gracefully).
    Caches results — only hits the API once per model per process.
    
    Args:
        model: Model ID (e.g., "anthropic/claude-3.5-sonnet")
        api_key: OpenRouter API key
        
    Returns:
        Context length in tokens, or None if not found
    """
    if model in _context_length_cache:
        return _context_length_cache[model]
    
    try:
        import aiohttp
        url = "https://openrouter.ai/api/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for m in data.get("data", []):
                        if m.get("id") == model:
                            val = m.get("context_length")
                            if val:
                                _context_length_cache[model] = int(val)
                                return _context_length_cache[model]
    except Exception:
        pass
    
    return None
