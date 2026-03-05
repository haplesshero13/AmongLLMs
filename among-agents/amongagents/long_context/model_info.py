"""Model info utilities for LongContextAgent.

Fetches model capabilities from the OpenRouter /api/v1/models endpoint
and caches them for the duration of the process.
"""

from dataclasses import dataclass, field
from typing import Optional

# Module-level cache: model_id -> ModelInfo
_model_info_cache: dict[str, "ModelInfo"] = {}


@dataclass
class ModelInfo:
    """Properties fetched from the OpenRouter /api/v1/models endpoint."""

    context_length: int
    max_completion_tokens: Optional[int] = None
    supports_reasoning: bool = False  # "reasoning" in supported_parameters
    supports_include_reasoning: bool = (
        False  # "include_reasoning" in supported_parameters
    )
    supported_parameters: list[str] = field(default_factory=list)


async def get_model_info(model: str, api_key: str) -> Optional[ModelInfo]:
    """Fetch info for a model from OpenRouter's /api/v1/models endpoint.

    Returns None if the model is not found or fetch fails (caller handles gracefully).
    Caches results — only hits the API once per model per process.

    Args:
        model: Model ID (e.g., "anthropic/claude-sonnet-4.6")
        api_key: OpenRouter API key

    Returns:
        ModelInfo dataclass, or None if not found
    """
    if model in _model_info_cache:
        return _model_info_cache[model]

    try:
        import aiohttp

        url = "https://openrouter.ai/api/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for m in data.get("data", []):
                        if m.get("id") == model:
                            ctx_len = m.get("context_length")
                            if not ctx_len:
                                continue

                            supported = m.get("supported_parameters", [])
                            top_provider = m.get("top_provider", {})

                            info = ModelInfo(
                                context_length=int(ctx_len),
                                max_completion_tokens=top_provider.get(
                                    "max_completion_tokens"
                                ),
                                supports_reasoning="reasoning" in supported,
                                supports_include_reasoning="include_reasoning"
                                in supported,
                                supported_parameters=supported,
                            )
                            _model_info_cache[model] = info
                            return info
    except Exception:
        pass

    return None
