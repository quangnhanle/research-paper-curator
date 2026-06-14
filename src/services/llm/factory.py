from functools import lru_cache

from src.config import get_settings
from src.services.llm.client import ExternalLLMClient


@lru_cache(maxsize=1)
def make_llm_client() -> ExternalLLMClient:
    """
    Create and return a singleton LLM client instance.

    Returns:
        ExternalLLMClient: Configured external LLM client
    """
    settings = get_settings()
    return ExternalLLMClient(settings)
