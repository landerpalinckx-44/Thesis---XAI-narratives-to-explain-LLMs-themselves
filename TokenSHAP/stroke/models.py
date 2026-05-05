# =============================================================================
# models.py — Custom model classes used in the experiment
# =============================================================================

import sys
from pathlib import Path

# Add parent (TokenSHAP root) to path so local token_shap package is importable
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from openai import OpenAI
from token_shap.base import OpenAIModel


class OpenRouterModel(OpenAIModel):
    """
    Subclass of TokenSHAP's OpenAIModel that redirects API calls to OpenRouter.

    WHY SUBCLASS?
    TokenSHAP's built-in OpenAIModel creates an OpenAI() client pointing to
    api.openai.com by default. OpenRouter is OpenAI-compatible but lives at a
    different URL. We let the parent __init__ run (loads model name, headers,
    etc.) and then replace self.client with one pointing to OpenRouter.
    """

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name=model_name, api_key=api_key)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
