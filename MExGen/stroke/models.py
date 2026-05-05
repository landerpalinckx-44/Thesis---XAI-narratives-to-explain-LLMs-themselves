# =============================================================================
# models.py — Custom model wrapper for MExGen via OpenRouter
# =============================================================================

from openai import OpenAI
from icx360.utils.model_wrappers import GeneratedOutput


class OpenRouterModel:
    """
    Adapter that makes an OpenRouter-hosted LLM compatible with MExGen's
    expected .generate() interface.

    WHY A WRAPPER?
    MExGen was built for local HuggingFace models and expects a .generate()
    method with a specific signature. OpenRouter provides a REST API instead,
    so this wrapper translates MExGen's calling convention to OpenRouter API
    calls on the inside.
    """

    def __init__(
        self,
        model_name:    str,
        api_key:       str,
        system_prompt: str  = "",
        temperature:   float = 0.0,
        max_tokens:    int   = 5,
    ):
        self.client        = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_name    = model_name
        self.system_prompt = system_prompt
        self.temperature   = temperature
        self.max_tokens    = max_tokens
    def generate(
        self,
        inputs,
        chat_template:    bool  = False,
        system_prompt:    str   = None,
        tokenizer_kwargs: dict  = {},
        text_only:        bool  = True,
        **kwargs,
    ):
        def flatten(inp):
            """MExGen sometimes passes inputs as nested lists — flatten to string."""
            if isinstance(inp, list):
                return " ".join(inp)
            return inp

        output_text = []
        for inp in inputs:
            prompt = flatten(inp)
            messages = []
            sp = system_prompt or self.system_prompt
            if sp:
                messages.append({"role": "system", "content": sp})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,  
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content
            if content is None:
                print(f"  Warning: model returned None content.")
            output_text.append(content)

        if text_only:
            return output_text
        else:
            return GeneratedOutput(
                output_ids=None,
                output_text=output_text,
                output_token_count=None,
            )

    def __call__(self, prompt: str) -> str:
        return self.generate([prompt], text_only=True)[0]
