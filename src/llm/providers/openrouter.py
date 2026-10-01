"""OpenRouter model, credentials and reasoning policy."""
from .base_provider import BaseProvider


class OpenRouterProvider(BaseProvider):
    key_env = 'OPENROUTER_API_KEY'
    endpoint = 'https://openrouter.ai/api/v1/chat/completions'

    def normalize_model(self, model):
        return 'google/' + model if model.startswith('gemini-') else model

    def reasoning_levels(self, model):
        return (('low', 'medium', 'high') if 'gemini-3.8-flash' in model else
                ('none', 'minimal', 'low', 'medium', 'high', 'xhigh'))

    def request_fields(self, reasoning):
        if reasoning == 'default':
            return {}
        # Preserve provider reasoning/signatures for subsequent native tool turns.
        return {'reasoning': {'effort': reasoning, 'exclude': False}}
