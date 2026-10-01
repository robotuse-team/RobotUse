"""Google direct API policy using its OpenAI-compatible endpoint."""
from .base_provider import BaseProvider


class GeminiProvider(BaseProvider):
    key_env = 'GOOGLE_AI_STUDIO_KEY'
    endpoint = 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'

    def normalize_model(self, model):
        model = model.removeprefix('google/')
        if not model.startswith('gemini-'):
            raise ValueError('Google direct API requires a gemini-* model')
        return model

    def reasoning_levels(self, model):
        return ('low', 'medium', 'high')

    def request_fields(self, reasoning):
        if reasoning == 'default':
            return {}
        return {'reasoning_effort': reasoning}
