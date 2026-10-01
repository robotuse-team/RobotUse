"""Explicit provider routing; no credential-dependent fallback between paid APIs."""
from dataclasses import dataclass
import os

from .providers.gemini import GeminiProvider
from .providers.openrouter import OpenRouterProvider


_GOOGLE = GeminiProvider()
_OPENROUTER = OpenRouterProvider()


def _provider_policy(provider):
    return _GOOGLE if provider == 'google' else _OPENROUTER


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    reasoning: str

    @classmethod
    def resolve(cls, model=None, *, provider=None, reasoning=None, environ=None):
        env = os.environ if environ is None else environ
        provider = provider or env.get('ROBOT_LLM_PROVIDER', 'openrouter')
        if provider not in ('google', 'openrouter'):
            raise ValueError('ROBOT_LLM_PROVIDER must be google or openrouter')
        model = model or env.get('ROBOT_LLM_MODEL') or 'gemini-3.8-flash'
        policy = _provider_policy(provider)
        model = policy.normalize_model(model)
        reasoning = reasoning or env.get('ROBOT_LLM_REASONING', 'default')
        allowed = policy.reasoning_levels(model)
        if reasoning != 'default' and reasoning not in allowed:
            raise ValueError(f'{model} via {provider}: reasoning must be one of {allowed}; got {reasoning!r}')
        return cls(provider, model, reasoning)

    @property
    def key_env(self):
        return _provider_policy(self.provider).key_env

    def key(self, environ=None):
        env = os.environ if environ is None else environ
        key = env.get(self.key_env, '')
        if not key:
            raise RuntimeError(f'{self.key_env} is missing')
        return key

    @property
    def endpoint(self):
        return _provider_policy(self.provider).endpoint

    def request_fields(self):
        return _provider_policy(self.provider).request_fields(self.reasoning)

    def metadata(self):
        return {'provider': self.provider, 'model': self.model, 'endpoint': self.endpoint,
                'reasoning_effort': self.reasoning}
