"""Provider policy boundary; transport and conversation state remain shared."""
from abc import ABC, abstractmethod


class BaseProvider(ABC):
    key_env: str
    endpoint: str

    @abstractmethod
    def normalize_model(self, model):
        raise NotImplementedError

    @abstractmethod
    def reasoning_levels(self, model):
        raise NotImplementedError

    @abstractmethod
    def request_fields(self, reasoning):
        raise NotImplementedError
