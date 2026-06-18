from abc import ABC, abstractmethod
from typing import AsyncGenerator, Any

class BaseAdapter(ABC):
    @abstractmethod
    async def generate(self, req: dict[str, Any]) -> dict[str, Any]:
        """
        Executes a non-streaming chat completion request.
        Returns standard OpenAI-like response dictionary.
        """
        pass

    @abstractmethod
    async def generate_stream(self, req: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """
        Executes a streaming chat completion request.
        Yields standard OpenAI-like response chunks.
        """
        pass
