"""Minimal fakes for the slices of the OpenAI client the code actually uses,
so the embedder and LLM wrapper are testable without any network call."""

from __future__ import annotations

from types import SimpleNamespace


class FakeEmbeddingsClient:
    """Mimics `client.embeddings.create`. Returns vectors out of order so tests
    can verify the embedder restores input order by index."""

    def __init__(self, reverse: bool = True) -> None:
        self.reverse = reverse
        self.embeddings = self

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        data = [
            SimpleNamespace(index=i, embedding=[float(i), float(len(text))])
            for i, text in enumerate(input)
        ]
        if self.reverse:
            data = list(reversed(data))
        return SimpleNamespace(data=data)


class FakeChatClient:
    """Mimics `client.chat.completions.create`, returning queued response bodies."""

    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)
        self.chat = self
        self.completions = self

    def create(self, **kwargs: object) -> SimpleNamespace:
        content = self._contents.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
