from __future__ import annotations

from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    platform: str

    @abstractmethod
    def extract_by_id(self, ids: list[str]) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def extract_by_keyword(self, **kwargs) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def extract_comments(self, source_id: str, limit: int | None = None) -> list[dict]:
        raise NotImplementedError

