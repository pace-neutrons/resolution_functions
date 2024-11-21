from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


class InvalidInputError(Exception):
    pass


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ModelData(ABC):
    function: str
    citation: str

    @property
    def restrictions(self) -> dict[str, list[int | float]]:
        return {}

    @property
    def defaults(self) -> dict:
        return {}


class InstrumentModel(ABC):
    input: ClassVar[int]
    output: ClassVar[int]

    data_class: ClassVar[type[ModelData]]

    def __init__(self, model_data: ModelData, **_):
        self._citation = model_data.citation

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def citation(self) -> str:
        return self._citation
