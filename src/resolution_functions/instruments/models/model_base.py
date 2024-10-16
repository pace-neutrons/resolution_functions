from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ModelData(ABC):
    function: str
    citation: str


class InstrumentModel(ABC):
    input: ClassVar[int]
    output: ClassVar[int]

    data_class: ClassVar[type[ModelData]]

    @abstractmethod
    def __init__(self, _: ModelData, **__):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
