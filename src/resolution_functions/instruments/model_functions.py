from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, ClassVar, TYPE_CHECKING

import numpy as np
from numpy.polynomial.polynomial import Polynomial


if TYPE_CHECKING:
    from jaxtyping import Float
    from .instrument import InstrumentModelData, InstrumentPolynomialModelData


class InstrumentModel(ABC):
    input: ClassVar[int]
    output: ClassVar[int]

    def __init__(self, _: InstrumentModelData, setting: list[str], **__):
        self.setting = setting

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class PolynomialModel1D(InstrumentModel):
    input = 1
    output = 1

    def __init__(self, model_data: InstrumentPolynomialModelData, setting: list[str], **kwargs):
        super().__init__(model_data, setting, **kwargs)
        self.polynomial = Polynomial(model_data.get_coefficients(setting))

    def __call__(self, frequencies: Float[np.ndarray, 'frequencies'], *args, **kwargs) -> Float[np.ndarray, 'sigma']:
        return self.polynomial(frequencies)


class DiscontinuousPolynomialModel1D(InstrumentModel):
    input = 1
    output = 1

    def __init__(self, model_data: InstrumentPolynomialModelData, setting: list[str], **kwargs):
        super().__init__(model_data, setting, **kwargs)

        self.polynomial = Polynomial(model_data.get_coefficients(setting))

        self.low_energy_cutoff = model_data.get_value('low_energy_cutoff', setting[0], - np.inf)
        self.low_energy_resolution = model_data.get_value('low_energy_resolution', setting[0], 0.)

        self.high_energy_cutoff = model_data.get_value('high_energy_cutoff', setting[0], np.inf)
        self.high_energy_resolution = model_data.get_value('high_energy_resolution', setting[0], 0.)

    def __call__(self, frequencies: Float[np.ndarray, 'frequencies']) -> Float[np.ndarray, 'sigma']:
        result = self.polynomial(frequencies)

        assert np.all(result > 0)

        result[frequencies < self.low_energy_cutoff] = self.low_energy_resolution
        result[frequencies > self.high_energy_cutoff] = self.high_energy_resolution

        return result * 0.5
