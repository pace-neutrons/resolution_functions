from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, TYPE_CHECKING

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from .model_base import InstrumentModel, ModelData

if TYPE_CHECKING:
    from jaxtyping import Float


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class PolynomialModelData(ModelData):
    fit: list[float]


class PolynomialModel1D(InstrumentModel):
    input = 1  # tuple of strings
    output = 1

    data_class: ClassVar[type[PolynomialModelData]] = PolynomialModelData

    def __init__(self, model_data: PolynomialModelData, **_):
        self.polynomial = Polynomial(model_data.fit)

    def __call__(self, frequencies: Float[np.ndarray, 'frequencies'], *args, **kwargs) -> Float[np.ndarray, 'sigma']:
        return self.polynomial(frequencies)


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class DiscontinuousPolynomialModelData(ModelData):
    fit: list[float]
    low_energy_cutoff: float = - np.inf
    low_energy_resolution: float = 0.
    high_energy_cutoff: float = np.inf
    high_energy_resolution: float = 0.


class DiscontinuousPolynomialModel1D(InstrumentModel):
    input = 1
    output = 1

    data_class: ClassVar[type[DiscontinuousPolynomialModelData]] = DiscontinuousPolynomialModelData

    def __init__(self, model_data: DiscontinuousPolynomialModelData, **_):

        self.polynomial = Polynomial(model_data.fit)

        self.low_energy_cutoff = model_data.low_energy_cutoff
        self.low_energy_resolution = model_data.low_energy_resolution

        self.high_energy_cutoff = model_data.high_energy_cutoff
        self.high_energy_resolution = model_data.high_energy_resolution

    def __call__(self, frequencies: Float[np.ndarray, 'frequencies']) -> Float[np.ndarray, 'sigma']:
        result = self.polynomial(frequencies)

        assert np.all(result > 0)

        result[frequencies < self.low_energy_cutoff] = self.low_energy_resolution
        result[frequencies > self.high_energy_cutoff] = self.high_energy_resolution

        return result * 0.5
