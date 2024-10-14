from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Callable, TYPE_CHECKING

from .instrument import Instrument, InstrumentModelData, ModelParameters, ModelSettings
from .model_functions import InstrumentModel

import numpy as np
from numpy.polynomial.polynomial import Polynomial

if TYPE_CHECKING:
    from jaxtyping import Float


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class PantherAbINSModelData(InstrumentModelData):
    parameters: PantherAbINSModelParameters


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class PantherAbINSModelParameters(ModelParameters):
    abs: list[float]
    ei_dependence: list[float]
    ei_energy_product: list[float]


class PantherAbINSModel(InstrumentModel):
    output = 1

    def __init__(self, model_data: PantherAbINSModelData, setting: list[str], e_init: float, **_):
        super().__init__(model_data, setting)

        self.e_init = e_init
        self.abs = Polynomial(model_data.parameters.abs)
        self.ei_dependence = Polynomial(model_data.parameters.ei_dependence)(e_init)
        self.ei_energy_product = Polynomial(model_data.parameters.ei_energy_product)

    def __call__(self, frequencies: Float[np.ndarray, 'frequencies'], *args, **kwargs) -> Float[np.ndarray, 'sigma']:
        resolution = (self.abs(frequencies) +
                      self.ei_dependence +
                      self.ei_energy_product(self.ei_dependence * frequencies))
        return resolution / (2 * np.sqrt(2 * np.log(2)))


@dataclass(init=True, repr=True, frozen=True, slots=True)
class PANTHER(Instrument):
    models: dict[str, PantherAbINSModelData]

    name: ClassVar[str] = 'panther'
    model_classes = {'AbINS': (PantherAbINSModelData, PantherAbINSModelParameters, ModelSettings)}
    model_functions = {'AbINS': PantherAbINSModel}

