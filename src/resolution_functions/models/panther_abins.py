from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from .model_base import InstrumentModel, ModelData

if TYPE_CHECKING:
    from jaxtyping import Float


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class PantherAbINSModelData(ModelData):
    abs: list[float]
    ei_dependence: list[float]
    ei_energy_product: list[float]


class PantherAbINSModel(InstrumentModel):
    input = 1
    output = 1

    data_class = PantherAbINSModelData

    def __init__(self, model_data: PantherAbINSModelData, e_init: float, **_):
        super().__init__(model_data)

        self.e_init = e_init
        self.abs = Polynomial(model_data.abs)
        self.ei_dependence = Polynomial(model_data.ei_dependence)(e_init)
        self.ei_energy_product = Polynomial(model_data.ei_energy_product)

    def __call__(self, frequencies: Float[np.ndarray, 'frequencies'], *args, **kwargs) -> Float[np.ndarray, 'sigma']:
        resolution = (self.abs(frequencies) +
                      self.ei_dependence +
                      self.ei_energy_product(self.e_init * frequencies))
        return resolution / (2 * np.sqrt(2 * np.log(2)))
