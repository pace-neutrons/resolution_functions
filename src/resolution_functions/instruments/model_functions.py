from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, ClassVar, TYPE_CHECKING

import numpy as np
from numpy.polynomial.polynomial import Polynomial


if TYPE_CHECKING:
    from jaxtyping import Float
    from .instrument import InstrumentModelData


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

    def __init__(self, model_data: InstrumentModelData, setting: list[str], **kwargs):
        super().__init__(model_data, setting, **kwargs)
        self.polynomial = Polynomial(model_data.get_coefficients())

    def __call__(self, frequencies: Float[np.ndarray, 'frequencies'], *args, **kwargs) -> Float[np.ndarray, 'sigma']:
        return self.polynomial(frequencies)


def create_discontinuous_polynomial(parameters: list[float],
                                    low_energy_cutoff: float = - np.inf,
                                    low_energy_resolution: float = 0.,
                                    high_energy_cutoff: float = np.inf,
                                    high_energy_resolution: float = 0., *_, **__
                                    ) -> Callable[[Float[np.ndarray, 'frequencies']], Float[np.ndarray, 'sigma']]:
    def discontinuous_polynomial(frequencies: Float[np.ndarray, 'frequencies']) -> Float[np.ndarray, 'sigma']:
        polynomial = Polynomial(parameters)
        result = polynomial(frequencies)

        print(result)
        assert np.all(result > 0)

        result[frequencies < low_energy_cutoff] = low_energy_resolution
        result[frequencies > high_energy_cutoff] = high_energy_resolution

        return result * 0.5

    return discontinuous_polynomial
