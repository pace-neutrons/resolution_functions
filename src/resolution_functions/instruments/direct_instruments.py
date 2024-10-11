from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Callable, TYPE_CHECKING

from .instrument import *

import numpy as np
from numpy.polynomial.polynomial import Polynomial

if TYPE_CHECKING:
    from jaxtyping import Float


class PANTHER(Instrument):
    name: ClassVar[str] = 'panther'

    @staticmethod
    def _convert_data(version_data: dict
                      ) -> tuple[InstrumentConstants, InstrumentSettings, dict[str, InstrumentModelData]]:
        constants = PantherConstants(**version_data['constants'])
        settings = InstrumentSettings()

        abins_model = version_data['models']['AbINS']
        models = {
            'AbINS': InstrumentModelData(function=abins_model['function'],
                                         citation=abins_model['citation'],
                                         constants=ModelConstants(),
                                         settings=ModelSettings(),
                                         parameters=PantherAbINSModelParameters(**abins_model['parameters']))
        }

        return constants, settings, models

    def get_resolution_function(self, model: str, setting: list[str], e_init: float, **_):
        model = self.models[model]

        if model.function == 'multiple_polynomial_ei':
            return self.create_multiple_polynomial_ei(model.parameters.abs,
                                                      model.parameters.ei_dependence,
                                                      model.parameters.ei_energy_product,
                                                      e_init)
        else:
            raise NotImplementedError()

    @staticmethod
    def create_multiple_polynomial_ei(abs: list[float],
                                      ei_dependence: list[float],
                                      ei_energy_product: list[float],
                                      e_init: float, *_, **__
                                      ) -> Callable[[Float[np.ndarray, 'frequencies']], Float[np.ndarray, 'sigma']]:
        def multiple_polynomial_ei(frequencies: Float[np.ndarray, 'frequencies']) -> Float[np.ndarray, 'sigma']:
            resolution_fwhm = (Polynomial(abs)(frequencies) +
                               Polynomial(ei_dependence)(e_init) +
                               Polynomial(ei_energy_product)(e_init * frequencies))
            return resolution_fwhm / (2 * np.sqrt(2 * np.log(2)))

        return multiple_polynomial_ei


@dataclass(init=True, repr=True, frozen=True, slots=True)
class PantherConstants(InstrumentConstants):
    q_size: int
    e_init: int


@dataclass(init=True, repr=True, frozen=True, slots=True)
class PantherAbINSModelParameters(ModelParameters):
    abs: list[float]
    ei_dependence: list[float]
    ei_energy_product: list[float]
