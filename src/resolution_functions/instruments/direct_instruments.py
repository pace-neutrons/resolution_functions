from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Callable, TYPE_CHECKING

from .instrument import *
from .instrument_model import InstrumentModel1D

import numpy as np
from numpy.polynomial.polynomial import Polynomial

if TYPE_CHECKING:
    from jaxtyping import Float


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True)
class PANTHER(Instrument):
    constants: PantherConstants
    models: dict[str, PantherAbINSModelData]

    name: ClassVar[str] = 'panther'

    @staticmethod
    def _convert_data(version_data: dict
                      ) -> tuple[InstrumentConstants, InstrumentSettings, dict[str, InstrumentModelData]]:
        constants = PantherConstants(**version_data['constants'])
        settings = InstrumentSettings()

        abins_model = version_data['models']['AbINS']
        models = {
            'AbINS': PantherAbINSModelData(function=abins_model['function'],
                                         citation=abins_model['citation'],
                                         constants=ModelConstants(),
                                         settings=ModelSettings(),
                                         parameters=PantherAbINSModelParameters(**abins_model['parameters']))
        }

        return constants, settings, models

    def get_resolution_function(self, model: str, setting: list[str], e_init: float, **_):
        model = self.models[model]

        if model.function == 'multiple_polynomial_ei':
            return PantherAbINSModel(e_init, model.parameters)
        else:
            raise NotImplementedError()


@dataclass(init=True, repr=True, frozen=True, slots=True)
class PantherConstants(InstrumentConstants):
    q_size: int
    e_init: int


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True)
class PantherAbINSModelData(InstrumentModelData):
    parameters: PantherAbINSModelParameters


@dataclass(init=True, repr=True, frozen=True, slots=True)
class PantherAbINSModelParameters(ModelParameters):
    abs: list[float]
    ei_dependence: list[float]
    ei_energy_product: list[float]


class PantherAbINSModel(InstrumentModel1D):
    output = 1

    def __init__(self, e_init: float, model_parameters: PantherAbINSModelParameters):
        self.e_init = e_init
        self.abs = Polynomial(model_parameters.abs)
        self.ei_dependence = Polynomial(model_parameters.ei_dependence)(e_init)
        self.ei_energy_product = Polynomial(model_parameters.ei_energy_product)

    def __call__(self, frequencies: Float[np.ndarray, 'frequencies'], *args, **kwargs) -> Float[np.ndarray, 'sigma']:
        resolution = (self.abs(frequencies) +
                      self.ei_dependence +
                      self.ei_energy_product(self.ei_dependence * frequencies))
        return resolution / (2 * np.sqrt(2 * np.log(2)))

