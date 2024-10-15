from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import ClassVar, Callable, TYPE_CHECKING, Union

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
    model_dataclasses = {'AbINS': (PantherAbINSModelData, PantherAbINSModelParameters, ModelSettings)}
    model_functions = {'AbINS': PantherAbINSModel}


@dataclass(init=True, repr=True, frozen=True, slots=True)
class PyChopModelData(InstrumentModelData):
    parameters: PyChopModelParameters
    settings: dict[str, PyChopModelSettings]


@dataclass(init=True, repr=True, frozen=True, slots=True)
class PyChopModelParameters(ModelParameters):
    d_chopper_sample: float
    d_sample_detector: float
    aperture_width: float
    theta: float
    q_size: float
    e_init: float
    max_wavenumber: float
    chopper_frequency_default: float
    chopper_allowed_frequencies: list[int]
    default_frequencies: list[float]
    frequency_matrix: list[list[float]]
    choppers: dict[str, PyChopModelChopperParameters]
    imod: int
    measured_wavelength: list[float]
    measured_width: list[float]


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class PyChopModelChopperParameters:
    fermi: bool
    distance: float
    nslot: int
    slot_width: float
    slot_ang_pos: Union[list[float], None]
    guide_width: float
    radius: float
    num_disk: int
    is_phase_independent: bool
    default_phase: Union[int, str]


@dataclass(init=True, repr=True, frozen=True, slots=True)
class PyChopModelSettings(ModelSettings):
    pslit: float
    radius: float
    rho: float
    tjit: float


@dataclass(init=True, repr=True, frozen=True, slots=True)
class PyChopInstrument(Instrument, ABC):
    models: dict[str, PyChopModelData]

    model_dataclasses = {'PyChop': (PyChopModelData, PyChopModelParameters, PyChopModelSettings)}

    @classmethod
    def _convert_data(cls, version_data: dict) -> dict[str, InstrumentModelData]:
        models = {}
        for model_name, model_data in version_data['models'].items():
            model_data_class, model_parameters_class, model_settings_class = cls.model_dataclasses[model_name]

            model_settings = {name: model_settings_class(**value) for name, value in model_data['settings'].items()}

            choppers = {name: PyChopModelChopperParameters(**value)
                        for name, value in model_data['parameters'].pop('choppers').items()}
            model_parameters = model_parameters_class(choppers=choppers, **model_data['parameters'])

            models[model_name] = model_data_class(function=model_data['function'],
                                                  citation=model_data['citation'],
                                                  settings=model_settings,
                                                  parameters=model_parameters
                                                  )
        return models


@dataclass(init=True, repr=True, frozen=True, slots=True)
class MAPS(PyChopInstrument):
    name = 'maps'
