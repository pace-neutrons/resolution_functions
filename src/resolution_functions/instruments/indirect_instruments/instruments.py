from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Union

from ..instrument import Instrument, InstrumentModelData, ModelParameters, ModelSettings
from ..model_functions import PolynomialModel1D, create_discontinuous_polynomial
from .models import ToscaBookModel, VisionPaperModel


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ToscaAbINSModelData(InstrumentModelData):
    parameters: ToscaAbINSModelParameters

    def get_coefficients(self) -> list[float]:
        return self.parameters.fit


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ToscaAbINSModelParameters(ModelParameters):
    fit: list[float]


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ToscaBookModelData(InstrumentModelData):
    parameters: ToscaBookModelParameters
    settings: dict[str, ToscaBookSettings]


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ToscaBookModelParameters(ModelParameters):
    primary_flight_path: float
    primary_flight_path_uncertainty: float
    water_moderator_constant: int
    time_channel_uncertainty: int
    sample_thickness: float
    graphite_thickness: float
    detector_thickness: float
    sample_width: float
    detector_width: float
    graphite_analyser_mosaic: float
    crystal_plane_spacing: float


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ToscaBookSettings(ModelSettings):
    angles: list[float]
    average_secondary_flight_path: float
    average_final_energy: float
    average_bragg_angle_graphite: float
    change_average_bragg_angle_graphite: float


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ToscaVisionModelData(InstrumentModelData):
    parameters: ToscaVisionModelParameters
    settings: dict[str, ToscaVisionSettings]


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ToscaVisionModelParameters(ModelParameters):
    primary_flight_path: float
    primary_flight_path_uncertainty: float
    sample_thickness: float
    detector_thickness: float
    crystal_plane_spacing: float
    d_r: float
    d_t: float


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ToscaVisionSettings(ModelSettings):
    angles: list[float]
    average_secondary_flight_path: float
    average_bragg_angle_graphite: float


@dataclass(init=True, repr=True, frozen=True, slots=True)
class ToscaLikeInstrument(Instrument):
    model_functions = {
        'AbINS': PolynomialModel1D,
        'book': ToscaBookModel,
        'vision': VisionPaperModel,
    }


@dataclass(init=True, repr=True, frozen=True, slots=True)
class TFXA(ToscaLikeInstrument):
    models: dict[str, ToscaBookModelData]

    name: ClassVar[str] = 'tfxa'
    model_classes = {'book': (ToscaBookModelData, ToscaBookModelParameters, ToscaBookSettings)}


@dataclass(init=True, repr=True, frozen=True, slots=True)
class TOSCA1(ToscaLikeInstrument):
    models: dict[str, ToscaBookModelData]

    name: ClassVar[str] = 'tosca1'
    model_classes = {'book': (ToscaBookModelData, ToscaBookModelParameters, ToscaBookSettings)}


@dataclass(init=True, repr=True, frozen=True, slots=True)
class TOSCA(ToscaLikeInstrument):
    models: dict[str, Union[ToscaAbINSModelData, ToscaBookModelData, ToscaVisionModelData]]

    name: ClassVar[str] = 'tosca'
    model_classes = {
        'AbINS': (ToscaAbINSModelData, ToscaAbINSModelParameters, ModelSettings),
        'book': (ToscaBookModelData, ToscaBookModelParameters, ToscaBookSettings),
        'vision': (ToscaVisionModelData, ToscaVisionModelParameters, ToscaVisionSettings)
    }


class Lagrange(Instrument):
    name: ClassVar[str] = 'lagrange'

    def get_resolution_function(self, model: str, setting: list[str], **_):
        if self.models[model]['function'] == 'discontinuous_polynomial':
            setting = self.settings[setting[0]]
            return create_discontinuous_polynomial(parameters=setting['abs_resolution'],
                                                   low_energy_cutoff=setting.get('low_energy_cutoff', -np.inf),
                                                   low_energy_resolution=setting.get('low_energy_resolution', 0.))
        else:
            raise NotImplementedError()