from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Union

import numpy as np

from ..instrument import Instrument, InstrumentModelData, ModelParameters, ModelSettings, InstrumentPolynomialModelData
from ..model_functions import PolynomialModel1D, DiscontinuousPolynomialModel1D
from .models import ToscaBookModel, VisionPaperModel


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ToscaAbINSModelData(InstrumentPolynomialModelData):
    parameters: ToscaAbINSModelParameters

    def get_coefficients(self, setting: list[str]) -> list[float]:
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
class _ToscaBase(Instrument):
    name: ClassVar[str] = 'tosca'
    model_functions = {
        'AbINS': PolynomialModel1D,
        'book': ToscaBookModel,
        'vision': VisionPaperModel,
    }


@dataclass(init=True, repr=True, frozen=True, slots=True)
class TFXA(_ToscaBase):
    models: dict[str, ToscaBookModelData]

    model_dataclasses = {'book': (ToscaBookModelData, ToscaBookModelParameters, ToscaBookSettings)}


@dataclass(init=True, repr=True, frozen=True, slots=True)
class TOSCA1(_ToscaBase):
    models: dict[str, ToscaBookModelData]

    model_dataclasses = {'book': (ToscaBookModelData, ToscaBookModelParameters, ToscaBookSettings)}


@dataclass(init=True, repr=True, frozen=True, slots=True)
class TOSCA(_ToscaBase):
    models: dict[str, Union[ToscaAbINSModelData, ToscaBookModelData, ToscaVisionModelData]]

    model_dataclasses = {
        'AbINS': (ToscaAbINSModelData, ToscaAbINSModelParameters, ModelSettings),
        'book': (ToscaBookModelData, ToscaBookModelParameters, ToscaBookSettings),
        'vision': (ToscaVisionModelData, ToscaVisionModelParameters, ToscaVisionSettings)
    }


@dataclass(init=True, repr=True, frozen=True, slots=True)
class VisionPaperModelData(InstrumentModelData):
    parameters: VisionPaperModelParameters
    settings: dict[str, VisionPaperModelSettings]


@dataclass(init=True, repr=True, frozen=True, slots=True)
class VisionPaperModelParameters(ModelParameters):
    primary_flight_path: float
    sample_thickness: float
    detector_thickness: float
    sample_width: float
    detector_width: float
    crystal_plane_spacing: float
    d_r: float
    d_t: float


@dataclass(init=True, repr=True, frozen=True, slots=True)
class VisionPaperModelSettings(ModelSettings):
    angles: list[float]
    distance_sample_analyzer: float
    average_bragg_angle_graphite: float


@dataclass(init=True, repr=True, frozen=True, slots=True)
class VISION(Instrument):
    models: dict[str, VisionPaperModelData]

    name = 'vision'
    model_dataclasses = {'vision': (VisionPaperModelData, VisionPaperModelParameters, VisionPaperModelSettings)}
    model_functions = {'vision': VisionPaperModel}


@dataclass(init=True, repr=True, frozen=True, slots=True)
class LagrangeAbINSModelData(InstrumentPolynomialModelData):
    parameters: LagrangeAbINSModelParameters
    settings: dict[str, LagrangeAbINSModelSettings]

    def get_coefficients(self, setting: list[str]) -> list[float]:
        return self.settings[setting[0]].abs_resolution


@dataclass(init=True, repr=True, frozen=True, slots=True)
class LagrangeAbINSModelParameters(ModelParameters):
    final_neutron_energy: float
    scattering_angle_range: list[float]
    angles_per_detector: int
    energy_bin_width: float


@dataclass(init=True, repr=True, frozen=True, slots=True)
class LagrangeAbINSModelSettings(ModelSettings):
    ei_range: list[float]
    abs_resolution: list[float]
    low_energy_cutoff: float = - np.inf
    low_energy_resolution: float = 0.


@dataclass(init=True, repr=True, frozen=True, slots=True)
class Lagrange(Instrument):
    models: dict[str, LagrangeAbINSModelData]

    name: ClassVar[str] = 'lagrange'
    model_dataclasses = {'AbINS': (LagrangeAbINSModelData, LagrangeAbINSModelParameters, LagrangeAbINSModelSettings)}
    model_functions = {'AbINS': DiscontinuousPolynomialModel1D}
