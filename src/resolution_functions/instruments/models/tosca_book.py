from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .model_base import InstrumentModel, ModelData

if TYPE_CHECKING:
    from jaxtyping import Float




@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ToscaBookModelData(ModelData):
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
    angles: list[float]
    average_secondary_flight_path: float
    average_final_energy: float
    average_bragg_angle_graphite: float
    change_average_bragg_angle_graphite: float


class ToscaBookModel(InstrumentModel):
    input = 1
    output = 1

    data_class = ToscaBookModelData

    REDUCED_PLANCK_SQUARED = 4.18019

    def __init__(self, model_data: ToscaBookModelData, **_):
        da = model_data.average_secondary_flight_path * np.sin(np.deg2rad(model_data.average_bragg_angle_graphite))

        self.time_dependent_term_factor = model_data.water_moderator_constant ** 2 * self.REDUCED_PLANCK_SQUARED
        self.final_energy_term_factor = (2 * model_data.average_final_energy *
                                         model_data.change_average_bragg_angle_graphite /
                                         np.tan(np.deg2rad(model_data.average_bragg_angle_graphite)))
        self.time_dependent_term_factor += (2 * model_data.average_final_energy *
                                            (model_data.sample_thickness ** 2 +
                                             4 * model_data.graphite_thickness ** 2 +
                                             model_data.detector_thickness ** 2) ** 0.5 / da) ** 2
        self.time_dependent_term_factor = np.sqrt(self.time_dependent_term_factor)

        self.average_final_energy = model_data.average_final_energy
        self.primary_flight_path = model_data.primary_flight_path
        self.primary_flight_path_uncertainty = model_data.primary_flight_path_uncertainty
        self.average_secondary_flight_path = model_data.average_secondary_flight_path
        self.average_bragg_angle = model_data.average_bragg_angle_graphite
        self.time_channel_uncertainty2 = model_data.time_channel_uncertainty ** 2

    def __call__(self, frequencies: Float[np.ndarray, 'frequencies'], *args, **kwargs) -> Float[np.ndarray, 'sigma']:
        ei = frequencies + self.average_final_energy

        time_dependent_term = (2 / NEUTRON_MASS) ** 0.5 * ei ** 1.5 / self.primary_flight_path
        time_dependent_term *= self.time_dependent_term_factor / (2 * NEUTRON_MASS * ei) + self.time_channel_uncertainty2

        incident_flight_term = 2 * ei / self.primary_flight_path * self.primary_flight_path_uncertainty

        final_energy_term = (self.time_dependent_term_factor *
                             (1 + self.average_secondary_flight_path / self.primary_flight_path *
                              (ei / self.average_final_energy) ** 1.5))

        final_flight_term = (2 / self.average_secondary_flight_path *
                             np.sqrt(ei ** 3 / self.average_final_energy) *
                             2 * self.primary_flight_path / np.sin(self.average_bragg_angle))

        return np.sqrt(time_dependent_term ** 2 + incident_flight_term ** 2 +
                       final_energy_term ** 2 + final_flight_term ** 2)
