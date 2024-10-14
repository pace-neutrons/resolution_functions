from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

from ..model_functions import InstrumentModel

if TYPE_CHECKING:
    from jaxtyping import Float
    from .instruments import ToscaBookModelData, ToscaVisionModelData, VisionPaperModelData


class ToscaBookModel(InstrumentModel):
    input = 1
    output = 1

    REDUCED_PLANCK_SQUARED = 4.18019

    def __init__(self, model_data: ToscaBookModelData, setting: list[str], **kwargs):
        super().__init__(model_data, setting, **kwargs)
        settings = model_data.settings[setting[0]]
        params = model_data.parameters

        da = settings.average_secondary_flight_path * np.sin(np.deg2rad(settings.average_bragg_angle_graphite))

        self.time_dependent_term_factor = params.water_moderator_constant ** 2 * self.REDUCED_PLANCK_SQUARED
        self.final_energy_term_factor = (2 * settings.average_final_energy *
                                         settings.change_average_bragg_angle_graphite /
                                         np.tan(np.deg2rad(settings.average_bragg_angle_graphite)))
        self.time_dependent_term_factor += (2 * settings.average_final_energy *
                                            (params.sample_thickness ** 2 +
                                             4 * params.graphite_thickness ** 2 +
                                             params.detector_thickness ** 2) ** 0.5 / da) ** 2
        self.time_dependent_term_factor = np.sqrt(self.time_dependent_term_factor)

        self.average_final_energy = settings.average_final_energy
        self.primary_flight_path = params.primary_flight_path
        self.primary_flight_path_uncertainty = params.primary_flight_path_uncertainty
        self.average_secondary_flight_path = settings.average_secondary_flight_path
        self.average_bragg_angle = settings.average_bragg_angle_graphite
        self.time_channel_uncertainty2 = params.time_channel_uncertainty ** 2

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


class VisionPaperModel(InstrumentModel):
    """https://doi.org/10.1016/j.nima.2009.03.204"""
    input = 1
    output = 1

    PLANCK = 6.626068e-34  # J s
    REDUCED_PLANCK = 1.054571817e-34  # J s
    NEUTRON_MASS = 1.67492749804e-27  # kg

    def __init__(self,
                 model_data: Union[ToscaVisionModelData, VisionPaperModelData],
                 setting: list[str], **kwargs):
        super().__init__(model_data, setting, **kwargs)
        settings = model_data.settings[setting[0]]

        self.l1 = model_data.parameters.primary_flight_path
        self.d_t = model_data.parameters.d_t

        self.e0 = self.PLANCK ** 2 * 0.5 / self.NEUTRON_MASS * (0.5 / model_data.parameters.crystal_plane_spacing) ** 2
        self.nu0 = 0.5 * self.PLANCK / (self.NEUTRON_MASS * model_data.parameters.crystal_plane_spacing)
        self.one_over_l1 = 1 / self.l1
        self.distance_ratio = model_data.parameters.primary_flight_path_uncertainty * self.one_over_l1

        self.theta = np.deg2rad(settings.average_bragg_angle_graphite)
        self.capital_t = 0.5 * 1 / np.tan(self.theta)

        try:
            self.z2 = settings.distance_sample_analyzer
        except AttributeError:
            self.z2 = 0.5 * settings.average_secondary_flight_path * np.sin(self.theta)
        self.capital_t_over_z2 = self.capital_t / self.z2

        self.d_a = model_data.parameters.sample_thickness ** 2 / 12
        d_b = 0.7e-6
        d_c = model_data.parameters.detector_thickness ** 2 / 12
        self.db_dc_factor = (2 * d_b + d_c)

        self.final_term = self.e0 / np.tan(self.theta) / self.z2 * model_data.parameters.d_r

    def __call__(self, frequencies: Float[np.ndarray, 'frequencies']) -> Float[np.ndarray, 'sigma']:
        e1 = frequencies * self.REDUCED_PLANCK + self.e0 * (1 / np.sin(self.theta))
        z0 = self.l1 * (self.e0 / e1) ** 0.5
        one_over_z0 = 1 / z0

        sigma = self.distance_ratio - self.nu0 * self.d_t / z0
        sigma += (self.one_over_l1 + one_over_z0 + self.capital_t_over_z2) * self.d_a
        sigma += (one_over_z0 + self.capital_t_over_z2) * self.db_dc_factor
        sigma *= 2 * e1
        sigma -= self.final_term

        return sigma
