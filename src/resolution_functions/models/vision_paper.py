from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np

from .model_base import InstrumentModel, ModelData

if TYPE_CHECKING:
    from jaxtyping import Float




@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class VisionPaperModelData(ModelData):
    primary_flight_path: float
    primary_flight_path_uncertainty: float
    sample_thickness: float
    detector_thickness: float
    crystal_plane_spacing: float
    d_r: float
    d_t: float
    angles: list[float]
    distance_sample_analyzer: float
    average_bragg_angle_graphite: float


class VisionPaperModel(InstrumentModel):
    """https://doi.org/10.1016/j.nima.2009.03.204"""
    input = 1
    output = 1

    data_class = VisionPaperModelData

    PLANCK = 6.626068e-34  # J s
    REDUCED_PLANCK = 1.054571817e-34  # J s
    NEUTRON_MASS = 1.67492749804e-27  # kg

    def __init__(self, model_data: VisionPaperModelData, **_):
        self.l1 = model_data.primary_flight_path
        self.d_t = model_data.d_t

        self.e0 = self.PLANCK ** 2 * 0.5 / self.NEUTRON_MASS * (0.5 / model_data.crystal_plane_spacing) ** 2
        self.nu0 = 0.5 * self.PLANCK / (self.NEUTRON_MASS * model_data.crystal_plane_spacing)
        self.one_over_l1 = 1 / self.l1
        self.distance_ratio = model_data.primary_flight_path_uncertainty * self.one_over_l1

        self.theta = np.deg2rad(model_data.average_bragg_angle_graphite)
        self.capital_t = 0.5 * 1 / np.tan(self.theta)

        self.z2 = model_data.distance_sample_analyzer
        self.capital_t_over_z2 = self.capital_t / self.z2

        self.d_a = model_data.sample_thickness ** 2 / 12
        d_b = 0.7e-6
        d_c = model_data.detector_thickness ** 2 / 12
        self.db_dc_factor = (2 * d_b + d_c)

        self.final_term = self.e0 / np.tan(self.theta) / self.z2 * model_data.d_r

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
