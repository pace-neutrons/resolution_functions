from __future__ import annotations

from typing import ClassVar, Callable, TYPE_CHECKING

import numpy as np

from .instrument import Instrument, InvalidSettingError
from .model_functions import create_polynomial, create_discontinuous_polynomial


if TYPE_CHECKING:
    from jaxtyping import Float


class TOSCA(Instrument):
    name: ClassVar[str] = 'tosca'

    def get_resolution_function(self, model: str, setting: list[str], **_):
        model, setting = self.models[model], setting[0]

        if model['function'] == 'polynomial':
            return create_polynomial(model['parameters'])
        elif model['function'] == 'tosca_book':
            if setting == 'All detectors':
                raise InvalidSettingError(
                    'The chosen setting must point to exactly 1 set of parameters; averaging over '
                    'parameters is not available.')
            else:
                return self._create_tosca_book(model['parameters'][setting])
        elif model['function'] == 'vision_paper':
            setting = self.settings[setting]
            theta = setting['average_bragg_angle_graphite']
            z2 = setting.get('distance_sample_analyzer', 0.5 * setting['average_secondary_flight_path'] * np.sin(theta))

            return self._create_vision_paper(self.constants['primary_flight_path'],
                                             self.constants['primary_flight_path_uncertainty'],
                                             z2,
                                             theta,
                                             self.constants['crystal_plane_spacing'],
                                             self.constants['sample_thickness'],
                                             self.constants['detector_thickness'],
                                             0,
                                             model['parameters']['d_r'])
        else:
            raise NotImplementedError()

    @staticmethod
    def _create_tosca_book(parameters: list[float] | list[list[float]], *_, **__
                           ) -> Callable[[Float[np.ndarray, 'frequencies']], Float[np.ndarray, 'sigma']]:
        ds, dd, dg, ddi, ws, wd, eta_g, theta_b, a, dtch, di, df, ef, theta_b, d_theta_b = parameters
        da = df * np.sin(np.deg2rad(theta_b))
        REDUCED_PLANCK_SQUARED = 4.18019

        def tosca_book(frequencies: Float[np.ndarray, 'frequencies']) -> Float[np.ndarray, 'sigma']:
            ei = frequencies + ef

            time_dependent_term = (2 / NEUTRON_MASS) ** 0.5 * ei ** 1.5 / di
            time_dependent_term *= (a ** 2 * REDUCED_PLANCK_SQUARED / (2 * NEUTRON_MASS * ei)) + dtch ** 2

            incident_flight_term = 2 * ei / di * ddi

            final_energy_term = 2 * ef * d_theta_b / np.tan(np.deg2rad(theta_b))
            final_energy_term += (2 * ef * (ds ** 2 + 4 * dg ** 2 + dd ** 2) ** 0.5 / da) ** 2
            final_energy_term = np.sqrt(final_energy_term)
            final_energy_term *= 1 + df / di * (ei / ef) ** 1.5

            final_flight_term = 2 / df * np.sqrt(ei ** 3 / ef) * 2 * di / np.sin(theta_b)

            return np.sqrt(time_dependent_term ** 2 + incident_flight_term ** 2 +
                           final_energy_term ** 2 + final_flight_term ** 2)

        return tosca_book

    @staticmethod
    def _create_vision_paper(l1: float, dl1: float, z2: float, theta_deg: float, l_c: float, w_s: float,
                             w_d: float, d_t: float, d_r: float, *_, **__
                             ) -> Callable[[Float[np.ndarray, 'frequencies']], Float[np.ndarray, 'sigma']]:
        """https://doi.org/10.1016/j.nima.2009.03.204"""
        PLANCK = 6.626068e-34  # J s
        REDUCED_PLANCK = 1.054571817e-34  # J s
        NEUTRON_MASS = 1.67492749804e-27  # kg

        e0 = PLANCK ** 2 * 0.5 / NEUTRON_MASS * (0.5 / l_c) ** 2
        nu0 = 0.5 * PLANCK / (NEUTRON_MASS * l_c)
        one_over_l1 = 1 / l1

        theta = np.deg2rad(theta_deg)
        capital_t = 0.5 * 1 / np.tan(theta)

        capital_t_over_z2 = capital_t / z2

        d_a = w_s ** 2 / 12
        d_b = 0.7e-6
        d_c = w_d ** 2 / 12
        db_dc_factor = (2 * d_b + d_c)

        def resolution(frequencies: Float[np.ndarray, 'frequencies']) -> Float[np.ndarray, 'sigma']:
            e1 = frequencies * REDUCED_PLANCK + e0 * (1 / np.sin(theta))
            z0 = l1 * (e0 / e1) ** 0.5
            one_over_z0 = 1 / z0

            sigma = dl1 * one_over_l1 - nu0 * d_t / z0 + (one_over_l1 + one_over_z0 + capital_t_over_z2) * d_a
            sigma += (one_over_z0 + capital_t_over_z2) * db_dc_factor
            sigma *= 2 * e1
            sigma -= e0 / np.tan(theta) / z2 * d_r

            return sigma

        return resolution


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
