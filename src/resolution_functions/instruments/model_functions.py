from collections.abc import Callable

from jaxtyping import Array, Float
import numpy as np
from numpy.polynomial.polynomial import Polynomial


class InvalidSettingError(Exception):
    pass


def create_polynomial(parameters: list[float], *_, **__
                      ) -> Callable[[Float[Array, 'frequencies']], Float[Array, 'sigma']]:
    return Polynomial(parameters)


def create_discontinuous_polynomial(parameters: list[float],
                                    low_energy_cutoff: float = - np.inf,
                                    low_energy_resolution: float = 0.,
                                    high_energy_cutoff: float = np.inf,
                                    high_energy_resolution: float = 0., *_, **__
                                    ) -> Callable[[Float[Array, 'frequencies']], Float[Array, 'sigma']]:
    def discontinuous_polynomial(frequencies: Float[Array, 'frequencies']) -> Float[Array, 'sigma']:
        polynomial = Polynomial(parameters)
        result = polynomial(frequencies)

        assert np.all(result > 0)

        result[frequencies < low_energy_cutoff] = low_energy_resolution
        result[frequencies > high_energy_cutoff] = high_energy_resolution

        return result * 0.5

    return discontinuous_polynomial


def create_multiple_polynomial_ei(parameters: list[list[float]],
                                  e_init: float, *_, **__
                                  ) -> Callable[[Float[Array, 'frequencies']], Float[Array, 'sigma']]:
    def multiple_polynomial_ei(frequencies: Float[Array, 'frequencies']) -> Float[Array, 'sigma']:
        resolution_fwhm = (Polynomial(parameters[0])(frequencies) +
                           Polynomial(parameters[1])(e_init) +
                           Polynomial(parameters[2])(e_init * frequencies))
        return resolution_fwhm / (2 * np.sqrt(2 * np.log(2)))

    return multiple_polynomial_ei


def create_2d_polynomial(parameters: list[float],
                         e_init: float, *_, **__
                         ) -> Callable[[Float[Array, 'frequencies']], Float[Array, 'sigma']]:
    def polynomial_2d(frequencies: Float[Array, 'frequencies']) -> Float[Array, 'sigma']:
        fake_frequencies = np.linspace(0, e_init, 40, endpoint=False)
        fake_frequencies[fake_frequencies >= e_init] = np.nan


    return polynomial_2d


def create_dummy(parameters: list[float],
                 e_init: float, *_, **__
                 ) -> Callable[[Float[Array, 'frequencies']], Float[Array, 'sigma']]:
    def dummy(frequencies: Float[Array, 'frequencies']) -> Float[Array, 'sigma']:
        return np.full_like(frequencies, e_init * parameters[0])

    return dummy


def create_tosca_book(parameters: list[float] | list[list[float]],
                      parameter_indices: list[int], *_, **__
                      ) -> Callable[[Float[Array, 'frequencies']], Float[Array, 'sigma']]:
    if isinstance(parameters[0], list):
        if len(parameter_indices) == 1:
            parameters = parameters[parameter_indices[0]]
        else:
            raise InvalidSettingError('The chosen setting must point to exactly 1 set of parameters; averaging over '
                                      'parameters is not available.')

    ds, dd, dg, ddi, ws, wd, eta_g, theta_b, a, dtch, di, df, ef, theta_b, d_theta_b = parameters
    da = df * np.sin(np.deg2rad(theta_b))
    REDUCED_PLANCK_SQUARED = 4.18019

    def tosca_book(frequencies: Float[Array, 'frequencies']) -> Float[Array, 'sigma']:
        ei = frequencies + ef

        time_dependent_term = (2 / NEUTRON_MASS) ** 0.5 * ei ** 1.5 / di
        time_dependent_term *= (a ** 2 * REDUCED_PLANCK_SQUARED / (2 * NEUTRON_MASS * ei)) + dtch ** 2

        incident_flight_term = 2 * ei / di * ddi

        final_energy_term = 2 * ef * d_theta_b / np.tan(np.deg2rad(theta_b))
        final_energy_term += (2 * ef * (ds ** 2 + 4 * dg ** 2 + dd ** 2) ** 0.5 / da) ** 2
        final_energy_term = np.sqrt(final_energy_term)
        final_energy_term *= 1 + df / di * (ei / ef) ** 1.5

        final_flight_term = 2 / df * np.sqrt(ei ** 3 / ef)* 2 * di / np.sin(theta_b)

        return np.sqrt(time_dependent_term ** 2 + incident_flight_term ** 2 +
                       final_energy_term ** 2 + final_flight_term ** 2)

    return tosca_book


MODEL_FUNCTIONS = {
    'polynomial': create_polynomial,
    'discontinuous_polynomial': create_discontinuous_polynomial,
    'multiple_polynomial_ei': create_multiple_polynomial_ei,
    'tosca_book': create_tosca_book,
    'dummy': create_dummy,
}
