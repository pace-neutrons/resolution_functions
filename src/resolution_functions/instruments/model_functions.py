from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import numpy as np
from numpy.polynomial.polynomial import Polynomial


if TYPE_CHECKING:
    from jaxtyping import Float


class InvalidSettingError(Exception):
    pass


def create_polynomial(parameters: list[float], *_, **__
                      ) -> Callable[[Float[np.ndarray, 'frequencies']], Float[np.ndarray, 'sigma']]:
    return Polynomial(parameters)


def create_discontinuous_polynomial(parameters: list[float],
                                    low_energy_cutoff: float = - np.inf,
                                    low_energy_resolution: float = 0.,
                                    high_energy_cutoff: float = np.inf,
                                    high_energy_resolution: float = 0., *_, **__
                                    ) -> Callable[[Float[np.ndarray, 'frequencies']], Float[np.ndarray, 'sigma']]:
    def discontinuous_polynomial(frequencies: Float[np.ndarray, 'frequencies']) -> Float[np.ndarray, 'sigma']:
        polynomial = Polynomial(parameters)
        result = polynomial(frequencies)

        print(result)
        assert np.all(result > 0)

        result[frequencies < low_energy_cutoff] = low_energy_resolution
        result[frequencies > high_energy_cutoff] = high_energy_resolution

        return result * 0.5

    return discontinuous_polynomial


def create_multiple_polynomial_ei(parameters: list[list[float]],
                                  e_init: float, *_, **__
                                  ) -> Callable[[Float[np.ndarray, 'frequencies']], Float[np.ndarray, 'sigma']]:
    def multiple_polynomial_ei(frequencies: Float[np.ndarray, 'frequencies']) -> Float[np.ndarray, 'sigma']:
        resolution_fwhm = (Polynomial(parameters[0])(frequencies) +
                           Polynomial(parameters[1])(e_init) +
                           Polynomial(parameters[2])(e_init * frequencies))
        return resolution_fwhm / (2 * np.sqrt(2 * np.log(2)))

    return multiple_polynomial_ei

def create_dummy(parameters: list[float],
                 e_init: float, *_, **__
                 ) -> Callable[[Float[np.ndarray, 'frequencies']], Float[np.ndarray, 'sigma']]:
    def dummy(frequencies: Float[np.ndarray, 'frequencies']) -> Float[np.ndarray, 'sigma']:
        return np.full_like(frequencies, e_init * parameters[0])

    return dummy
