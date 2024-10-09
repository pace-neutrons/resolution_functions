from typing import Callable

from instrument import Instrument

from jaxtyping import Array, Float
import numpy as np
from numpy.polynomial.polynomial import Polynomial


class PANTHER(Instrument):
    def get_resolution_function(self, model: str, setting: list[str], e_init: float, **_):
        model = self.models[model]

        if model['function'] == 'multiple_polynomial_ei':
            return self.create_multiple_polynomial_ei(model['parameters']['abs'],
                                                      model['parameters']['ei_dependence'],
                                                      model['parameters']['ei_energy_product'],
                                                      e_init)
        else:
            raise NotImplementedError()

    @staticmethod
    def create_multiple_polynomial_ei(abs: list[float],
                                      ei_dependence: list[float],
                                      ei_energy_product: list[float],
                                      e_init: float, *_, **__
                                      ) -> Callable[[Float[Array, 'frequencies']], Float[Array, 'sigma']]:
        def multiple_polynomial_ei(frequencies: Float[Array, 'frequencies']) -> Float[Array, 'sigma']:
            resolution_fwhm = (Polynomial(abs)(frequencies) +
                               Polynomial(ei_dependence)(e_init) +
                               Polynomial(ei_energy_product)(e_init * frequencies))
            return resolution_fwhm / (2 * np.sqrt(2 * np.log(2)))

        return multiple_polynomial_ei
