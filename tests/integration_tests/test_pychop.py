import itertools

import numpy as np
from numpy.testing import assert_allclose
import pytest

import mantid
from abins.instruments.pychop import PyChopInstrument

from resolution_functions.instrument import Instrument


WAVENUMBER_TO_MEV = 0.12398419843320028
MEV_TO_WAVENUMBER = 1 / WAVENUMBER_TO_MEV

instruments = [[('MAPS', 'MAPS')],]
instrument_settings = [['A', 'S'],]
energies = np.arange(50, 500, 10)
chopper_frequencies = np.arange(50, 601, 50)

instrument_matrix, instrument_ids = [], []
for instr, settings in zip(instruments, instrument_settings):
    lst = list(itertools.product(instr, settings))
    instrument_matrix.extend(lst)
    instrument_ids.extend([f'{i[0]}_{s}' for i, s in lst])

print(instrument_matrix, instrument_ids)
print(len(instrument_matrix), len(instrument_ids))


@pytest.fixture(scope="module", params=instrument_matrix, ids=instrument_ids)
def abins_rf_2d(request):
    (name, version), setting = request.param
    abins = PyChopInstrument(name=name, setting=setting)
    rf = Instrument.from_default(name, version)

    return abins, rf, setting


ef_matrix = list(itertools.product(energies, chopper_frequencies))
ef_ids = [f'ei={e},f={f}' for e, f in ef_matrix]


@pytest.fixture(params=ef_matrix, ids=ef_ids)
def d2_abins_plus_rf_abins_resolution_function(abins_rf_2d, request):
    abins, rf_2d, setting = abins_rf_2d
    energy, chopper_frequency = request.param

    frequencies = np.linspace(0, energy, 1000)

    abins.set_incident_energy(energy, 'meV')
    abins._chopper_frequency = chopper_frequency
    abins._pychop_instrument.chopper_system.setFrequency(chopper_frequency)
    abins._pychop_instrument.frequency = chopper_frequency

    rf = rf_2d.get_resolution_function('PyChop_fit', chopper_package=setting, e_init=energy, chopper_frequency=chopper_frequency)
    return abins, rf, frequencies


def test_2d_against_abins(d2_abins_plus_rf_abins_resolution_function):
    abins, rf, frequencies = d2_abins_plus_rf_abins_resolution_function

    actual = rf(frequencies)

    abins._polyfits = {}
    expected = abins.calculate_sigma(frequencies * MEV_TO_WAVENUMBER) * WAVENUMBER_TO_MEV

    assert_allclose(actual, expected, rtol=1e-5)


# def test_polynomial(d2_abins_plus_rf_abins_resolution_function):
#     abins, rf = d2_abins_plus_rf_abins_resolution_function
#     abins._polyfits = {}
#     abins._polyfit_resolution()
#
#     abins_polynomial = list(abins._polyfits.values())
#     if len(abins_polynomial) == 1:
#         # AbINS uses np.polyfit() which has the coefficients in the opposite order to Polynomial.fit()
#         abins_polynomial = np.array(list(reversed(list(abins_polynomial[0]))))
#     else:
#         raise Exception()
#
#     rf_polynomial = rf.polynomial.coef
#
#     assert_allclose(rf_polynomial, abins_polynomial * WAVENUMBER_TO_MEV)

