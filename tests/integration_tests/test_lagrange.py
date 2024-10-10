import numpy as np
from numpy.testing import assert_allclose
import pytest

import mantid
from abins.instruments.lagrangeinstrument import LagrangeInstrument

from resolution_functions.instruments.indirect_instruments import Lagrange


WAVENUMBER_TO_MEV = 0.12398419843320028
MEV_TO_WAVENUMBER = 1 / WAVENUMBER_TO_MEV
LAGRANGE_SETTINGS = ['Cu(220)', 'Cu(331)', 'Si(311)', 'Si(111)']


@pytest.fixture
def lagrange_rf():
    return Lagrange.from_default('Lagrange')


@pytest.fixture(params=LAGRANGE_SETTINGS)
def lagrange_abins_plus_rf_abins_resolution_function(lagrange_rf, request):
    abins = LagrangeInstrument(setting=request.param + ' (Lagrange)')
    rf = lagrange_rf.get_resolution_function('AbINS', [request.param])
    return abins, rf


@pytest.mark.parametrize('frequencies',
                         [np.arange(0, 400, 10), np.arange(100, 1000, 50)])
def test_lagrange_against_abins(frequencies,
                                lagrange_abins_plus_rf_abins_resolution_function):
    abins, rf = lagrange_abins_plus_rf_abins_resolution_function

    actual = rf(frequencies)
    expected = abins.get_sigma(frequencies * MEV_TO_WAVENUMBER) * WAVENUMBER_TO_MEV

    assert_allclose(actual, expected)
