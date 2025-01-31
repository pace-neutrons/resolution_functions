import numpy as np
from numpy.testing import assert_allclose
import pytest

import mantid
from abins.instruments.panther import PantherInstrument

from resolution_functions.instrument import Instrument


WAVENUMBER_TO_MEV = 0.12398419843320028
MEV_TO_WAVENUMBER = 8.06554465


@pytest.fixture(scope="module")
def panther_rf():
    return Instrument.from_default('PANTHER', 'PANTHER')


@pytest.fixture(scope="module", params=np.arange(0, 1209, 20), ids=lambda e: f'ei={e}')
def panther_abins_plus_rf_abins_resolution_function(panther_rf, request):
    abins = PantherInstrument()
    abins.set_incident_energy(request.param, 'meV')

    rf = panther_rf.get_resolution_function('AbINS', e_init=request.param)
    return abins, rf, request.param


def test_panther_against_abins(panther_abins_plus_rf_abins_resolution_function):
    abins, rf, energy = panther_abins_plus_rf_abins_resolution_function

    frequencies = np.linspace(0, energy, 1000)

    actual = rf(frequencies)
    expected = abins.calculate_sigma(frequencies * MEV_TO_WAVENUMBER) * WAVENUMBER_TO_MEV

    assert_allclose(actual, expected)
