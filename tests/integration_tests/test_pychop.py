import itertools

import numpy as np
from numpy.linalg.linalg import LinAlgError
from numpy.testing import assert_allclose
import pytest

import mantid
from abins.instruments.pychop import PyChopInstrument

from resolution_functions.instrument import Instrument
from resolution_functions.models.pychop import NoTransmissionError

WAVENUMBER_TO_MEV = 0.12398419843320028
MEV_TO_WAVENUMBER = 1 / WAVENUMBER_TO_MEV

INSTRUMENTS = [[('MAPS', 'MAPS')], [('MARI', 'MARI')], [('MERLIN', 'MERLIN')]]
INSTRUMENT_SETTINGS = [['A', 'S'], ['A', 'R', 'S'], ['S']]
ENERGIES = np.arange(50, 500, 10)
CHOPPER_FREQUENCIES = np.arange(50, 601, 50)


def get_instrument_matrix(instruments, instrument_settings):
    instrument_matrix, instrument_ids = [], []
    for instr, settings in zip(instruments, instrument_settings):
        lst = list(itertools.product(instr, settings))
        instrument_matrix.extend(lst)
        instrument_ids.extend([f'{i[0]}_{s}' for i, s in lst])

    return instrument_matrix, instrument_ids


def get_ef_matrix(energies, chopper_frequencies):
    ef_matrix = list(itertools.product(energies, chopper_frequencies))
    ef_ids = [f'ei={e},f={f}' for e, f in ef_matrix]
    return ef_matrix, ef_ids


def _abins_rf_2d(name, version, setting):
    abins = PyChopInstrument(name=name, setting=setting)
    rf = Instrument.from_default(name, version)

    return abins, rf, setting


INSTRUMENT_MATRIX, INSTRUMENT_IDS = get_instrument_matrix(INSTRUMENTS, INSTRUMENT_SETTINGS)


@pytest.fixture(scope="module", params=INSTRUMENT_MATRIX, ids=INSTRUMENT_IDS)
def abins_rf_2d(request):
    (name, version), setting = request.param
    return _abins_rf_2d(name, version, setting)


EF_MATRIX, EF_IDS = get_ef_matrix(ENERGIES, CHOPPER_FREQUENCIES)


@pytest.mark.parametrize('matrix', EF_MATRIX, ids=EF_IDS)
def test_against_abins(matrix, abins_rf_2d,):
    abins, rf_2d, setting = abins_rf_2d
    _test_against_abins(abins, rf_2d, setting, matrix)


def _test_against_abins(abins, rf_2d, setting, matrix):
    energy, chopper_frequency = matrix
    frequencies = np.linspace(0, energy, 1000)

    abins.set_incident_energy(energy, 'meV')
    abins._chopper_frequency = chopper_frequency
    abins._pychop_instrument.chopper_system.setFrequency(chopper_frequency)
    abins._pychop_instrument.frequency = chopper_frequency

    try:
        abins._polyfits = {}
        expected = abins.calculate_sigma(frequencies * MEV_TO_WAVENUMBER) * WAVENUMBER_TO_MEV
    except LinAlgError:
        with pytest.raises(NoTransmissionError):
            rf_2d.get_resolution_function('PyChop_fit', chopper_package=setting, e_init=energy, chopper_frequency=chopper_frequency)
    else:
        rf = rf_2d.get_resolution_function('PyChop_fit', chopper_package=setting, e_init=energy,
                                           chopper_frequency=chopper_frequency)
        actual = rf(frequencies)

        assert_allclose(actual, expected, rtol=1e-5)


@pytest.fixture(scope='module',
                params=[(('MARI', 'MARI'), 'G'), (('MERLIN', 'MERLIN'), 'G')],
                ids=['MARI_G', 'MERLIN_G'])
def g_choppers(request):
    (name, version), setting = request.param
    return _abins_rf_2d(name, version, setting)


ef, ids = get_ef_matrix(np.arange(10, 181, 5), CHOPPER_FREQUENCIES)


@pytest.mark.parametrize('matrix', ef, ids=ids)
def test_g_choppers_against_abins(matrix, g_choppers,):
    abins, rf_2d, setting = g_choppers
    _test_against_abins(abins, rf_2d, setting, matrix)

