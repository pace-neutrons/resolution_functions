import itertools

import numpy as np
from numpy.testing import assert_allclose
import pytest

from PyChop.Instruments import Instrument as PyChopInstrument, soft_hat
from PyChop.Chop import tube_mts

from resolution_functions.instrument import Instrument
from resolution_functions.models.pychop import PyChopModel, PyChopModelData, SIGMA2FWHM

DEBUG = False

EINIT = np.arange(50, 2000, 50)
CHOPPER_FREQ = np.arange(50, 601, 50)
MATRIX = list(itertools.product(EINIT, CHOPPER_FREQ))
MATRIX_IDS = [f'e_init={ei},f={f}' for ei, f in MATRIX]


def get_fake_frequencies(e_init: float):
    return np.linspace(0, e_init, 40, endpoint=False)


@pytest.fixture(params=['A', 'B', 'S'])
def maps_data(request):
    maps = Instrument.from_default('MAPS', 'MAPS')
    rf = maps.get_model_data('PyChop_fit', chopper_package=request.param)

    pc = PyChopInstrument('MAPS', chopper=request.param)
    return rf, pc


def test_distances(maps_data: tuple[PyChopModelData, PyChopInstrument]):
    maps_data, pychop = maps_data
    x0, xa, xm = PyChopModel._get_distances(maps_data.choppers)
    expected = pychop.chopper_system.getDistances()

    assert x0 == expected[0]
    assert xa == expected[1]
    assert xm == expected[-1]


@pytest.mark.parametrize('e_init', EINIT)
def test_moderator_width_analytical(e_init, maps_data: tuple[PyChopModelData, PyChopInstrument]):
    maps_data, pychop = maps_data
    data = maps_data.moderator
    actual = PyChopModel._get_moderator_width_analytical(data['type'], data['parameters'], data['scaling_function'],
                                                         data['scaling_parameters'], e_init)
    expected = pychop.moderator.getAnalyticWidthsSquared(e_init)

    assert_allclose(actual, expected, rtol=0, atol=1e-8)


@pytest.mark.parametrize('e_init', EINIT, ids=[f'ei={ei}' for ei in EINIT])
def test_moderator_width(e_init, maps_data: tuple[PyChopModelData, PyChopInstrument]):
    maps_data, pychop = maps_data
    actual = PyChopModel.get_moderator_width_squared(maps_data.moderator, e_init)
    expected = pychop.moderator.getWidthSquared(e_init)

    assert_allclose(actual, expected, rtol=0, atol=1e-8)


@pytest.mark.parametrize('matrix', MATRIX)
def test_chopper_width(matrix, maps_data: tuple[PyChopModelData, PyChopInstrument]):
    e_init, chopper_frequency = matrix
    maps_data, pychop = maps_data

    actual = PyChopModel.get_chopper_width_squared(maps_data, True, e_init, chopper_frequency)
    pychop.chopper_system.setFrequency(chopper_frequency)
    expected = pychop.chopper_system.getWidthSquared(e_init)

    assert_allclose(actual[0], expected[0], rtol=0, atol=1e-8)

    if actual[1] is None or expected[1] is None:
        assert actual[1] == expected[1]
    else:
        assert_allclose(actual[1], expected[1], rtol=0, atol=1e-8)


def test_get_he_detector_width_squared():
    data = np.array([7.19448838, 11.06500966, 12.43517103, 3.9478169, 10.27173622,
                     2.02389188, 3.42669603, 16.25052101, 14.55653963, 10.65643823,
                     1.47904548, 0.16923375, 10.19567838, 13.56701051, 9.96840209,
                     5.01442892, 1.53962029, 12.4874124, 19.08556836, 0.23525492,
                     3.51331984, 13.66939967, 4.74033833, 6.32958761, 7.14991049,
                     14.54004725, 9.00723817, 19.77900076, 14.48882752, 11.67445804,
                     3.43512478, 14.49896383, 3.86199652, 4.40980132, 7.88499867,
                     1.05552782, 4.26679215, 12.84775443, 19.14961564, 11.86131258,
                     11.68877294, 1.41722509, 9.85780501, 12.78118287, 9.48000153,
                     17.454911, 5.27828154, 6.39579851, 5.38761666, 5.05070064,
                     65.23083872, 7.95510104, 7.11598887, 20.66733863, 21.67808584,
                     12.63920315, 53.48396713, 84.48455992, 10.73392297, 59.32004465,
                     99.60248082, 94.73063131, 81.97253032, 54.97999684, 16.76602762,
                     74.68253242, 47.86843811, 96.52613688, 15.16096189, 18.7041231])

    actual = PyChopModel._get_he_detector_width_squared(data)
    expected = np.array([tube_mts(val)[3] for val in data])

    assert_allclose(actual, expected)


@pytest.mark.parametrize('matrix', MATRIX, ids=MATRIX_IDS)
def test_get_detector_width_squared(matrix, maps_data: tuple[PyChopModelData, PyChopInstrument]):
    e_init, chopper_frequency = matrix
    maps_data, pychop = maps_data

    fake_frequencies = get_fake_frequencies(e_init)

    actual = PyChopModel._get_detector_width_squared(maps_data.detector, fake_frequencies, e_init)
    expected = np.array([pychop.detector.getWidthSquared(e_init, en) for en in fake_frequencies])

    assert_allclose(actual, expected)


def test_get_sample_width_squared(maps_data: tuple[PyChopModelData, PyChopInstrument]):
    maps_data, pychop = maps_data

    actual = PyChopModel._get_sample_width_squared(maps_data.sample)
    expected = pychop.sample.getWidthSquared()

    assert_allclose(actual, expected)


@pytest.mark.skipif(DEBUG, reason='Debugging precompute_van_var; its outputs have been temporarily changed.')
@pytest.mark.parametrize('matrix', MATRIX, ids=MATRIX_IDS)
def test_precompute_van_var(matrix, maps_data: tuple[PyChopModelData, PyChopInstrument]):
    e_init, chopper_frequency = matrix
    maps_data, pychop = maps_data
    fake_frequencies = get_fake_frequencies(e_init)

    actual = PyChopModel._precompute_van_var(maps_data, e_init, chopper_frequency, fake_frequencies)

    pychop.chopper_system.setFrequency(chopper_frequency)
    expected, _, _ = pychop.getVanVar(Ei_in=e_init, Etrans=fake_frequencies)

    assert_allclose(actual, expected)


@pytest.mark.skipif(not DEBUG, reason='Not debugging; normal version of the function only returns vsq_van')
@pytest.mark.parametrize('matrix', MATRIX, ids=MATRIX_IDS)
def test_debug_precompute_van_var(matrix, maps_data: tuple[PyChopModelData, PyChopInstrument]):
    e_init, chopper_frequency = matrix
    maps_data, pychop = maps_data
    fake_frequencies = get_fake_frequencies(e_init)

    vsq_van, tsq_moderator, tsq_chopper, tsq_jit, tsq_aperture, tsq_detector, tsq_sample = \
        PyChopModel._precompute_van_var(maps_data, e_init, chopper_frequency, fake_frequencies)

    pychop.chopper_system.setFrequency(chopper_frequency)
    expected_result, expected, _ = pychop.getVanVar(Ei_in=e_init, Etrans=fake_frequencies)

    assert_allclose(tsq_moderator, expected['moderator'])
    assert_allclose(tsq_chopper, expected['chopper'])
    assert_allclose(tsq_jit, expected['jitter'])
    assert_allclose(tsq_aperture, expected['aperture'])
    assert_allclose(tsq_detector, expected['detector'])
    assert_allclose(tsq_sample, expected['sample'])
    assert_allclose(vsq_van, expected_result)


@pytest.mark.skipif(DEBUG, reason='Debugging precompute_van_var; its outputs have been temporarily changed.')
@pytest.mark.parametrize('matrix', MATRIX, ids=MATRIX_IDS)
def test_precompute_resolution(matrix, maps_data: tuple[PyChopModelData, PyChopInstrument]):
    e_init, chopper_frequency = matrix
    maps_data, pychop = maps_data

    actual = PyChopModel._precompute_resolution(maps_data, e_init, chopper_frequency)

    pychop.chopper_system.setFrequency(chopper_frequency)
    fake_frequencies = np.linspace(0, e_init, 40, endpoint=False)
    expected_resolution = pychop.getResolution(Ei_in=e_init, Etrans=fake_frequencies)

    assert_allclose(actual[1], expected_resolution / SIGMA2FWHM)