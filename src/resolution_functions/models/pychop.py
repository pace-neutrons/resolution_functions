from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from math import erf
from typing import Optional, TypedDict, TYPE_CHECKING, Union

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d

from .model_base import InstrumentModel, ModelData

if TYPE_CHECKING:
    from jaxtyping import Float

E2L = 81.8042103582802156
E2V = 437.393362604208619
E2K = 0.48259640220781652
SIGMA2FWHM = 2 * np.sqrt(2 * np.log(2))
SIGMA2FWHMSQ = SIGMA2FWHM**2


@dataclass(init=True, repr=True, frozen=True, slots=True)
class PyChopModelData(ModelData):
    d_chopper_sample: float
    d_sample_detector: float
    aperture_width: float
    theta: float
    q_size: float
    e_init: float
    max_wavenumber: float
    chopper_frequency_default: float
    chopper_allowed_frequencies: list[int]
    default_frequencies: list[float]
    frequency_matrix: list[list[float]]
    choppers: dict[str, Chopper]
    moderator: Moderator
    detector: None | Detector
    sample: None | Sample
    pslit: float
    radius: float
    rho: float
    tjit: float


class Chopper(TypedDict):
    fermi: bool
    distance: float
    nslot: int
    slot_width: float
    slot_ang_pos: Union[list[float], None]
    guide_width: float
    radius: float
    num_disk: int
    is_phase_independent: bool
    default_phase: Union[int, str]


class Sample(TypedDict):
    type: int
    thickness: float
    width: float
    height: float
    gamma: float
    
    
class Detector(TypedDict):
    type: int
    phi: float
    depth: float


class Moderator(TypedDict):
    type: int
    parameters: list[float]
    scaling_function: None | str
    scaling_parameters: list[float]
    measured_wavelength: list[float]
    measured_width: list[float]


class PyChopModel(InstrumentModel):
    input = 1
    output = 1

    data_class = PyChopModelData

    def __init__(self,
                 model_data: PyChopModelData,
                 e_init: float,
                 chopper_frequency: Optional[float] = None,
                 fitting_order: Optional[int] = 4,
                 **_):

        if chopper_frequency is None:
            chopper_frequency = model_data.chopper_frequency_default

        # TODO: chopper frequency may be a bit more complicated
        fake_frequencies, resolution = self._precompute_resolution(model_data, e_init, chopper_frequency)
        self.polynomial = Polynomial.fit(fake_frequencies, resolution, fitting_order)

    def __call__(self, frequencies: Float[np.ndarray, 'frequencies'], *args, **kwargs) -> Float[np.ndarray, 'sigma']:
        return self.polynomial(frequencies)

    @classmethod
    def _precompute_resolution(cls,
                               model_data: PyChopModelData,
                               e_init: float,
                               chopper_frequency: float
                               ) -> tuple[Float[np.ndarray, 'frequency'], Float[np.ndarray, 'resolution']]:
        fake_frequencies = np.linspace(0, e_init, 40, endpoint=False)
        vsq_van = cls._precompute_van_var(model_data, e_init, chopper_frequency, fake_frequencies)
        e_final = e_init - fake_frequencies
        resolution = (2 * E2V * np.sqrt(e_final ** 3 * vsq_van)) / model_data.d_sample_detector

        return fake_frequencies, resolution / SIGMA2FWHM

    @classmethod
    def _precompute_van_var(cls,
                            model_data: PyChopModelData,
                            e_init: float,
                            chopper_frequency: float,
                            fake_frequencies: Float[np.ndarray, 'frequency'],
                            ) -> Float[np.ndarray, 'resolution']:
        tsq_jit = model_data.tjit ** 2
        x0, xa, xm = cls._get_distances(model_data.choppers)
        x1, x2 = model_data.d_chopper_sample, model_data.d_sample_detector

        tsq_moderator = cls.get_moderator_width_squared(model_data.moderator, e_init)
        tsq_chopper = cls.get_chopper_width_squared(model_data, True, e_init, chopper_frequency)

        # For Disk chopper spectrometers, the opening times of the first chopper can be the effective moderator time
        if tsq_chopper[1] is not None:
            frac_dist = 1 - (xm / x0)
            tsmeff = tsq_moderator * frac_dist ** 2  # Effective moderator time at first chopper
            x0 -= xm  # Propagate from first chopper, not from moderator (after rescaling tmod)
            tsq_moderator = tsmeff if (tsq_chopper[1] > tsmeff) else tsq_chopper[1]

        tsq_chopper = tsq_chopper[0]
        tanthm = np.tan(np.deg2rad(model_data.theta))
        omega = chopper_frequency * 2 * np.pi

        vi = E2V * np.sqrt(e_init)
        vf = E2V * np.sqrt(e_init - fake_frequencies)
        vratio = (vi / vf) ** 3

        factor = omega * (xa + x1)
        g1 = (1.0 - ((omega * tanthm / vi) * (xa + x1)))
        f1 = (1.0 + (x1 / x0) * g1) / factor
        g1 /= factor

        modfac = (x1 + vratio * x2) / x0
        chpfac = 1.0 + modfac
        apefac = f1 + ((vratio * x2 / x0) * g1)

        tsq_moderator *= modfac ** 2
        tsq_chopper *= chpfac ** 2
        tsq_jit *= chpfac ** 2
        tsq_aperture = apefac ** 2 * (model_data.aperture_width ** 2 / 12.0) * SIGMA2FWHMSQ

        vsq_van = tsq_moderator + tsq_chopper + tsq_jit + tsq_aperture

        if model_data.detector is not None:
            tsq_detector = (1. / vf) ** 2 * cls._get_detector_width_squared(model_data.detector, fake_frequencies, e_init)
            vsq_van += tsq_detector

            phi = np.deg2rad(model_data.detector['phi'])
        else:
            phi = 0.

        if model_data.sample is not None:
            g2 = (1.0 - (omega * tanthm / vi) * (x0 - xa))
            f2 = (1.0 + (x1 / x0) * g2) / factor
            g2 /= factor

            gamma = np.deg2rad(model_data.sample['gamma'])
            bb = - np.sin(gamma) / vi + np.sin(gamma - phi) / vf - f2 * np.cos(gamma)
            sample_factor = bb - (vratio * x2 / x0) * g2 * np.cos(gamma)

            tsq_sample = sample_factor ** 2 * cls._get_sample_width_squared(model_data.sample)
            vsq_van += tsq_sample

        # return vsq_van, tsq_moderator, tsq_chopper, tsq_jit, tsq_aperture, tsq_detector, tsq_sample
        return vsq_van

    def parse_chopper_data(self, chopper_parameters: dict[str, PyChopModelChopperParameters]):
        distances, nslot, slot_ang_pos, slot_width, guide_width, radius, num_disk = [], [], [], [], [], [], []
        idx_phase_independent, default_phase = [], []
        for i, chopper in enumerate(self.constants['choppers'].values()):
            distances.append(chopper['distance'])
            nslot.append(chopper['nslot'])
            slot_ang_pos.append(chopper['slot_ang_pos'])
            slot_width.append(chopper['slot_width'])
            guide_width.append(chopper['guide_width'])
            radius.append(chopper['radius'])
            num_disk.append(chopper['num_disk'])

            if not chopper['fermi'] and chopper['is_phase_independent']:
                idx_phase_independent.append(i)
                default_phase.append(chopper['default_phase'])
            else:
                idx_phase_independent.append(False)
                default_phase.append(False)

        return distances, nslot, slot_ang_pos, slot_width, guide_width, radius, num_disk, idx_phase_independent, default_phase

    @staticmethod
    def get_long_frequency(frequency: list[float], model_data: PyChopModelData):
        frequency += model_data.default_frequencies[len(frequency):]
        frequency_matrix = np.array(model_data.frequency_matrix)
        try:
            f0 = model_data.constant_frequencies
        except AttributeError:
            f0 = np.zeros(np.shape(frequency_matrix)[0])

        return np.dot(frequency_matrix, frequency) + f0

    @classmethod
    def get_moderator_width_squared(cls,
                                    moderator_data: Moderator,
                                    e_init: float,):
        # TODO: Sort the data in the yaml file and remove sorting below
        wavelengths = np.array(moderator_data['measured_wavelength'])
        idx = np.argsort(wavelengths)
        wavelengths = wavelengths[idx]
        widths = np.array(moderator_data['measured_width'])[idx]

        interpolated_width = interp1d(wavelengths, widths, kind='slinear')

        wavelength = np.sqrt(E2L / e_init)
        if wavelength >= wavelengths[0]:  # Larger than the smallest OG value
            width = interpolated_width(min([wavelength, wavelengths[-1]])) / 1e6  # Table has widths in microseconds
            return width ** 2  # in FWHM
        else:
            return cls._get_moderator_width_analytical(moderator_data['type'],
                                                       moderator_data['parameters'],
                                                       moderator_data['scaling_function'],
                                                       moderator_data['scaling_parameters'],
                                                       e_init)

    @staticmethod
    def _get_moderator_width_analytical(imod: int,
                                        mod_pars: list[float],
                                        scaling_function: str | None,
                                        scaling_parameters: list[float],
                                        e_init: float) -> float:
        # TODO: Look into composition
        if imod == 0:
            return np.array(mod_pars) * 1e-3 / 1.95 / (437.392 * np.sqrt(e_init)) ** 2 * SIGMA2FWHMSQ
        elif imod == 1:
            return PyChopModel._get_moderator_width_ikeda_carpenter(*mod_pars, e_init=e_init)
        elif imod == 2:
            ei_sqrt = np.sqrt(e_init)
            delta_0, delta_G = mod_pars[0] * 1e-3, mod_pars[1] * 1e-3

            if scaling_function is not None:
                func = MODERATOR_MODIFICATION_FUNCTIONS[scaling_function]
                delta_0 *= func(e_init, scaling_parameters)

            return ((delta_0 + delta_G * ei_sqrt) / 1.96 / (437.392 * ei_sqrt)) ** 2 * SIGMA2FWHMSQ
        elif imod == 3:
            return Polynomial(mod_pars)(np.sqrt(E2L / e_init)) ** 2 * 1e-12
        else:
            raise NotImplementedError()

    @staticmethod
    def _get_moderator_width_ikeda_carpenter(s1: float, s2: float, b1: float, b2: float, e_mod: float, e_init: float):
        sig = np.sqrt(s1 ** 2 + s2 ** 2 * 81.8048 / e_init)
        a = 4.37392e-4 * sig * np.sqrt(e_init)
        b = b2 if e_init > 130. else b1
        r = np.exp(- e_init / e_mod)
        return (3. / a ** 2 + (r * (2. - r)) / b ** 2) * 1e-12 * SIGMA2FWHMSQ

    @classmethod
    def get_chopper_width_squared(cls,
                                  model_data: PyChopModelData,
                                  is_fermi: bool,
                                  e_init: float,
                                  chopper_frequency: float) -> tuple[float, float | None]:
        if is_fermi:
            pslit, radius, rho = model_data.pslit, model_data.radius, model_data.rho

            return cls.get_fermi_width_squared(e_init, chopper_frequency, pslit, radius, rho), None
        else:
            distances, nslot, slot_ang_pos, slot_width, guide_width, radius, num_disk, idx_phase_independent, \
                default_phase = cls.parse_chopper_data(chopper_parameters)
            frequencies = cls.get_long_frequency([chopper_frequency], model_data)

            return cls.get_other_width_squared(e_init, frequencies, distances, nslot, slot_ang_pos, slot_width,
                                                guide_width, radius, num_disk, default_phase, idx_phase_independent)

    @staticmethod
    def get_fermi_width_squared(e_init: float,
                                chopper_frequency: float,
                                pslit: float,
                                radius: float,
                                rho: float) -> float:
        chopper_frequency *= 2 * np.pi
        gamm = (2.00 * radius ** 2 / pslit) * abs(1.00 / rho - 2.00 * chopper_frequency / (437.392 * np.sqrt(e_init)))

        if gamm >= 4.:
            # TODO: Log warning
            return np.nan
        elif gamm <= 1.:
            gsqr = (1.00 - (gamm ** 2) ** 2 / 10.00) / (1.00 - (gamm ** 2) / 6.00)
        else:
            groot = np.sqrt(gamm)
            gsqr = 0.60 * gamm * ((groot - 2.00) ** 2) * (groot + 8.00) / (groot + 4.00)

        sigma =  ((pslit / (2.00 * radius * chopper_frequency)) ** 2 / 6.00) * gsqr
        return sigma * SIGMA2FWHMSQ

    @staticmethod
    def get_other_width_squared(e_init: float,
                                frequencies: list[float],
                                chopper_distances: list[float],
                                nslot: list[int | None],
                                slots_ang_pos: list[list[float] | None],
                                slot_widths: list[float],
                                guide_widths: list[float],
                                radii: list[float],
                                num_disk: list[int],
                                phase: list[str | int | bool],
                                idx_phase_independent: list[int | bool],
                                source_rep: float = 50,
                                n_frame: int = 1,
                                ) -> tuple[float, float]:
        # conversion factors
        lam2TOF = 252.7784  # the conversion from wavelength to TOF at 1m, multiply by distance
        uSec = 1e6  # seconds to microseconds
        lam = np.sqrt(81.8042 / e_init)  # convert from energy to wavelenth

        # if there's only one disk we prepend a dummy disk with full opening at zero distance
        # so that the distance calculations (which needs the difference between disk pos) works
        if len(chopper_distances) == 1:
            for lst, prepend in zip([chopper_distances, nslot, slots_ang_pos, slot_widths, guide_widths, radii, num_disk],
                                    [0, 1, None, 3141, 10, 500, 1]):
                lst.insert(0, prepend)

            prepend_disk = True
        else:
            prepend_disk = False

        p_frames = source_rep / n_frame

        if prepend_disk:
            frequencies = np.array([source_rep, frequencies[0]])

        chop_times = []

        # first we optimise on the main Ei
        for i, (freq, distance, slot, angles, slot_width, guide_width, radius, n_disk, this_phase, ph_ind) in (
                enumerate(zip(frequencies, chopper_distances, nslot, slots_ang_pos, slot_widths, guide_widths, radii, num_disk, phase, idx_phase_independent))):
            # loop over each chopper
            # checks whether this chopper should have an independently set phase / delay
            islt = int(this_phase) if (ph_ind and isinstance(this_phase, str)) else 0

            if ph_ind and not isinstance(this_phase, str):
                # effective chopper velocity (if 2 disks effective velocity is double)
                chopVel = 2 * np.pi * radius * n_disk * freq
                # full opening time
                t_full_op = uSec * (slot_width + guide_width) / chopVel
                realTimeOp = np.array([this_phase, this_phase + t_full_op])
            else:
                # the opening time of the chopper so that it is open for the focus wavelength
                t_open = lam2TOF * lam * distance
                # effective chopper velocity (if 2 disks effective velocity is double)
                chopVel = 2 * np.pi * radius * n_disk * freq
                # full opening time
                t_full_op = uSec * (slot_width + guide_width) / chopVel
                # set the chopper phase to be as close to zero as possible
                realTimeOp = np.array([(t_open - t_full_op / 2.0), (t_open + t_full_op / 2.0)])

            chop_times.append([])
            if slots_ang_pos and slot > 1 and angles:
                tslots = [(uSec * angles[j] / 360.0 / freq) for j in range(slot)]
                tslots = [[(t + r * (uSec / freq)) - tslots[0] for r in range(int(freq / p_frames))] for t in
                          tslots]
                realTimeOp -= np.max(tslots[islt % slot])
                islt = 0
                next_win_t = uSec / source_rep + (uSec / freq)

                while realTimeOp[0] < next_win_t:
                    chop_times[i].append(deepcopy(realTimeOp))
                    slt0 = islt % slot
                    slt1 = (islt + 1) % slot
                    angdiff = angles[slt1] - angles[slt0]
                    if (slt1 - slt0) != 1:
                        angdiff += 360
                    realTimeOp += uSec * (angdiff / 360.0) / freq
                    islt += 1
            else:
                # If angular positions of slots not defined, assumed evenly spaced (LET, MERLIN)
                next_win_t = uSec / (slot * freq)
                realTimeOp -= next_win_t * np.ceil(realTimeOp[0] / next_win_t)

                while realTimeOp[0] < (uSec / p_frames + next_win_t):
                    chop_times[i].append(deepcopy(realTimeOp))
                    realTimeOp += next_win_t

        wd0 = (chop_times[1][1] - chop_times[1][0]) / 2.0 / 1.0e6
        wd1 = (chop_times[0][1] - chop_times[0][0]) / 2.0 / 1.0e6

        return wd0 ** 2, wd1 ** 2

    @staticmethod
    def _get_distances(choppers: dict[str, Chopper]) -> tuple[float, float, float]:
        choppers = list(choppers.values())
        mod_chop = choppers[-1]['distance']
        try:
            ap_chop = choppers[-1]['aperture_distance']
        except AttributeError:
            ap_chop = mod_chop

        return mod_chop, ap_chop, choppers[0]['distance']

    @classmethod
    def _get_detector_width_squared(cls, detector_data: Detector,
                                    fake_frequencies: Float[np.ndarray, 'frequency'],
                                    e_init: float):
        wfs = np.sqrt(E2K * (e_init - fake_frequencies))
        t2rad = 0.063
        atms = 10.
        const = 50.04685368

        if detector_data['type'] == 1:
            raise NotImplementedError()
        else:
            rad = detector_data['depth'] * 0.5
            reff = rad * (1.0 - t2rad)
            var = 2.0 * (rad * (1.0 - t2rad)) * (const * atms)
            alf = var / wfs

            assert not np.any(alf < 0.)

            return cls._get_he_detector_width_squared(alf) * reff ** 2 * SIGMA2FWHMSQ

    @staticmethod
    def _get_he_detector_width_squared(alf: Float[np.ndarray, 'ALF']) -> Float[np.ndarray, 'ALF']:
        out = np.zeros(np.shape(alf))
        coefficients_low = [0.613452291529095, -0.3621914072547197, 6.0117947617747081e-02,
                  1.8037337764424607e-02, -1.4439005957980123e-02, 3.8147446724517908e-03, 1.3679160269450818e-05,
                  -3.7851338401354573e-04, 1.3568342238781006e-04, -1.3336183765173537e-05, -7.5468390663036011e-06,
                  3.7919580869305580e-06, -6.4560788919254541e-07, -1.0509789897250599e-07, 9.0282233408123247e-08,
                  -2.1598200223849062e-08, -2.6200750125049410e-10, 1.8693270043002030e-09, -6.0097600840247623e-10,
                  4.7263196689684150e-11, 3.3052446335446462e-11, -1.4738090470256537e-11, 2.1945176231774610e-12,
                  4.7409048908875206e-13, -3.3502478569147342e-13]

        coefficients_high = [0.9313232069059375, 7.5988886169808666e-02, -8.3110620384910993e-03,
                  1.1236935254690805e-03, -1.0549380723194779e-04, -3.8256672783453238e-05, 2.2883355513325654e-05,
                  -2.4595515448511130e-06, -2.2063956882489855e-06, 7.2331970290773207e-07, 2.2080170614557915e-07,
                  -1.2957057474505262e-07, -2.9737380539129887e-08, 2.2171316129693253e-08, 5.9127004825576534e-09,
                  -3.7179338302495424e-09, -1.4794271269158443e-09, 5.5412448241032308e-10, 3.8726354734119894e-10,
                  -4.6562413924533530e-11, -9.2734525614091013e-11, -1.1246343578630302e-11, 1.6909724176450425e-11,
                  5.6146245985821963e-12, -2.7408274955176282e-12]

        g0 = (32.0 - 3.0 * (np.pi ** 2)) / 48.0
        g1 = 14.0 / 3.0 - (np.pi ** 2) / 8.0

        chebyshev_low = np.polynomial.Chebyshev(coefficients_low, [0., 10.])
        chebyshev_high = np.polynomial.Chebyshev(coefficients_high, [-1., 1.])

        first_indices = alf <= 9.
        last_indices = alf >= 10.
        mid_indices = np.logical_not(np.logical_or(first_indices, last_indices))

        out[first_indices] = 0.25 * chebyshev_low(alf[first_indices])
        out[last_indices] = g0 + g1 * chebyshev_high(1. - 18. / alf[last_indices]) / alf[last_indices] ** 2

        mid_alf = alf[mid_indices]
        guess1 = 0.25 * chebyshev_low(mid_alf)
        guess2 = g0 + g1 * chebyshev_high(1. - 18. / mid_alf) / mid_alf ** 2
        out[mid_indices] = (10. - mid_alf) * guess1 + (mid_alf - 9.) * guess2

        return out

    @staticmethod
    def _get_sample_width_squared(sample_data: Sample) -> float:
        scaling_factor = 0.125 if sample_data['type'] == 2 else 1 / 12
        return sample_data['width'] ** 2 * scaling_factor * SIGMA2FWHMSQ

    # def _create_tau_resolution(self, model: str,
    #                           setting: list[str],
    #                           e_init: float,
    #                           chopper_frequency: float
    #                            ) -> Callable[[Float[np.ndarray, 'frequencies']], Float[np.ndarray, 'sigma']]:
    #     model_data = self.models[model]['parameters']
    #
    #     tsq_moderator = self.get_moderator_width(params['measured_width'], e_init, params['imod']) ** 2
    #     tsq_chopper = self.get_chopper_width_squared(setting, True, e_init, chopper_frequency)
    #
    #     l0 = self.constants['Fermi']['distance']
    #     l1 = self.constants['d_chopper_sample']
    #     l2 = self.constants['d_sample_detector']
    #
    #     def resolution(frequencies: Float[np.ndarray, 'frequencies']) -> Float[np.ndarray, 'sigma']:
    #         e_final = frequencies - e_init
    #         energy_term = (e_final / e_init) ** 1.5
    #
    #         term1 = (1 + (l0 + l1) / l2 * energy_term) ** 2
    #         term2 = (1 + l1 / l2 * energy_term) ** 2
    #
    #     return resolution


def soft_hat(x, p):
    """
    ! Soft hat function, from Herbert subroutine library.
    ! For rescaling t-mod at low energy to account for broader moderator term
    """
    x = np.array(x)
    sig2fwhh = np.sqrt(8 * np.log(2))
    height, grad, x1, x2 = tuple(p[:4])
    sig1, sig2 = tuple(np.abs(p[4:6] / sig2fwhh))
    # linearly interpolate sig for x1<x<x2
    sig = ((x2 - x) * sig1 - (x1 - x) * sig2) / (x2 - x1)
    if np.shape(sig):
        sig[x < x1] = sig1
        sig[x > x2] = sig2
    # calculate blurred hat function with gradient
    e1 = (x1 - x) / (np.sqrt(2) * sig)
    e2 = (x2 - x) / (np.sqrt(2) * sig)
    y = (erf(e2) - erf(e1)) * ((height + grad * (x - (x2 + x1) / 2)) / 2)
    y = y + 1
    return y


MODERATOR_MODIFICATION_FUNCTIONS = {
    'soft_hat': soft_hat
}
