from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Optional, TypedDict, TYPE_CHECKING, Union

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d

from .model_base import InstrumentModel, ModelData

if TYPE_CHECKING:
    from jaxtyping import Float

E2L = 81.8042103582802156
E2V = 437.393362604208619
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
    mod_pars: list[float]
    choppers: dict[str, Chopper]
    imod: int
    measured_wavelength: list[float]
    measured_width: list[float]
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


class PyChopModel(InstrumentModel):
    input = 1
    output = 1

    data_class = PyChopModelData

    def __init__(self,
                 model_data: PyChopModelData,
                 e_init: float,
                 chopper_frequency: Optional[float] = None,
                 **kwargs):

        if chopper_frequency is None:
            chopper_frequency = model_data.chopper_frequency_default

        # TODO: chopper frequency may be a bit more complicated
        self.polynomial = self._precompute_resolution(model_data, e_init, chopper_frequency)

    def __call__(self, frequencies: Float[np.ndarray, 'frequencies'], *args, **kwargs) -> Float[np.ndarray, 'sigma']:
        return self.polynomial(frequencies)

    def _precompute_resolution(self,
                               model_data: PyChopModelData,
                               e_init: float,
                               chopper_frequency: float,
                               fitting_order: int = 4) -> Polynomial:
        fake_frequencies = np.linspace(0, e_init, 40, endpoint=False)
        #fake_frequencies[fake_frequencies >= e_init] = np.nan

        tsq_jit = model_data.tjit ** 2
        x0, xa, xm = self._get_distances(model_data.choppers)
        x1, x2 = model_data.d_chopper_sample, model_data.d_sample_detector

        tsq_moderator = self.get_moderator_width_squared(model_data.measured_width, model_data.measured_wavelength,
                                                         model_data.mod_pars, e_init, model_data.imod)
        tsq_chopper = self.get_chopper_width_squared(model_data, True, e_init, chopper_frequency)

        # For Disk chopper spectrometers, the opening times of the first chopper can be the effective moderator time
        if tsq_chopper[1] is not None:
            frac_dist = 1 - (xm / x0)
            tsmeff = tsq_moderator * frac_dist ** 2  # Effective moderator time at first chopper
            x0 -= xm  # Propagate from first chopper, not from moderator (after rescaling tmod)
            tsq_moderator = tsmeff if (tsq_chopper[1] > tsmeff) else tsq_chopper[1]

        tsq_chopper = tsq_chopper[0]
        tanthm = np.tan(model_data.theta * np.pi / 180.0)

        vi = E2V * np.sqrt(e_init)
        vf = E2V * np.sqrt(e_init - fake_frequencies)
        g1 = 1.0 - ((chopper_frequency * 2 * np.pi * tanthm / vi) * (xa + x1))
        f1 = 1.0 + (x1 / x0) * g1

        vratio = (vi / vf) ** 3

        modfac = (x1 + vratio * x2) / x0
        chpfac = 1.0 + modfac
        apefac = f1 + ((vratio * x2 / x0) * g1)

        tsq_moderator *= modfac ** 2
        tsq_chopper *= chpfac ** 2
        tsq_jit *= chpfac ** 2
        tsq_aperture = apefac ** 2 * (model_data.aperture_width ** 2 / 12.0) * SIGMA2FWHMSQ

        vsq_van = tsq_moderator + tsq_chopper + tsq_jit + tsq_aperture
        e_final = e_init - fake_frequencies
        resolution =  (2 * E2V * np.sqrt(e_final ** 3 * vsq_van)) / x2 / SIGMA2FWHM

        return Polynomial.fit(fake_frequencies, resolution, fitting_order)

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

    def get_long_frequency(self, frequency: list[float], model_data: PyChopModelData):
        frequency += model_data.default_frequencies[len(frequency):]
        frequency_matrix = np.array(model_data.frequency_matrix)
        try:
            f0 = model_data.constant_frequencies
        except AttributeError:
            f0 = np.zeros(np.shape(frequency_matrix)[0])

        return np.dot(frequency_matrix, frequency) + f0

    @staticmethod
    def get_moderator_width_squared(measured_width: list[float],
                                    measured_wavelength: list[float],
                                    mod_pars: list[float],
                                    e_init: float,
                                    imod: int):
        wavelengths = np.array(measured_wavelength)
        idx = np.argsort(wavelengths)
        wavelengths = wavelengths[idx]
        widths = np.array(measured_width)[idx]

        interpolated_width = interp1d(wavelengths, widths, kind='slinear')

        wavelength = np.sqrt(E2L / e_init)
        if wavelength >= wavelengths[0]:  # Larger than the smallest OG value
            width = interpolated_width(min([wavelength, wavelengths[-1]])) / 1e6  # Table has widths in microseconds
            return width ** 2  # in FWHM
        else:
            return PyChopModel._get_moderator_width_analytical(mod_pars, e_init, imod)

    @staticmethod
    def _get_moderator_width_analytical(mod_pars: list[float], e_init: float, imod: int) -> float:
        if imod == 0:
            return np.array(mod_pars) * 1e-3 / 1.95 / (437.392 * np.sqrt(e_init)) ** 2 * SIGMA2FWHMSQ
        elif imod == 1:
            return PyChopModel._get_moderator_width_ikeda_carpenter(*mod_pars, e_init=e_init)
        elif imod == 2:
            ei_sqrt = np.sqrt(e_init)
            delta_0, delta_G = mod_pars[0] * 1e-3, mod_pars[1] * 1e-3
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

    def get_chopper_width_squared(self,
                                  model_data: PyChopModelData,
                                  is_fermi: bool,
                                  e_init: float,
                                  chopper_frequency: float) -> tuple[float, float | None]:
        if is_fermi:
            pslit, radius, rho = model_data.pslit, model_data.radius, model_data.rho

            return self.get_fermi_width_squared(e_init, chopper_frequency, pslit, radius, rho), None
        else:
            distances, nslot, slot_ang_pos, slot_width, guide_width, radius, num_disk, idx_phase_independent, \
                default_phase = self.parse_chopper_data(chopper_parameters)
            frequencies = self.get_long_frequency([chopper_frequency], model_data)

            return self.get_other_width_squared(e_init, frequencies, distances, nslot, slot_ang_pos, slot_width,
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

