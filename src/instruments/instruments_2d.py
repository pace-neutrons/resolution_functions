from copy import deepcopy
from typing import Callable

from jaxtyping import Array, Float
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d

from instrument import Instrument, InvalidSettingError
from model_functions import create_polynomial, create_discontinuous_polynomial


E2L = 81.8042103582802156
E2V = 437.393362604208619
SIGMA2FWHM = 2 * np.sqrt(2 * np.log(2))
SIGMA2FWHMSQ = SIGMA2FWHM**2


class Instrument2D(Instrument):
    def get_resolution_function(self, model: str, setting: list[str], e_init: float, chopper_frequency: float | None = None,
                                **_):
        if self.models[model]['function'] == '2d_fit':
            if chopper_frequency is None:
                chopper_frequency = self.constants['chopper_frequency_default']
                # TODO: chopper frequency may be a bit more complicated

            polynomial = self.precompute_resolution(model, setting, e_init, chopper_frequency)

            def resolution_2d(frequencies: Float[Array, 'frequencies']) -> Float[Array, 'sigma']:
                return polynomial(frequencies)
            return resolution_2d

        elif self.models[model]['function'] == '2d_tau':
            return self._create_tau_resolution(model, setting, e_init, chopper_frequency)

    def _create_tau_resolution(self, model: str,
                              setting: list[str],
                              e_init: float,
                              chopper_frequency: float
                               ) -> Callable[[Float[Array, 'frequencies']], Float[Array, 'sigma']]:
        params = self.models[model]['parameters']

        tsq_moderator = self.get_moderator_width(params['measured_width']['width'], e_init, params['imod']) ** 2
        tsq_chopper = self.get_chopper_width_squared(setting, True, e_init, chopper_frequency)

        l0 = self.constants['Fermi']['distance']
        l1 = self.constants['d_chopper_sample']
        l2 = self.constants['d_sample_detector']

        def resolution(frequencies: Float[Array, 'frequencies']) -> Float[Array, 'sigma']:
            e_final = frequencies - e_init
            energy_term = (e_final / e_init) ** 1.5

            term1 = (1 + (l0 + l1) / l2 * energy_term) ** 2
            term2 = (1 + l1 / l2 * energy_term) ** 2

        return resolution

    def precompute_resolution(self,
                              model: str,
                              setting: list[str],
                              e_init: float,
                              chopper_frequency: float,
                              fitting_order: int = 4) -> Polynomial:
        params = self.models[model]['parameters']

        fake_frequencies = np.linspace(0, e_init, 40, endpoint=False)
        #fake_frequencies[fake_frequencies >= e_init] = np.nan

        tsq_jit = self.settings[setting[0]]['tjit'] ** 2
        x0, xa, x1, x2, xm = self.get_distances()

        tsq_moderator = self.get_moderator_width(params['measured_width']['width'], e_init, params['imod']) ** 2
        tsq_chopper = self.get_chopper_width_squared(setting, True, e_init, chopper_frequency)

        # For Disk chopper spectrometers, the opening times of the first chopper can be the effective moderator time
        if tsq_chopper[1] is not None:
            frac_dist = 1 - (xm / x0)
            tsmeff = tsq_moderator * frac_dist ** 2  # Effective moderator time at first chopper
            x0 -= xm  # Propagate from first chopper, not from moderator (after rescaling tmod)
            tsq_moderator = tsmeff if (tsq_chopper[1] > tsmeff) else tsq_chopper[1]

        tsq_chopper = tsq_chopper[0]
        tanthm = np.tan(self.constants['theta'] * np.pi / 180.0)

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
        tsq_aperture = apefac ** 2 * (self.constants['aperture_width'] ** 2 / 12.0) * SIGMA2FWHMSQ

        vsq_van = tsq_moderator + tsq_chopper + tsq_jit + tsq_aperture
        e_final = e_init - fake_frequencies
        resolution =  (2 * E2V * np.sqrt(e_final ** 3 * vsq_van)) / x2 / SIGMA2FWHM

        return Polynomial.fit(fake_frequencies, resolution, fitting_order)

    def parse_chopper_data(self):
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

    def get_long_frequency(self, frequency: list[float]):
        frequency += self.constants['default_frequencies'][len(frequency):]
        frequency_matrix = self.constants['frequency_matrix']
        try:
            f0 = self.constants['constant_frequencies']
        except KeyError:
            f0 = np.zeros(np.shape(frequency_matrix)[0])

        return np.dot(frequency_matrix, frequency) + f0

    @staticmethod
    def get_moderator_width(measured_width: dict[str, bool | list[float]],
                            e_init: float,
                            imod: int):
        wavelengths = np.array(measured_width['wavelength'])
        idx = np.argsort(wavelengths)
        wavelengths = wavelengths[idx]
        widths = np.array(measured_width['width'])[idx]

        interpolated_width = interp1d(wavelengths, widths, kind='slinear')

        wavelength = np.sqrt(E2L / e_init)
        if wavelength >= wavelengths[0]:  # Larger than the smallest OG value
            width = interpolated_width(min([wavelength, wavelengths[-1]])) / 1e6  # Table has widths in microseconds
            return width  # in FWHM
        elif imod == 3:
            return Polynomial()
        else:
            return np.sqrt()

    def get_chopper_width_squared(self, setting: list[str], is_fermi: bool, e_init: float, chopper_frequency: float) -> tuple[float, float | None]:
        if is_fermi:
            settings = self.settings[setting[1]]
            pslit, radius, rho = settings['pslit'], settings['radius'], settings['rho']

            return self.get_fermi_width_squared(e_init, chopper_frequency, pslit, radius, rho), None
        else:
            distances, nslot, slot_ang_pos, slot_width, guide_width, radius, num_disk, idx_phase_independent, \
                default_phase = self.parse_chopper_data()
            frequencies = self.get_long_frequency([chopper_frequency])

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

    def get_distances(self) -> tuple[float, float, float, float, float]:
        choppers = list(self.settings.values())
        mod_chop = choppers[-1]['distance']
        try:
            ap_chop = choppers[-1]['aperture_distance']
        except KeyError:
            ap_chop = mod_chop

        consts = self.settings['constants']

        return mod_chop, ap_chop, consts['d_chopper_sample'], consts['d_sample_detector'], choppers[0]['distance']
