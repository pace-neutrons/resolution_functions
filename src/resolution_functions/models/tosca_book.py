"""
Model for the TOSCA :term:`instrument` from the [INS-book]_.

All classes within are exposed for reference only and should not be instantiated directly. For
obtaining the :term:`resolution function` of an :term:`instrument`, please use the
`resolution_functions.instrument.Instrument.get_resolution_function` method.

.. [INS-book] PCH Mitchell, SF Parker, AJ Ramirez-Cuesta and J Tomkinson, Vibrational Spectroscopy with Neutrons With Applications in Chemistry, Biology, Materials Science and Catalysis, World Scientific Publishing Co. Pte. Ltd., Singapore, 2005.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
try:
    from warnings import deprecated
except ImportError:
    from typing_extensions import deprecated

import numpy as np

from .model_base import InstrumentModel, ModelData, DEPRECATION_MSG

if TYPE_CHECKING:
    from jaxtyping import Float


FWHM2SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ToscaBookModelData(ModelData):
    """
    Data for the `ToscaBookModel` :term:`model`.

    Attributes
    ----------
    function
        The name of the function, i.e. the alias for `PantherAbINSModel`.
    citation
        The citation for the model. Please use this to look up more details and cite the model.
    primary_flight_path
        Distance between the :term:`moderator` and the :term:`sample` in meters (m).
    primary_flight_path_uncertainty
        The uncertainty associated with the `primary_flight_path`, in meters (m).
    water_moderator_constant
        Moderator constant, in the units of $\hbar^2$.
    time_channel_uncertainty
        Time channel uncertainty in microseconds (us).
    sample_thickness
        Thickness of the :term:`sample` in meters (m).
    graphite_thickness
        Thickness of the graphite analyser in meters (m).
    detector_thickness
        Thickness of the :term:`detector` in meters (m).
    sample_width
        Width of the :term:`sample` in meters (m).
    detector_width
        Width of the :term:`detector` in meters (m).
    crystal_plane_spacing
        Distance between the layers of atoms making up the :term:`detector`, in meters (m).
    angles
        Angle between the :term:`sample` and the analyser, in degrees.
    average_secondary_flight_path
        Average length of the path from the :term:`sample` to the :term:`detector` in meters (m).
    average_final_energy
        Average energy of the neutrons hitting the :term:`detector` in meV.
    average_bragg_angle_graphite
        Average Bragg angle of the graphite analyser, in degrees.
    change_average_bragg_angle_graphite
        Uncertainty associated with `average_bragg_angle_graphite`.
    restrictions
    defaults
    """
    primary_flight_path: float
    primary_flight_path_uncertainty: float
    water_moderator_constant: int
    time_channel_uncertainty: int
    sample_thickness: float
    graphite_thickness: float
    detector_thickness: float
    sample_width: float
    detector_width: float
    crystal_plane_spacing: float
    angles: list[float]
    average_secondary_flight_path: float
    average_final_energy: float
    average_bragg_angle_graphite: float
    change_average_bragg_angle_graphite: float


class ToscaBookModel(InstrumentModel):
    """
    Model for the TOSCA :term:`instrument` from the [INS book]_.

    Models the :term:`resolution` as a function of energy transfer (frequencies) only, with the
    output model being a Gaussian. This is done by taking into account the contributions from the
    various parts of the :term:`instrument` (for more information, please see the reference).

    Parameters
    ----------
    model_data
        The data associated with the model for a given version of a given instrument.

    Attributes
    ----------
    input
        The input that the ``__call__`` method expects.
    data_class
        Reference to the `ToscaBookModelData` type.
    citation
    """
    input = ('energy_transfer',)

    data_class = ToscaBookModelData

    REDUCED_PLANCK_SQUARED = 4.18019
    NEUTRON_MASS = 1

    def __init__(self, model_data: ToscaBookModelData, **_):
        super().__init__(model_data)
        theta = np.deg2rad(model_data.average_bragg_angle_graphite)
        da = 0.5 * model_data.average_secondary_flight_path * np.sin(theta)
        dt = (model_data.sample_thickness ** 2 +
              4 * model_data.graphite_thickness ** 2 +
              model_data.detector_thickness ** 2) ** 0.5

        self.time_dependent_term_factor = (0.5 * model_data.water_moderator_constant ** 2
                                           * self.REDUCED_PLANCK_SQUARED / self.NEUTRON_MASS)

        self.final_energy_factor = (2 * model_data.average_final_energy * dt / da) ** 2
        angle_contrib = np.tan(theta * np.deg2rad(model_data.change_average_bragg_angle_graphite))
        self.final_energy_factor += (2 * model_data.average_final_energy / angle_contrib) ** 2
        self.final_energy_factor **= 0.5

        self.final_flight_factor = 2 * dt / np.sin(theta)

        self.average_final_energy = model_data.average_final_energy
        self.primary_flight_path = model_data.primary_flight_path
        self.primary_flight_path_uncertainty = model_data.primary_flight_path_uncertainty
        self.average_secondary_flight_path = model_data.average_secondary_flight_path
        self.time_channel_uncertainty2 = model_data.time_channel_uncertainty ** 2

    def get_characteristics(self, energy_transfer: Float[np.ndarray, 'energy_transfer']
                            ) -> dict[str, Float[np.ndarray, 'sigma']]:
        """
        Computes the broadening width at each value of `energy_transfer`.

        The model approximates the broadening using the Gaussian distribution, so the returned
        widths are in the form of the standard deviation (sigma) in meV.

        Parameters
        ----------
        energy_transfer
            The energy transfer in meV at which to compute the broadening.

        Returns
        -------
        characteristics
            The characteristics of the broadening function, i.e. the Gaussian width as sigma.
        """
        ei = energy_transfer + self.average_final_energy

        time_term = 2 * (2 / self.NEUTRON_MASS) ** 0.5 * ei ** 1.5 / self.primary_flight_path
        time_term *= (self.time_dependent_term_factor / ei + self.time_channel_uncertainty2) ** 0.5

        incident_flight_term = 2 * ei / self.primary_flight_path * self.primary_flight_path_uncertainty

        final_energy_term = (self.final_energy_factor *
                             (-1 - self.average_secondary_flight_path / self.primary_flight_path *
                              (ei / self.average_final_energy) ** 1.5))

        final_flight_term = (self.final_flight_factor * 2 / self.primary_flight_path *
                             np.sqrt(ei ** 3 / self.average_final_energy))

        result =  np.sqrt(time_term ** 2 + incident_flight_term ** 2 +
                          final_energy_term ** 2 + final_flight_term ** 2)
        return {'sigma': result}

    @deprecated(DEPRECATION_MSG)
    def __call__(self, frequencies: Float[np.ndarray, 'frequencies'], *args, **kwargs) -> Float[np.ndarray, 'sigma']:
        """
        Evaluates the model at given energy transfer values (`frequencies`), returning the
        corresponding Gaussian widths (sigma).

        Parameters
        ----------
        frequencies
            Energy transfer in meV. The frequencies at which to return widths.

        Returns
        -------
        sigma
            The Gaussian widths at `frequencies` as predicted by this model.
        """
        return self.get_characteristics(frequencies)['sigma']
