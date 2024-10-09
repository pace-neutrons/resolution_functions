from collections.abc import Callable

from instruments import instrument
from model_functions import MODEL_FUNCTIONS


def get_resolution_function(instrument: instrument.Instrument,
                            model: str = None,
                            setting: str = None,
                            **user_parameters) -> Callable:
    if model is None:
        model = instrument.default_model

    if setting is None:
        setting = instrument.default_settings

    func = MODEL_FUNCTIONS[instrument.models[model]['function']]
    return func(parameters=instrument.models[model]['parameters'],
                **instrument.get_relevant_settings(model, setting),
                **user_parameters)
