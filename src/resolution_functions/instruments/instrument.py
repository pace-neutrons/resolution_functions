from __future__ import annotations

from collections import ChainMap
import dataclasses
import os
import yaml
from typing import Any, ClassVar, Optional, TYPE_CHECKING, Union

from .models import MODELS


INSTRUMENT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instrument_data')


class InvalidSettingError(Exception):
    pass


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True)
class Instrument:
    name: str
    version: str
    models: dict[str, dict[str, Union[str, Union[dict[str, Union[float, int, str, list[float], dict]],
                                                 dict[str, dict[str, Union[float, int, str, list[float]]]]]]]]
    default_model: str

    @classmethod
    def from_file(cls, path: str, version: Optional[str] = None):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if version is None:
            version = data['default_version']

        version_data = data['version'][version]

        return cls(
            data['name'],
            version,
            version_data['models'],
            version_data['default_model'],
        )

    @classmethod
    def from_default(cls, name: str, version: Optional[str] = None):
        return cls.from_file(os.path.join(INSTRUMENT_DATA_PATH, name.lower() + '.yaml'), version)

    def get_model_parameter(self, model_name: str, parameter_name: str, setting: str) -> Union[Any, None]:
        return self.models[model_name].get_value(parameter_name, setting)

    def get_resolution_function(self,
                                model_name: Optional[str] = None,
                                **kwargs):
        if model_name is None:
            model_name = self.default_model

        model = self.models[model_name]
        available_settings = model['settings']

        settings = []
        for setting_name, options in available_settings.items():
            kwarg = kwargs.pop(setting_name, None)
            if kwarg is None:
                kwarg = options['default_setting']

            settings.append(options[kwarg])

        model_class = MODELS[model['function']]
        return model_class(model_class.data_class(function=model['function'],
                                                  citation=model['citation'],
                                                  **ChainMap(*settings, model['parameters'])),
                           **kwargs)

    @property
    def available_models(self) -> list[str]:
        return list(self.models.keys())
