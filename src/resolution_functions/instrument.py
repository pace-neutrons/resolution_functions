from __future__ import annotations

from collections import ChainMap
import dataclasses
import os
import yaml
from typing import Any, Optional, Union, TYPE_CHECKING

from .models import MODELS

if TYPE_CHECKING:
    from .models.model_base import ModelData, InstrumentModel
    from inspect import Signature

INSTRUMENT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instrument_data')

INSTRUMENT_MAP: dict[str, tuple[str, Union[None, str]]] = {
    'ARCS': ('arcs', None),
    'CNCS': ('cncs', None),
    'Lagrange': ('lagrange', None),
    'MAPS': ('maps', None),
    'MARI': ('mari', None),
    'MERLIN': ('merlin', None),
    'PANTHER': ('panther', None),
    'TFXA': ('tosca', 'TFXA'),
    'TOSCA': ('tosca', None),
    'VISION': ('vision', None),
    'SEQUOIA': ('sequoia', None),
}


class InvalidInstrumentError(Exception):
    pass


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
    def available_versions(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return data.keys()

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
        path, implied_version = cls._get_file(name)

        if version is None:
            version = implied_version

        return cls.from_file(path, version)

    @staticmethod
    def _get_file(instrument_name: str) -> tuple[str, Union[str, None]]:
        try:
            file_name, implied_version = INSTRUMENT_MAP[instrument_name]
        except KeyError:
            raise InvalidInstrumentError(
                f'"{instrument_name}" is not a valid instrument name. Only the following instruments are '
                f'supported: {list(INSTRUMENT_MAP.keys())}')

        return os.path.join(INSTRUMENT_DATA_PATH, file_name + '.yaml'), implied_version

    def get_model_data(self, model_name: Optional[str] = None, **kwargs) -> ModelData:
        out, _ = self._get_model_data(model_name, **kwargs)
        return out

    def _get_model_data(self, model_name: Optional[str] = None, **kwargs) -> tuple[ModelData, str]:
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
        return model_class.data_class(function=model['function'],
                                      citation=model['citation'],
                                      **ChainMap(*settings, model['parameters'])), model_name

    def get_resolution_function(self, model_name: Optional[str] = None, **kwargs) -> InstrumentModel:
        model_data, model_name = self._get_model_data(model_name, **kwargs)
        model_class = MODELS[self.models[model_name]['function']]

        return model_class(model_data, **kwargs)

    def get_model_signature(self, model_name: Optional[str] = None, **kwargs) -> Signature:
        from inspect import signature, Signature, Parameter
        from typing import Annotated, Literal

        model_data, model_name = self._get_model_data(model_name, **kwargs)
        model_class = MODELS[self.models[model_name]['function']]

        signature = signature(model_class)

        params = {
            'model_name': Parameter('model_name',
                                    Parameter.POSITIONAL_OR_KEYWORD,
                                    default=model_name,
                                    annotation=Optional[str])
        }

        for setting_name, options in self.models[model_name]['settings'].items():
            option_names = self._get_options(options)
            params[setting_name] = Parameter(setting_name,
                                             Parameter.KEYWORD_ONLY,
                                             default=options['default_setting'],
                                             annotation=Literal[tuple(option_names)])

        for key, value in signature.parameters.items():
            if key == 'model_data':
                continue

            args = {}
            try:
                args['default'] = model_data.defaults[key]
            except KeyError:
                pass

            try:
                args['annotation'] = Annotated[value.annotation, f'restriction={model_data.restrictions[key]}']
            except KeyError:
                pass

            params[key] = value.replace(**args, kind=Parameter.KEYWORD_ONLY)

        return Signature(parameters=list(params.values()))

    @classmethod
    def instrument_versions(cls, instrument_name: str) -> list[str]:
        path, _ = cls._get_file(instrument_name)

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return list(data.keys())

    @property
    def available_models(self) -> list[str]:
        return list(self.models.keys())

    @property
    def available_models_and_settings(self) -> dict[str, list[str]]:
        return {model_name: list(model['settings'].keys()) for model_name, model in self.models.items()}

    @property
    def all_available_models_options(self) -> dict[str, dict[str, list[str]]]:
        return {model_name: {setting: self._get_options(value) for setting, value in list(model['settings'].items())}
                for model_name, model in self.models.items()}

    def possible_settings_for_model(self, model: str) -> list[str]:
        return list(self.models[model]['settings'].keys())

    def possible_options_for_model(self, model: str) -> dict[str, list[str]]:
        return {setting: self._get_options(value) for setting, value in self.models[model]['settings'].items()}

    def possible_options_for_model_and_setting(self, model: str, setting: str) -> list[str]:
        return self._get_options(self.models[model]['settings'][setting])

    @staticmethod
    def _get_options(setting: dict[str, Union[str, dict]]) -> list[str]:
        return [value for value in setting.keys() if value != 'default_setting']

    def default_option_for_setting(self, model: str, setting: str) -> str:
        return self.models[model]['settings'][setting]['default_setting']
