from __future__ import annotations

from abc import ABC, abstractmethod
import dataclasses
import os
import yaml
from typing import ClassVar, Optional, Union


INSTRUMENT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instrument_data')


class InvalidSettingError(Exception):
    pass


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True)
class Instrument(ABC):
    version: str
    models: dict[str, InstrumentModelData]
    default_settings: str
    default_model: str

    name: ClassVar[str]
    model_classes: ClassVar[dict[str, tuple[type[InstrumentModelData], type[ModelSettings], type[ModelParameters]]]]

    @classmethod
    def from_file(cls, path: str, version: Optional[str] = None):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if version is None:
            version = data['default_version']

        models = cls._convert_data(data['version'][version])

        return cls(
            version,
            models,
            data['default_settings'],
            data['default_model'],
        )

    @classmethod
    def from_default(cls, version: Optional[str] = None):
        return cls.from_file(os.path.join(INSTRUMENT_DATA_PATH, cls.name + '.yaml'), version)

    @classmethod
    def _convert_data(cls, version_data: dict) -> dict[str, InstrumentModelData]:
        models = {}
        for model_name, model_data in version_data['models'].items():
            model_data_class, model_settings_class, model_parameters_class = cls.model_classes[model_name]
            models[model_name] = model_data_class(function=model_data['function'],
                                                  citation=model_data['citation'],
                                                  settings=model_settings_class(**model_data['settings']),
                                                  parameters=model_parameters_class(**model_data['parameters'])
                                                  )
        return models

    def get_constant(self, name: str, setting: str):
        return self.settings[setting].get(name, self.constants[name])

    @abstractmethod
    def get_resolution_function(self, model: str, setting: list[str], **_):
        raise NotImplementedError()

    @property
    def available_models(self) -> list[str]:
        return list(self.models.keys())


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ModelSettings:
    pass


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ModelParameters:
    pass


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class InstrumentModelData:
    function: str
    citation: str
    settings: ModelSettings = ModelSettings()
    parameters: ModelParameters = ModelParameters()

    def get_coefficients(self) -> list[float]:
        raise NotImplementedError()