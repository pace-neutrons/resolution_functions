from __future__ import annotations

from abc import ABC, abstractmethod
import dataclasses
import os
import yaml
from typing import Any, ClassVar, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from .model_functions import InstrumentModel


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
    model_classes: ClassVar[dict[str, tuple[type[InstrumentModelData], type[ModelParameters], type[ModelSettings]]]]
    model_functions: ClassVar[dict[str, type[InstrumentModel]]]

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
            model_data_class, model_parameters_class, model_settings_class = cls.model_classes[model_name]
            model_settings = {name: model_settings_class(**value) for name, value in model_data['settings'].items()}

            models[model_name] = model_data_class(function=model_data['function'],
                                                  citation=model_data['citation'],
                                                  settings=model_settings,
                                                  parameters=model_parameters_class(**model_data['parameters'])
                                                  )
        return models

    def get_constant(self, name: str, setting: str):
        return self.settings[setting].get(name, self.constants[name])

    def get_resolution_function(self, model: str, setting: list[str], **kwargs):
        return self.model_functions[model](self.models[model], setting, **kwargs)

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
class InstrumentModelData(ABC):
    function: str
    citation: str
    parameters: ModelParameters = ModelParameters()
    settings: dict[str, ModelSettings] = dataclasses.field(default_factory=lambda: {'': ModelSettings()})

    def get_value(self, attribute: str, setting: str, default: Optional[Any] = None) -> Any:
        try:
            return getattr(self.settings[setting], attribute)
        except AttributeError:
            try:
                return getattr(self.parameters, attribute)
            except AttributeError:
                return default


class InstrumentPolynomialModelData(InstrumentModelData, ABC):
    @abstractmethod
    def get_coefficients(self, setting: list[str]) -> list[float]:
        raise NotImplementedError()