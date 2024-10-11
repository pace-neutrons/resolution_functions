from __future__ import annotations

import dataclasses
import os
import yaml
from typing import ClassVar, Optional, Union


INSTRUMENT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instrument_data')


class InvalidSettingError(Exception):
    pass


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True)
class Instrument:
    version: str
    constants: InstrumentConstants
    settings: InstrumentSettings
    models: dict[str, InstrumentModelData]
    default_settings: str
    default_model: str

    name: ClassVar[str]

    @classmethod
    def from_file(cls, path: str, version: Optional[str] = None):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if version is None:
            version = data['default_version']

        constants, settings, models = cls._convert_data(data['version'][version])

        return cls(
            version,
            constants,
            settings,
            models,
            data['default_settings'],
            data['default_model'],
        )

    @classmethod
    def from_default(cls, version: Optional[str] = None):
        return cls.from_file(os.path.join(INSTRUMENT_DATA_PATH, cls.name + '.yaml'), version)

    @staticmethod
    def _convert_data(version_data: dict) -> tuple[InstrumentConstants, InstrumentSettings, dict[str, InstrumentModelData]]:
        raise NotImplementedError()

    def get_constant(self, name: str, setting: str):
        return self.settings[setting].get(name, self.constants[name])

    def get_resolution_function(self, model: str, setting: list[str], **_):
        raise NotImplementedError()

    @property
    def available_models(self) -> list[str]:
        return list(self.models.keys())


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True)
class InstrumentConstants:
    pass


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True)
class InstrumentSettings:
    pass


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True)
class InstrumentModelData:
    function: str
    citation: str
    constants: ModelConstants
    settings: ModelSettings
    parameters: ModelParameters


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True)
class ModelConstants:
    pass


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True)
class ModelSettings:
    pass


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True)
class ModelParameters:
    pass

